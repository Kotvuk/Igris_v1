"""
IGRIS Security System — Access levels, command filtering, PIN auth,
brute-force protection, rate limiting, and audit logging.
"""

import asyncio
import hashlib
import logging
import re
import time
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Any, Optional

import bcrypt

logger = logging.getLogger("igris.security")


class AccessLevel(IntEnum):
    READ_ONLY = 0
    SAFE = 1
    STANDARD = 2
    ELEVATED = 3
    FULL = 4


ACCESS_LEVEL_NAMES = {
    AccessLevel.READ_ONLY: "ReadOnly",
    AccessLevel.SAFE: "Safe",
    AccessLevel.STANDARD: "Standard",
    AccessLevel.ELEVATED: "Elevated",
    AccessLevel.FULL: "Full",
}

# Commands that require at least ELEVATED access
DANGEROUS_PATTERNS = [
    r"rm\s+-rf",
    r"del\s+/[sfq]",
    r"format\s+\w:",
    r"mkfs\.",
    r"dd\s+if=",
    r"shutdown",
    r"reboot",
    r":(){ :\|:& };:",
    r">\s*/dev/sd[a-z]",
    r"chmod\s+-R\s+777",
    r"reg\s+delete",
    r"net\s+user\s+\w+\s+/delete",
    r"netsh\s+advfirewall\s+set\s+allprofiles\s+state\s+off",
]

# Commands blocked entirely
BLOCKED_PATTERNS = [
    r":(){ :\|:& };:",  # fork bomb
    r">\s*/dev/sd[a-z]",  # disk overwrite
    r"mkfs\.\w+\s+/dev/sd[a-z]",  # format disk
    r"dd\s+if=/dev/(zero|random)\s+of=/dev/sd[a-z]",
]

# Safe commands that don't need confirmation
SAFE_COMMANDS = [
    r"^(ls|dir|pwd|cd|echo|cat|type|head|tail|wc|date|whoami|hostname)(\s|$)",
    r"^(python|node|git|pip|npm|cargo)\s+(--version|-V|version)$",
    r"^git\s+(status|log|diff|branch)",
    r"^(find|where|which)\s+",
]


@dataclass
class AuthState:
    pin_hash: Optional[bytes] = None
    current_level: AccessLevel = AccessLevel.SAFE
    failed_attempts: int = 0
    lockout_until: float = 0.0
    last_activity: float = field(default_factory=time.time)
    inactivity_timeout: int = 300  # 5 minutes to auto-lock


@dataclass
class RateLimiter:
    max_per_minute: int = 60
    _timestamps: list[float] = field(default_factory=list)

    def check(self) -> bool:
        now = time.time()
        self._timestamps = [t for t in self._timestamps if now - t < 60]
        if len(self._timestamps) >= self.max_per_minute:
            return False
        self._timestamps.append(now)
        return True

    @property
    def remaining(self) -> int:
        now = time.time()
        self._timestamps = [t for t in self._timestamps if now - t < 60]
        return max(0, self.max_per_minute - len(self._timestamps))


class SecurityManager:
    """Central security manager for IGRIS."""

    MAX_FAILED_ATTEMPTS = 3
    LOCKOUT_DURATION = 300  # 5 minutes

    def __init__(self, db: Any, default_level: AccessLevel = AccessLevel.SAFE):
        self._db = db
        self._auth = AuthState(current_level=default_level)
        self._rate_limiter = RateLimiter()
        self._dangerous_re = [re.compile(p, re.IGNORECASE) for p in DANGEROUS_PATTERNS]
        self._blocked_re = [re.compile(p, re.IGNORECASE) for p in BLOCKED_PATTERNS]
        self._safe_re = [re.compile(p, re.IGNORECASE) for p in SAFE_COMMANDS]

    def set_pin(self, pin: str) -> None:
        """Set or update the authentication PIN."""
        self._auth.pin_hash = bcrypt.hashpw(pin.encode(), bcrypt.gensalt())
        logger.info("PIN updated")

    def authenticate(self, pin: str) -> bool:
        """Authenticate with PIN to elevate access."""
        if self._auth.lockout_until > time.time():
            remaining = int(self._auth.lockout_until - time.time())
            logger.warning(f"Account locked for {remaining}s")
            return False

        if self._auth.pin_hash is None:
            logger.error("No PIN configured")
            return False

        if bcrypt.checkpw(pin.encode(), self._auth.pin_hash):
            self._auth.failed_attempts = 0
            self._auth.current_level = AccessLevel.FULL
            self._auth.last_activity = time.time()
            logger.info("Authentication successful, elevated to FULL")
            return True

        self._auth.failed_attempts += 1
        if self._auth.failed_attempts >= self.MAX_FAILED_ATTEMPTS:
            self._auth.lockout_until = time.time() + self.LOCKOUT_DURATION
            self._auth.current_level = AccessLevel.READ_ONLY
            logger.warning(f"Too many failed attempts. Locked out for {self.LOCKOUT_DURATION}s")
        return False

    def check_inactivity(self) -> None:
        """Auto-lock if inactive too long."""
        if time.time() - self._auth.last_activity > self._auth.inactivity_timeout:
            if self._auth.current_level > AccessLevel.SAFE:
                self._auth.current_level = AccessLevel.SAFE
                logger.info("Auto-locked due to inactivity")

    def touch(self) -> None:
        """Record user activity."""
        self._auth.last_activity = time.time()

    @property
    def access_level(self) -> AccessLevel:
        self.check_inactivity()
        return self._auth.current_level

    @property
    def access_level_name(self) -> str:
        return ACCESS_LEVEL_NAMES[self.access_level]

    def set_level(self, level: AccessLevel) -> None:
        """Manually set access level (for testing or admin override)."""
        self._auth.current_level = level
        self._auth.last_activity = time.time()

    def classify_command(self, command: str) -> tuple[AccessLevel, str]:
        """Classify a command's required access level and reason."""
        for pattern in self._blocked_re:
            if pattern.search(command):
                return AccessLevel.FULL, f"Blocked pattern: {pattern.pattern}"

        for pattern in self._dangerous_re:
            if pattern.search(command):
                return AccessLevel.ELEVATED, f"Dangerous pattern: {pattern.pattern}"

        for pattern in self._safe_re:
            if pattern.search(command):
                return AccessLevel.SAFE, "Safe command"

        return AccessLevel.STANDARD, "Standard command"

    def can_execute(self, command: str) -> tuple[bool, str]:
        """Check if the current access level allows executing a command."""
        if not self._rate_limiter.check():
            return False, "Rate limit exceeded"

        required_level, reason = self.classify_command(command)

        for pattern in self._blocked_re:
            if pattern.search(command):
                return False, f"Command blocked: {reason}"

        if required_level > self._auth.current_level:
            return False, (
                f"Insufficient access. Required: {ACCESS_LEVEL_NAMES[required_level]}, "
                f"Current: {ACCESS_LEVEL_NAMES[self._auth.current_level]}. "
                f"Reason: {reason}"
            )

        return True, "Allowed"

    def needs_confirmation(self, command: str) -> bool:
        """Check if a command needs user confirmation."""
        required_level, _ = self.classify_command(command)
        return required_level >= AccessLevel.STANDARD

    async def audit(
        self,
        action: str,
        details: Optional[str] = None,
        success: bool = True,
    ) -> None:
        """Log an action to the audit trail."""
        try:
            await self._db.insert("audit_log", {
                "action": action,
                "details": details,
                "access_level": ACCESS_LEVEL_NAMES[self._auth.current_level],
                "user_ip": "local",
                "success": 1 if success else 0,
                "created_at": time.time(),
            })
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")

    async def get_audit_log(self, limit: int = 50) -> list[dict]:
        """Retrieve recent audit entries."""
        return await self._db.fetch_all(
            "SELECT * FROM audit_log ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )

    def lock(self) -> None:
        """Immediately lock to minimum access."""
        self._auth.current_level = AccessLevel.READ_ONLY
        logger.info("Security locked to READ_ONLY")

    def get_status(self) -> dict[str, Any]:
        """Get current security status."""
        return {
            "access_level": self.access_level_name,
            "is_locked_out": self._auth.lockout_until > time.time(),
            "failed_attempts": self._auth.failed_attempts,
            "rate_limit_remaining": self._rate_limiter.remaining,
            "has_pin": self._auth.pin_hash is not None,
            "inactivity_timeout": self._auth.inactivity_timeout,
            "seconds_since_activity": int(time.time() - self._auth.last_activity),
        }
