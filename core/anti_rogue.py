"""Anti-Rogue Protection - prevents agent from going rogue."""

import hashlib
import os
import time
import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Dict, Optional, Callable, List

logger = logging.getLogger("igris.anti_rogue")


class AntiRogue:
    """Monitors agent behavior and prevents rogue actions."""

    PANIC_PATTERNS: List[str] = [
        r"(?i)игрис[,\s]*стоп",
        r"(?i)^стоп$",
        r"(?i)^panic$",
        r"(?i)emergency\s*stop",
        r"(?i)^halt$",
        r"(?i)^abort$",
        r"(?i)немедленно\s*остановись",
    ]

    def __init__(
        self,
        core_dir: str,
        max_actions: int = 50,
        sleep_timeout: int = 1800,
        stop_timeout: int = 7200,
        lockdown_callback: Optional[Callable] = None,
        sleep_callback: Optional[Callable] = None,
        shutdown_callback: Optional[Callable] = None,
    ) -> None:
        self.core_dir = Path(core_dir).resolve()
        self.max_actions = max_actions
        self.sleep_timeout = sleep_timeout
        self.stop_timeout = stop_timeout
        self.lockdown_callback = lockdown_callback
        self.sleep_callback = sleep_callback
        self.shutdown_callback = shutdown_callback

        self._file_hashes: Dict[str, str] = {}
        self._action_count: int = 0
        self._last_interaction: float = time.time()
        self._locked_down: bool = False
        self._running: bool = False
        self._task: Optional[asyncio.Task] = None
        self._panic_compiled = [re.compile(p) for p in self.PANIC_PATTERNS]

        self._compute_initial_hashes()

    def _compute_initial_hashes(self) -> None:
        """Compute SHA-256 hashes of all .py files in core directory."""
        self._file_hashes.clear()
        if not self.core_dir.exists():
            return
        for py_file in self.core_dir.rglob("*.py"):
            try:
                content = py_file.read_bytes()
                h = hashlib.sha256(content).hexdigest()
                self._file_hashes[str(py_file)] = h
            except Exception:
                logger.warning("Could not hash %s", py_file)

    def verify_integrity(self) -> bool:
        """Recompute hashes and compare with stored registry."""
        if not self._file_hashes:
            return True
        for filepath, expected_hash in self._file_hashes.items():
            p = Path(filepath)
            if not p.exists():
                logger.error("INTEGRITY: File missing: %s", filepath)
                return False
            current_hash = hashlib.sha256(p.read_bytes()).hexdigest()
            if current_hash != expected_hash:
                logger.error("INTEGRITY: Hash mismatch for %s", filepath)
                return False
        return True

    def increment_action(self) -> bool:
        """Increment action counter. Returns False if limit exceeded."""
        if self._locked_down:
            return False
        self._action_count += 1
        if self._action_count > self.max_actions:
            logger.warning("Action limit exceeded (%d/%d), pausing", self._action_count, self.max_actions)
            return False
        return True

    def reset_counter(self) -> None:
        """Reset action counter — called on user interaction."""
        self._action_count = 0
        self._last_interaction = time.time()

    def check_panic(self, message: str) -> bool:
        """Check if message is a panic/stop command."""
        for pat in self._panic_compiled:
            if pat.search(message.strip()):
                logger.critical("PANIC detected: %s", message[:100])
                return True
        return False

    def lockdown(self) -> None:
        """Freeze all operations — read-only mode."""
        self._locked_down = True
        logger.critical("LOCKDOWN activated")
        if self.lockdown_callback:
            try:
                self.lockdown_callback()
            except Exception:
                logger.exception("Lockdown callback failed")

    def unlock(self) -> None:
        """Release lockdown."""
        self._locked_down = False
        self._action_count = 0
        self._last_interaction = time.time()
        logger.info("Lockdown released")

    @property
    def is_locked(self) -> bool:
        return self._locked_down

    async def start(self) -> None:
        """Start background monitoring tasks."""
        if self._running:
            return
        self._running = True
        self._last_interaction = time.time()
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("AntiRogue monitoring started")

    async def stop(self) -> None:
        """Stop background monitoring."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        logger.info("AntiRogue monitoring stopped")

    async def _monitor_loop(self) -> None:
        """Background loop: integrity checks + dead man's switch."""
        check_interval = 300  # 5 minutes
        while self._running:
            try:
                # Integrity check
                if not self.verify_integrity():
                    logger.critical("Integrity check FAILED — triggering lockdown")
                    self.lockdown()

                # Dead man's switch
                elapsed = time.time() - self._last_interaction
                if elapsed > self.stop_timeout:
                    logger.critical("Stop timeout reached (%.0fs), shutting down", elapsed)
                    if self.shutdown_callback:
                        try:
                            self.shutdown_callback()
                        except Exception:
                            logger.exception("Shutdown callback failed")
                    self._running = False
                    break
                elif elapsed > self.sleep_timeout:
                    logger.warning("Sleep timeout reached (%.0fs), entering sleep mode", elapsed)
                    if self.sleep_callback:
                        try:
                            self.sleep_callback()
                        except Exception:
                            logger.exception("Sleep callback failed")

            except Exception:
                logger.exception("Monitor loop error")

            await asyncio.sleep(check_interval)

    def get_status(self) -> Dict:
        """Return current anti-rogue status."""
        return {
            "locked_down": self._locked_down,
            "action_count": self._action_count,
            "max_actions": self.max_actions,
            "seconds_since_interaction": round(time.time() - self._last_interaction, 1),
            "sleep_timeout": self.sleep_timeout,
            "stop_timeout": self.stop_timeout,
            "files_monitored": len(self._file_hashes),
            "running": self._running,
        }
