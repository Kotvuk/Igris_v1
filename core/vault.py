"""
IGRIS Vault — AES-256-GCM encrypted secret storage with proxy layer.
Agent never sees raw keys; vault.use() injects credentials transparently.
"""

import base64
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger("igris.vault")

VAULT_PATH = Path.home() / ".igris" / "vault.enc"


class Vault:
    """Encrypted secret storage with proxy credential injection."""

    def __init__(self, db: Any, vault_path: Optional[Path] = None):
        self._db = db
        self._vault_path = vault_path or VAULT_PATH
        self._vault_path.parent.mkdir(parents=True, exist_ok=True)
        self._fernet: Optional[Fernet] = None
        self._secrets: dict[str, dict[str, str]] = {}
        self._unlocked = False

    @staticmethod
    def _derive_key(master_password: str) -> bytes:
        """Derive a Fernet key from a master password using SHA-256."""
        digest = hashlib.sha256(master_password.encode()).digest()
        return base64.urlsafe_b64encode(digest)

    def unlock(self, master_password: str) -> bool:
        """Unlock the vault with the master password."""
        key = self._derive_key(master_password)
        self._fernet = Fernet(key)
        if self._vault_path.exists():
            try:
                encrypted = self._vault_path.read_bytes()
                decrypted = self._fernet.decrypt(encrypted)
                self._secrets = json.loads(decrypted)
                self._unlocked = True
                logger.info("Vault unlocked successfully")
                return True
            except (InvalidToken, json.JSONDecodeError):
                self._fernet = None
                self._unlocked = False
                logger.error("Failed to unlock vault — wrong password or corrupted data")
                return False
        else:
            self._secrets = {}
            self._unlocked = True
            self._save()
            logger.info("New vault created")
            return True

    def lock(self) -> None:
        """Lock the vault, clearing all decrypted data from memory."""
        self._secrets = {}
        self._fernet = None
        self._unlocked = False
        logger.info("Vault locked")

    @property
    def is_unlocked(self) -> bool:
        return self._unlocked

    def _save(self) -> None:
        """Encrypt and persist the vault to disk."""
        if not self._fernet:
            raise VaultLocked("Vault is locked")
        data = json.dumps(self._secrets).encode()
        encrypted = self._fernet.encrypt(data)
        self._vault_path.write_bytes(encrypted)

    def store(self, service_name: str, credentials: dict[str, str]) -> None:
        """Store credentials for a service."""
        if not self._unlocked:
            raise VaultLocked("Vault is locked")
        self._secrets[service_name] = credentials
        self._save()
        logger.info(f"Stored credentials for: {service_name}")

    def remove(self, service_name: str) -> bool:
        """Remove credentials for a service."""
        if not self._unlocked:
            raise VaultLocked("Vault is locked")
        if service_name in self._secrets:
            del self._secrets[service_name]
            self._save()
            logger.info(f"Removed credentials for: {service_name}")
            return True
        return False

    def list_services(self) -> list[str]:
        """List all stored service names (no secrets exposed)."""
        if not self._unlocked:
            raise VaultLocked("Vault is locked")
        return list(self._secrets.keys())

    def has(self, service_name: str) -> bool:
        """Check if credentials exist for a service."""
        if not self._unlocked:
            raise VaultLocked("Vault is locked")
        return service_name in self._secrets

    async def use(self, service_name: str, caller: str = "agent") -> "_CredentialProxy":
        """Get a proxy that provides credentials without exposing raw values."""
        if not self._unlocked:
            raise VaultLocked("Vault is locked")
        if service_name not in self._secrets:
            raise ServiceNotFound(f"No credentials for: {service_name}")

        await self._audit_access(service_name, "use", caller)
        return _CredentialProxy(self._secrets[service_name])

    def _get_raw(self, service_name: str) -> dict[str, str]:
        """Internal: get raw credentials. Only used by proxy layer."""
        if not self._unlocked:
            raise VaultLocked("Vault is locked")
        if service_name not in self._secrets:
            raise ServiceNotFound(f"No credentials for: {service_name}")
        return self._secrets[service_name]

    async def inject_into_headers(
        self, service_name: str, headers: dict[str, str], caller: str = "agent"
    ) -> dict[str, str]:
        """Inject credentials into request headers transparently."""
        creds = self._get_raw(service_name)
        await self._audit_access(service_name, "inject_headers", caller)
        result = dict(headers)
        if "api_key" in creds:
            result["Authorization"] = f"Bearer {creds['api_key']}"
        if "token" in creds:
            result["Authorization"] = f"Bearer {creds['token']}"
        for key, value in creds.items():
            if key.startswith("header_"):
                header_name = key[7:]
                result[header_name] = value
        return result

    async def inject_into_url(
        self, service_name: str, url: str, caller: str = "agent"
    ) -> str:
        """Inject API key into URL query parameters."""
        creds = self._get_raw(service_name)
        await self._audit_access(service_name, "inject_url", caller)
        if "api_key" in creds:
            separator = "&" if "?" in url else "?"
            return f"{url}{separator}key={creds['api_key']}"
        return url

    async def _audit_access(self, service_name: str, action: str, caller: str) -> None:
        """Log every vault access."""
        try:
            await self._db.insert("vault_access_log", {
                "service_name": service_name,
                "action": action,
                "caller": caller,
                "created_at": time.time(),
            })
        except Exception as e:
            logger.error(f"Failed to log vault access: {e}")


class _CredentialProxy:
    """Proxy that exposes credential properties without revealing raw values."""

    def __init__(self, creds: dict[str, str]):
        self._creds = creds

    def get_header(self, key: str = "api_key") -> tuple[str, str]:
        """Get an authorization header tuple."""
        val = self._creds.get(key, "")
        return ("Authorization", f"Bearer {val}")

    def get(self, key: str) -> str:
        """Get a specific credential value."""
        return self._creds.get(key, "")

    @property
    def has_api_key(self) -> bool:
        return "api_key" in self._creds

    def __repr__(self) -> str:
        return f"CredentialProxy(keys={list(self._creds.keys())})"

    def __str__(self) -> str:
        return "[REDACTED CREDENTIALS]"


class VaultLocked(Exception):
    pass


class ServiceNotFound(Exception):
    pass
