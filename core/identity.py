"""
IGRIS Identity Lock — Hardcoded creator info and integrity verification.
"""

import asyncio
import hashlib
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("igris.identity")

CORE_DIR = Path(__file__).parent

IDENTITY = {
    "name": "IGRIS",
    "codename": "Shadow Soldier — Red Knight",
    "creator": "Damir",
    "builder": "Ash",
    "owner": "Jasmin",
    "version": "1.0.0",
    "origin": "Solo Leveling",
}


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha = hashlib.sha256()
    try:
        sha.update(file_path.read_bytes())
    except FileNotFoundError:
        return "MISSING"
    return sha.hexdigest()


def compute_core_hashes() -> dict[str, str]:
    """Hash all Python files in the core directory."""
    hashes = {}
    for py_file in sorted(CORE_DIR.glob("*.py")):
        hashes[py_file.name] = compute_file_hash(py_file)
    return hashes


class IdentityGuard:
    """Verifies identity and core file integrity."""

    def __init__(self, on_tamper: Optional[callable] = None):
        self._baseline_hashes: dict[str, str] = {}
        self._on_tamper = on_tamper
        self._check_interval = 300  # 5 minutes
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._tampered = False

    def initialize(self) -> dict[str, str]:
        """Capture baseline hashes at startup."""
        self._baseline_hashes = compute_core_hashes()
        logger.info(f"Identity guard initialized: {len(self._baseline_hashes)} files baselined")
        return self._baseline_hashes

    def verify_integrity(self) -> tuple[bool, list[str]]:
        """Check all core files against baseline hashes."""
        if not self._baseline_hashes:
            self.initialize()

        current = compute_core_hashes()
        violations = []

        for filename, expected_hash in self._baseline_hashes.items():
            actual = current.get(filename, "MISSING")
            if actual != expected_hash:
                violations.append(f"{filename}: expected {expected_hash[:12]}..., got {actual[:12]}...")

        for filename in current:
            if filename not in self._baseline_hashes:
                violations.append(f"{filename}: unexpected new file")

        if violations:
            self._tampered = True
            logger.critical(f"INTEGRITY VIOLATION: {violations}")

        return len(violations) == 0, violations

    async def start_monitoring(self) -> None:
        """Start periodic integrity checks."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())

    async def stop_monitoring(self) -> None:
        """Stop periodic integrity checks."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _monitor_loop(self) -> None:
        """Periodic integrity check loop."""
        while self._running:
            await asyncio.sleep(self._check_interval)
            ok, violations = self.verify_integrity()
            if not ok:
                logger.critical(f"Tamper detected: {violations}")
                if self._on_tamper:
                    self._on_tamper(violations)

    @property
    def is_tampered(self) -> bool:
        return self._tampered

    @staticmethod
    def get_identity() -> dict[str, str]:
        return dict(IDENTITY)

    @staticmethod
    def verify_creator() -> bool:
        """Verify the identity constants haven't been altered at runtime."""
        return (
            IDENTITY["creator"] == "Damir"
            and IDENTITY["builder"] == "Ash"
            and IDENTITY["owner"] == "Jasmin"
        )
