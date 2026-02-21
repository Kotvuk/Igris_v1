"""File Protection - trash, versioning, snapshots, restore."""

import shutil
import os
import json
import time
import hashlib
import asyncio
import zipfile
import logging
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger("igris.file_protection")


@dataclass
class TrashEntry:
    original_path: str
    trash_path: str
    deleted_at: float
    size: int
    checksum: str


@dataclass
class FileVersion:
    path: str
    version: int
    saved_at: float
    size: int
    backup_path: str


class TrashManager:
    """Manages a recoverable trash system."""

    def __init__(self, trash_dir: str = "~/.igris/trash") -> None:
        self.trash_dir = Path(trash_dir).expanduser().resolve()
        self.trash_dir.mkdir(parents=True, exist_ok=True)

    def delete(self, path: str) -> TrashEntry:
        """Move file to trash with metadata for recovery."""
        p = Path(path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")

        size = p.stat().st_size if p.is_file() else self._dir_size(p)
        checksum = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else ""

        ts = int(time.time() * 1000)
        trash_name = f"{ts}_{p.name}"
        trash_path = self.trash_dir / trash_name

        shutil.move(str(p), str(trash_path))

        entry = TrashEntry(
            original_path=str(p),
            trash_path=str(trash_path),
            deleted_at=time.time(),
            size=size,
            checksum=checksum,
        )

        meta_path = trash_path.parent / f"{trash_name}.meta.json"
        meta_path.write_text(json.dumps({
            "original_path": entry.original_path,
            "trash_path": entry.trash_path,
            "deleted_at": entry.deleted_at,
            "size": entry.size,
            "checksum": entry.checksum,
        }, indent=2))

        logger.info("Trashed: %s → %s", p, trash_path)
        return entry

    def restore(self, entry: TrashEntry) -> bool:
        """Restore a file from trash to its original location."""
        trash_p = Path(entry.trash_path)
        orig_p = Path(entry.original_path)
        if not trash_p.exists():
            logger.error("Trash file not found: %s", trash_p)
            return False
        orig_p.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(trash_p), str(orig_p))
        meta = trash_p.parent / f"{trash_p.name}.meta.json"
        if meta.exists():
            meta.unlink()
        logger.info("Restored: %s → %s", trash_p, orig_p)
        return True

    def list_trash(self) -> List[TrashEntry]:
        """List all items in trash."""
        entries = []
        for meta_file in sorted(self.trash_dir.glob("*.meta.json"), reverse=True):
            try:
                data = json.loads(meta_file.read_text())
                entries.append(TrashEntry(**data))
            except Exception:
                continue
        return entries

    def empty_trash(self, older_than_days: int = 30) -> int:
        """Remove trash entries older than N days. Returns count removed."""
        cutoff = time.time() - (older_than_days * 86400)
        removed = 0
        for entry in self.list_trash():
            if entry.deleted_at < cutoff:
                tp = Path(entry.trash_path)
                if tp.exists():
                    if tp.is_dir():
                        shutil.rmtree(str(tp))
                    else:
                        tp.unlink()
                meta = tp.parent / f"{tp.name}.meta.json"
                if meta.exists():
                    meta.unlink()
                removed += 1
        logger.info("Emptied %d trash items older than %d days", removed, older_than_days)
        return removed

    @staticmethod
    def _dir_size(path: Path) -> int:
        total = 0
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
        return total


class VersionManager:
    """Manages file versioning with automatic cleanup."""

    MAX_VERSIONS = 50

    def __init__(self, versions_dir: str = "~/.igris/versions") -> None:
        self.versions_dir = Path(versions_dir).expanduser().resolve()
        self.versions_dir.mkdir(parents=True, exist_ok=True)

    def _version_dir_for(self, path: str) -> Path:
        """Get the version storage directory for a given file path."""
        file_hash = hashlib.md5(path.encode()).hexdigest()[:16]
        d = self.versions_dir / file_hash
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_version(self, path: str) -> FileVersion:
        """Save a version of the file before modification."""
        p = Path(path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")

        ver_dir = self._version_dir_for(str(p))
        existing = self.get_versions(str(p))
        next_version = (existing[-1].version + 1) if existing else 1

        backup_name = f"v{next_version:04d}_{p.name}"
        backup_path = ver_dir / backup_name
        shutil.copy2(str(p), str(backup_path))

        version = FileVersion(
            path=str(p),
            version=next_version,
            saved_at=time.time(),
            size=p.stat().st_size,
            backup_path=str(backup_path),
        )

        manifest_path = ver_dir / "manifest.json"
        manifest: List[Dict] = []
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
            except Exception:
                manifest = []
        manifest.append({
            "path": version.path, "version": version.version,
            "saved_at": version.saved_at, "size": version.size,
            "backup_path": version.backup_path,
        })
        manifest_path.write_text(json.dumps(manifest, indent=2))

        # Auto-cleanup oldest if exceeding max
        if len(manifest) > self.MAX_VERSIONS:
            to_remove = manifest[:len(manifest) - self.MAX_VERSIONS]
            for old in to_remove:
                old_p = Path(old["backup_path"])
                if old_p.exists():
                    old_p.unlink()
            manifest = manifest[len(manifest) - self.MAX_VERSIONS:]
            manifest_path.write_text(json.dumps(manifest, indent=2))

        logger.info("Saved version %d of %s", next_version, p)
        return version

    def get_versions(self, path: str) -> List[FileVersion]:
        """Get all versions of a file, sorted by version number."""
        p = Path(path).expanduser().resolve()
        ver_dir = self._version_dir_for(str(p))
        manifest_path = ver_dir / "manifest.json"
        if not manifest_path.exists():
            return []
        try:
            manifest = json.loads(manifest_path.read_text())
            versions = [FileVersion(**entry) for entry in manifest if entry.get("path") == str(p)]
            versions.sort(key=lambda v: v.version)
            return versions
        except Exception:
            return []

    def restore_version(self, path: str, version: int) -> bool:
        """Restore a specific version of a file."""
        p = Path(path).expanduser().resolve()
        versions = self.get_versions(str(p))
        target = None
        for v in versions:
            if v.version == version:
                target = v
                break
        if not target:
            logger.error("Version %d not found for %s", version, p)
            return False
        backup = Path(target.backup_path)
        if not backup.exists():
            logger.error("Backup file missing: %s", backup)
            return False
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(backup), str(p))
        logger.info("Restored %s to version %d", p, version)
        return True


class SnapshotManager:
    """Manages periodic workspace snapshots."""

    MAX_SNAPSHOTS = 48

    def __init__(self, snapshots_dir: str = "~/.igris/snapshots", interval: int = 1800) -> None:
        self.snapshots_dir = Path(snapshots_dir).expanduser().resolve()
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.interval = interval
        self._task: Optional[asyncio.Task] = None
        self._running = False

    def create_snapshot(self, workspace_dir: str) -> str:
        """Create a zip snapshot of the workspace."""
        ws = Path(workspace_dir).expanduser().resolve()
        if not ws.exists():
            raise FileNotFoundError(f"Workspace not found: {ws}")

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        snap_path = self.snapshots_dir / f"snapshot_{ts}.zip"

        with zipfile.ZipFile(str(snap_path), "w", zipfile.ZIP_DEFLATED) as zf:
            for fp in ws.rglob("*"):
                if fp.is_file():
                    rel = fp.relative_to(ws)
                    # Skip hidden dirs, __pycache__, node_modules
                    parts = rel.parts
                    if any(p.startswith(".") or p in ("__pycache__", "node_modules", ".git") for p in parts):
                        continue
                    try:
                        zf.write(str(fp), str(rel))
                    except Exception:
                        continue

        self._cleanup_old_snapshots()
        logger.info("Snapshot created: %s", snap_path)
        return str(snap_path)

    def restore_snapshot(self, snapshot_path: str, target_dir: str) -> None:
        """Restore a snapshot to a target directory."""
        snap = Path(snapshot_path).resolve()
        target = Path(target_dir).resolve()
        if not snap.exists():
            raise FileNotFoundError(f"Snapshot not found: {snap}")
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(str(snap), "r") as zf:
            zf.extractall(str(target))
        logger.info("Restored snapshot %s → %s", snap, target)

    def list_snapshots(self) -> List[Dict]:
        """List all snapshots with dates and sizes."""
        snaps = []
        for f in sorted(self.snapshots_dir.glob("snapshot_*.zip"), reverse=True):
            stat = f.stat()
            snaps.append({
                "path": str(f),
                "name": f.name,
                "size": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
        return snaps

    def _cleanup_old_snapshots(self) -> None:
        files = sorted(self.snapshots_dir.glob("snapshot_*.zip"), key=lambda f: f.stat().st_mtime)
        while len(files) > self.MAX_SNAPSHOTS:
            oldest = files.pop(0)
            oldest.unlink()
            logger.info("Removed old snapshot: %s", oldest)

    async def start(self, workspace_dir: str) -> None:
        """Start auto-snapshot background task."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._auto_snapshot_loop(workspace_dir))
        logger.info("Auto-snapshot started (interval=%ds)", self.interval)

    async def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    async def _auto_snapshot_loop(self, workspace_dir: str) -> None:
        while self._running:
            await asyncio.sleep(self.interval)
            try:
                self.create_snapshot(workspace_dir)
            except Exception:
                logger.exception("Auto-snapshot failed")


class FileProtection:
    """Combined file protection: trash + versioning + snapshots."""

    def __init__(
        self,
        trash_dir: str = "~/.igris/trash",
        versions_dir: str = "~/.igris/versions",
        snapshots_dir: str = "~/.igris/snapshots",
    ) -> None:
        self.trash = TrashManager(trash_dir)
        self.versions = VersionManager(versions_dir)
        self.snapshots = SnapshotManager(snapshots_dir)

    def safe_delete(self, path: str) -> TrashEntry:
        """Delete a file safely (move to trash)."""
        return self.trash.delete(path)

    def safe_write(self, path: str, content: str, encoding: str = "utf-8") -> Optional[FileVersion]:
        """Save a version before writing new content."""
        p = Path(path).expanduser().resolve()
        version = None
        if p.exists():
            try:
                version = self.versions.save_version(str(p))
            except Exception:
                logger.warning("Could not save version before write for %s", p)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding=encoding)
        return version

    def safe_move(self, path: str, dest: str) -> Optional[FileVersion]:
        """Save a version before moving."""
        p = Path(path).expanduser().resolve()
        version = None
        if p.exists() and p.is_file():
            try:
                version = self.versions.save_version(str(p))
            except Exception:
                logger.warning("Could not save version before move for %s", p)
        d = Path(dest).expanduser().resolve()
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), str(d))
        return version

    def restore_file(self, path: str, version: Optional[int] = None) -> bool:
        """Restore from version or trash."""
        p = Path(path).expanduser().resolve()
        if version is not None:
            return self.versions.restore_version(str(p), version)
        # Try trash
        for entry in self.trash.list_trash():
            if entry.original_path == str(p):
                return self.trash.restore(entry)
        logger.error("No version or trash entry found for %s", p)
        return False

    async def start_auto_snapshots(self, workspace_dir: str) -> None:
        await self.snapshots.start(workspace_dir)

    async def stop_auto_snapshots(self) -> None:
        await self.snapshots.stop()
