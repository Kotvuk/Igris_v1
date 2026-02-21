"""
IGRIS Database Layer — SQLite ORM with migrations, connection pool, and all tables.
"""

import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional

import aiosqlite

DB_PATH = Path.home() / ".igris" / "igris.db"

SCHEMA_VERSION = 1

TABLES_SQL = [
    """
    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY,
        applied_at REAL NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        model TEXT,
        tokens_in INTEGER DEFAULT 0,
        tokens_out INTEGER DEFAULT 0,
        latency_ms REAL DEFAULT 0,
        session_id TEXT,
        created_at REAL NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS memory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        category TEXT NOT NULL,
        key TEXT NOT NULL,
        value TEXT NOT NULL,
        embedding BLOB,
        importance REAL DEFAULT 0.5,
        access_count INTEGER DEFAULT 0,
        last_accessed REAL,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_memory_category ON memory(category)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_memory_key ON memory(key)
    """,
    """
    CREATE TABLE IF NOT EXISTS analytics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_type TEXT NOT NULL,
        event_data TEXT NOT NULL,
        key_id TEXT,
        tokens_used INTEGER DEFAULT 0,
        duration_ms REAL DEFAULT 0,
        success INTEGER DEFAULT 1,
        created_at REAL NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_analytics_event ON analytics(event_type, created_at)
    """,
    """
    CREATE TABLE IF NOT EXISTS audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        action TEXT NOT NULL,
        details TEXT,
        access_level TEXT,
        user_ip TEXT,
        success INTEGER DEFAULT 1,
        created_at REAL NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_log(created_at)
    """,
    """
    CREATE TABLE IF NOT EXISTS snippets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        language TEXT,
        code TEXT NOT NULL,
        tags TEXT DEFAULT '[]',
        description TEXT,
        use_count INTEGER DEFAULT 0,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_snippets_tags ON snippets(tags)
    """,
    """
    CREATE TABLE IF NOT EXISTS media_gallery (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        media_type TEXT NOT NULL,
        file_path TEXT NOT NULL,
        prompt TEXT,
        model TEXT,
        provider TEXT,
        width INTEGER,
        height INTEGER,
        metadata TEXT DEFAULT '{}',
        created_at REAL NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS file_versions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_path TEXT NOT NULL,
        version INTEGER NOT NULL,
        hash TEXT NOT NULL,
        backup_path TEXT NOT NULL,
        size INTEGER NOT NULL,
        created_at REAL NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_file_versions_path ON file_versions(file_path)
    """,
    """
    CREATE TABLE IF NOT EXISTS episodes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        action TEXT NOT NULL,
        tool_name TEXT,
        input_summary TEXT,
        output_summary TEXT,
        success INTEGER DEFAULT 1,
        duration_ms REAL DEFAULT 0,
        created_at REAL NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS user_profile (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        key TEXT UNIQUE NOT NULL,
        value TEXT NOT NULL,
        confidence REAL DEFAULT 0.5,
        source TEXT,
        updated_at REAL NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS vault_access_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        service_name TEXT NOT NULL,
        action TEXT NOT NULL,
        caller TEXT,
        created_at REAL NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS focus_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_type TEXT NOT NULL DEFAULT 'pomodoro',
        started_at REAL NOT NULL,
        ended_at REAL,
        planned_duration_min INTEGER NOT NULL,
        actions_count INTEGER DEFAULT 0,
        lines_of_code INTEGER DEFAULT 0,
        completed INTEGER DEFAULT 0
    )
    """,
]

MIGRATIONS: dict[int, list[str]] = {
    # Future migrations go here: version -> list of SQL statements
    # 2: ["ALTER TABLE messages ADD COLUMN metadata TEXT DEFAULT '{}'"],
}


class ConnectionPool:
    """Simple async SQLite connection pool."""

    def __init__(self, db_path: Path, max_connections: int = 5):
        self._db_path = db_path
        self._max = max_connections
        self._pool: asyncio.Queue[aiosqlite.Connection] = asyncio.Queue(maxsize=max_connections)
        self._created = 0
        self._lock = asyncio.Lock()

    async def acquire(self) -> aiosqlite.Connection:
        """Get a connection from the pool or create a new one."""
        try:
            return self._pool.get_nowait()
        except asyncio.QueueEmpty:
            pass

        async with self._lock:
            if self._created < self._max:
                conn = await aiosqlite.connect(str(self._db_path))
                conn.row_factory = aiosqlite.Row
                await conn.execute("PRAGMA journal_mode=WAL")
                await conn.execute("PRAGMA foreign_keys=ON")
                self._created += 1
                return conn

        return await self._pool.get()

    async def release(self, conn: aiosqlite.Connection) -> None:
        """Return a connection to the pool."""
        try:
            self._pool.put_nowait(conn)
        except asyncio.QueueFull:
            await conn.close()
            async with self._lock:
                self._created -= 1

    async def close_all(self) -> None:
        """Close every connection in the pool."""
        while not self._pool.empty():
            conn = self._pool.get_nowait()
            await conn.close()
        self._created = 0


class Database:
    """Main database class for IGRIS."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.pool = ConnectionPool(self.db_path)
        self._initialized = False

    async def initialize(self) -> None:
        """Create all tables and run pending migrations."""
        if self._initialized:
            return

        conn = await self.pool.acquire()
        try:
            for sql in TABLES_SQL:
                await conn.execute(sql)
            await conn.commit()
            await self._run_migrations(conn)
            self._initialized = True
        finally:
            await self.pool.release(conn)

    async def _run_migrations(self, conn: aiosqlite.Connection) -> None:
        """Apply any pending migrations."""
        cursor = await conn.execute(
            "SELECT COALESCE(MAX(version), 0) FROM schema_version"
        )
        row = await cursor.fetchone()
        current_version = row[0] if row else 0

        for version in sorted(MIGRATIONS.keys()):
            if version > current_version:
                for sql in MIGRATIONS[version]:
                    await conn.execute(sql)
                await conn.execute(
                    "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                    (version, time.time()),
                )
                await conn.commit()

    async def execute(self, sql: str, params: tuple = ()) -> aiosqlite.Cursor:
        """Execute a single SQL statement."""
        conn = await self.pool.acquire()
        try:
            cursor = await conn.execute(sql, params)
            await conn.commit()
            return cursor
        finally:
            await self.pool.release(conn)

    async def fetch_one(self, sql: str, params: tuple = ()) -> Optional[dict]:
        """Fetch a single row as a dict."""
        conn = await self.pool.acquire()
        try:
            cursor = await conn.execute(sql, params)
            row = await cursor.fetchone()
            if row is None:
                return None
            return dict(row)
        finally:
            await self.pool.release(conn)

    async def fetch_all(self, sql: str, params: tuple = ()) -> list[dict]:
        """Fetch all rows as list of dicts."""
        conn = await self.pool.acquire()
        try:
            cursor = await conn.execute(sql, params)
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]
        finally:
            await self.pool.release(conn)

    async def insert(self, table: str, data: dict[str, Any]) -> int:
        """Insert a row and return the last row id."""
        columns = ", ".join(data.keys())
        placeholders = ", ".join("?" for _ in data)
        sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        conn = await self.pool.acquire()
        try:
            cursor = await conn.execute(sql, tuple(data.values()))
            await conn.commit()
            return cursor.lastrowid  # type: ignore
        finally:
            await self.pool.release(conn)

    async def close(self) -> None:
        """Shut down the connection pool."""
        await self.pool.close_all()
