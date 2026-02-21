"""Analytics - track productivity and usage statistics."""

import os
import time
import json
import asyncio
import logging
import aiosqlite
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime, date, timedelta
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger("igris.analytics")


@dataclass
class TaskRecord:
    id: str
    description: str
    started_at: float
    completed_at: Optional[float]
    success: bool
    tools_used: List[str]
    tokens_used: int
    mode: str


@dataclass
class DailyStats:
    date: str
    tasks_completed: int
    tasks_failed: int
    total_tokens: int
    tokens_by_key: Dict[str, int]
    tools_used: Dict[str, int]
    time_by_mode: Dict[str, float]
    messages_sent: int
    messages_received: int
    active_hours: List[int]


_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    description TEXT,
    started_at REAL,
    completed_at REAL,
    success INTEGER,
    tools_used TEXT,
    tokens_used INTEGER,
    mode TEXT,
    date_key TEXT
);
CREATE TABLE IF NOT EXISTS token_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key_name TEXT,
    tokens INTEGER,
    model TEXT,
    ts REAL,
    date_key TEXT
);
CREATE TABLE IF NOT EXISTS tool_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tool_name TEXT,
    ts REAL,
    date_key TEXT
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    direction TEXT,
    ts REAL,
    date_key TEXT,
    hour INTEGER
);
CREATE TABLE IF NOT EXISTS mode_time (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mode TEXT,
    seconds REAL,
    ts REAL,
    date_key TEXT
);
"""


class Analytics:
    """Tracks productivity and usage statistics with SQLite persistence."""

    def __init__(self, db_path: str = "~/.igris/analytics.db"):
        self._db_path = os.path.expanduser(db_path)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialized = False

    async def _ensure_db(self) -> aiosqlite.Connection:
        db = await aiosqlite.connect(self._db_path)
        if not self._initialized:
            await db.executescript(_SCHEMA)
            await db.commit()
            self._initialized = True
        return db

    @staticmethod
    def _today() -> str:
        return date.today().isoformat()

    async def record_task(self, task: TaskRecord) -> None:
        """Persist a completed or failed task."""
        db = await self._ensure_db()
        try:
            await db.execute(
                "INSERT OR REPLACE INTO tasks VALUES (?,?,?,?,?,?,?,?,?)",
                (task.id, task.description, task.started_at, task.completed_at,
                 int(task.success), json.dumps(task.tools_used), task.tokens_used,
                 task.mode, self._today()),
            )
            await db.commit()
        finally:
            await db.close()

    async def record_token_usage(self, key_name: str, tokens: int, model: str = "") -> None:
        db = await self._ensure_db()
        try:
            await db.execute(
                "INSERT INTO token_usage (key_name,tokens,model,ts,date_key) VALUES (?,?,?,?,?)",
                (key_name, tokens, model, time.time(), self._today()),
            )
            await db.commit()
        finally:
            await db.close()

    async def record_tool_usage(self, tool_name: str) -> None:
        db = await self._ensure_db()
        try:
            await db.execute(
                "INSERT INTO tool_usage (tool_name,ts,date_key) VALUES (?,?,?)",
                (tool_name, time.time(), self._today()),
            )
            await db.commit()
        finally:
            await db.close()

    async def record_message(self, direction: str) -> None:
        """Record a message. direction: 'in' or 'out'."""
        now = time.time()
        hour = datetime.fromtimestamp(now).hour
        db = await self._ensure_db()
        try:
            await db.execute(
                "INSERT INTO messages (direction,ts,date_key,hour) VALUES (?,?,?,?)",
                (direction, now, self._today(), hour),
            )
            await db.commit()
        finally:
            await db.close()

    async def record_mode_time(self, mode: str, seconds: float) -> None:
        db = await self._ensure_db()
        try:
            await db.execute(
                "INSERT INTO mode_time (mode,seconds,ts,date_key) VALUES (?,?,?,?)",
                (mode, seconds, time.time(), self._today()),
            )
            await db.commit()
        finally:
            await db.close()

    async def _stats_for_date(self, date_key: str) -> DailyStats:
        db = await self._ensure_db()
        try:
            # Tasks
            cur = await db.execute(
                "SELECT success, COUNT(*) FROM tasks WHERE date_key=? GROUP BY success", (date_key,)
            )
            task_counts = dict(await cur.fetchall())
            completed = task_counts.get(1, 0)
            failed = task_counts.get(0, 0)

            # Tokens
            cur = await db.execute(
                "SELECT COALESCE(SUM(tokens),0) FROM token_usage WHERE date_key=?", (date_key,)
            )
            total_tokens = (await cur.fetchone())[0]

            cur = await db.execute(
                "SELECT key_name, SUM(tokens) FROM token_usage WHERE date_key=? GROUP BY key_name",
                (date_key,),
            )
            tokens_by_key = {r[0]: r[1] for r in await cur.fetchall()}

            # Tools
            cur = await db.execute(
                "SELECT tool_name, COUNT(*) FROM tool_usage WHERE date_key=? GROUP BY tool_name",
                (date_key,),
            )
            tools_used = {r[0]: r[1] for r in await cur.fetchall()}

            # Mode time
            cur = await db.execute(
                "SELECT mode, SUM(seconds) FROM mode_time WHERE date_key=? GROUP BY mode",
                (date_key,),
            )
            time_by_mode = {r[0]: round(r[1], 1) for r in await cur.fetchall()}

            # Messages
            cur = await db.execute(
                "SELECT direction, COUNT(*) FROM messages WHERE date_key=? GROUP BY direction",
                (date_key,),
            )
            msg_counts = dict(await cur.fetchall())

            # Active hours
            cur = await db.execute(
                "SELECT DISTINCT hour FROM messages WHERE date_key=? ORDER BY hour", (date_key,)
            )
            active_hours = [r[0] for r in await cur.fetchall()]

            return DailyStats(
                date=date_key,
                tasks_completed=completed,
                tasks_failed=failed,
                total_tokens=total_tokens,
                tokens_by_key=tokens_by_key,
                tools_used=tools_used,
                time_by_mode=time_by_mode,
                messages_sent=msg_counts.get("out", 0),
                messages_received=msg_counts.get("in", 0),
                active_hours=active_hours,
            )
        finally:
            await db.close()

    async def get_today_stats(self) -> DailyStats:
        return await self._stats_for_date(self._today())

    async def get_stats_range(self, start_date: str, end_date: str) -> List[DailyStats]:
        """Return stats for each day in [start_date, end_date] (ISO format)."""
        results: List[DailyStats] = []
        current = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)
        while current <= end:
            stats = await self._stats_for_date(current.isoformat())
            results.append(stats)
            current += timedelta(days=1)
        return results

    async def get_heatmap_data(self, days: int = 90) -> Dict[str, int]:
        """Return {date_iso: activity_count} for the last N days."""
        db = await self._ensure_db()
        try:
            cutoff = (date.today() - timedelta(days=days)).isoformat()
            heatmap: Dict[str, int] = {}

            for table in ("tasks", "token_usage", "tool_usage", "messages"):
                cur = await db.execute(
                    f"SELECT date_key, COUNT(*) FROM {table} WHERE date_key>=? GROUP BY date_key",
                    (cutoff,),
                )
                for row in await cur.fetchall():
                    heatmap[row[0]] = heatmap.get(row[0], 0) + row[1]

            return heatmap
        finally:
            await db.close()

    async def get_productivity_score(self) -> float:
        """Return 0-100 score based on tasks completed, consistency, and activity."""
        today_stats = await self.get_today_stats()
        heatmap = await self.get_heatmap_data(days=7)

        # Task score (up to 40 pts): 5 pts per completed task, max 40
        task_score = min(today_stats.tasks_completed * 5, 40)

        # Consistency score (up to 30 pts): active days in last 7
        active_days = sum(1 for v in heatmap.values() if v > 0)
        consistency_score = min(active_days * (30 / 7), 30)

        # Activity score (up to 30 pts): based on messages + tools
        activity = today_stats.messages_sent + today_stats.messages_received + sum(today_stats.tools_used.values())
        activity_score = min(activity * 1.5, 30)

        return round(min(task_score + consistency_score + activity_score, 100), 1)

    async def compare_with_yesterday(self) -> Dict[str, float]:
        """Return deltas between today and yesterday stats."""
        today = await self._stats_for_date(self._today())
        yesterday_key = (date.today() - timedelta(days=1)).isoformat()
        yesterday = await self._stats_for_date(yesterday_key)

        return {
            "tasks_completed_delta": today.tasks_completed - yesterday.tasks_completed,
            "tasks_failed_delta": today.tasks_failed - yesterday.tasks_failed,
            "tokens_delta": today.total_tokens - yesterday.total_tokens,
            "messages_sent_delta": today.messages_sent - yesterday.messages_sent,
            "messages_received_delta": today.messages_received - yesterday.messages_received,
        }

    async def get_top_tools(self, limit: int = 10) -> List[Tuple[str, int]]:
        db = await self._ensure_db()
        try:
            cur = await db.execute(
                "SELECT tool_name, COUNT(*) as cnt FROM tool_usage GROUP BY tool_name ORDER BY cnt DESC LIMIT ?",
                (limit,),
            )
            return [(r[0], r[1]) for r in await cur.fetchall()]
        finally:
            await db.close()
