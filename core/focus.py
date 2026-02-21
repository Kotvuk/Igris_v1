"""Focus Timer - Pomodoro technique with productivity tracking."""

import asyncio
import os
import time
import uuid
import json
import logging
from typing import Optional, Callable, Dict, List
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

logger = logging.getLogger("igris.focus")


class FocusPhase(Enum):
    WORK = "work"
    SHORT_BREAK = "short_break"
    LONG_BREAK = "long_break"
    IDLE = "idle"


@dataclass
class FocusSession:
    id: str
    started_at: float
    ended_at: Optional[float]
    phase: FocusPhase
    pomodoro_count: int
    work_minutes: int
    break_minutes: int
    stats: Dict


class FocusTimer:
    """Pomodoro focus timer with auto-transitions and stat tracking."""

    def __init__(
        self,
        work_min: int = 25,
        short_break_min: int = 5,
        long_break_min: int = 15,
        long_break_after: int = 4,
        on_phase_change: Optional[Callable[[FocusPhase, FocusPhase], None]] = None,
        history_path: str = "~/.igris/focus_history.json",
    ):
        self._work_sec = work_min * 60
        self._short_break_sec = short_break_min * 60
        self._long_break_sec = long_break_min * 60
        self._long_break_after = long_break_after
        self._on_phase_change = on_phase_change
        self._history_path = os.path.expanduser(history_path)
        Path(self._history_path).parent.mkdir(parents=True, exist_ok=True)

        self._phase: FocusPhase = FocusPhase.IDLE
        self._pomodoro_count: int = 0
        self._remaining: float = 0
        self._paused: bool = False
        self._task: Optional[asyncio.Task] = None
        self._session_id: Optional[str] = None
        self._session_start: float = 0
        self._total_work_sec: float = 0
        self._total_break_sec: float = 0
        self._stats: Dict[str, int] = {}

    def _notify_phase(self, old: FocusPhase, new: FocusPhase) -> None:
        logger.info("Phase: %s -> %s", old.value, new.value)
        if self._on_phase_change:
            try:
                self._on_phase_change(old, new)
            except Exception as e:
                logger.error("Phase change callback error: %s", e)

    def _next_phase(self) -> FocusPhase:
        if self._phase == FocusPhase.WORK:
            if self._pomodoro_count > 0 and self._pomodoro_count % self._long_break_after == 0:
                return FocusPhase.LONG_BREAK
            return FocusPhase.SHORT_BREAK
        return FocusPhase.WORK

    def _duration_for(self, phase: FocusPhase) -> float:
        if phase == FocusPhase.WORK:
            return self._work_sec
        elif phase == FocusPhase.SHORT_BREAK:
            return self._short_break_sec
        elif phase == FocusPhase.LONG_BREAK:
            return self._long_break_sec
        return 0

    async def _countdown(self) -> None:
        while True:
            while self._remaining > 0:
                if self._paused:
                    await asyncio.sleep(0.5)
                    continue
                await asyncio.sleep(1)
                self._remaining -= 1
                # Accumulate time
                if self._phase == FocusPhase.WORK:
                    self._total_work_sec += 1
                elif self._phase in (FocusPhase.SHORT_BREAK, FocusPhase.LONG_BREAK):
                    self._total_break_sec += 1

            # Phase ended
            old = self._phase
            if old == FocusPhase.WORK:
                self._pomodoro_count += 1

            new = self._next_phase()
            self._phase = new
            self._remaining = self._duration_for(new)
            self._notify_phase(old, new)

    def start(self) -> Dict:
        """Begin a focus session starting with WORK phase."""
        if self._task and not self._task.done():
            return {"error": "Session already active"}

        self._session_id = uuid.uuid4().hex[:10]
        self._session_start = time.time()
        self._pomodoro_count = 0
        self._total_work_sec = 0
        self._total_break_sec = 0
        self._stats = {}
        self._paused = False

        old = self._phase
        self._phase = FocusPhase.WORK
        self._remaining = self._work_sec
        self._notify_phase(old, FocusPhase.WORK)

        self._task = asyncio.ensure_future(self._countdown())
        logger.info("Focus session %s started", self._session_id)
        return self.get_status()

    def pause(self) -> Dict:
        """Pause the timer."""
        self._paused = True
        return self.get_status()

    def resume(self) -> Dict:
        """Resume the timer."""
        self._paused = False
        return self.get_status()

    def skip(self) -> Dict:
        """Skip to the next phase."""
        self._remaining = 0
        return self.get_status()

    def stop(self) -> Optional[FocusSession]:
        """Stop the session and return the completed FocusSession."""
        if self._task:
            self._task.cancel()
            self._task = None

        if not self._session_id:
            return None

        session = FocusSession(
            id=self._session_id,
            started_at=self._session_start,
            ended_at=time.time(),
            phase=self._phase,
            pomodoro_count=self._pomodoro_count,
            work_minutes=round(self._total_work_sec / 60, 1),
            break_minutes=round(self._total_break_sec / 60, 1),
            stats=dict(self._stats),
        )

        self._phase = FocusPhase.IDLE
        self._remaining = 0
        self._session_id = None
        self._save_session(session)
        logger.info("Focus session stopped: %d pomodoros", session.pomodoro_count)
        return session

    def get_status(self) -> Dict:
        """Return current timer status."""
        return {
            "session_id": self._session_id,
            "phase": self._phase.value,
            "remaining_seconds": max(0, int(self._remaining)),
            "paused": self._paused,
            "pomodoro_count": self._pomodoro_count,
            "work_minutes": round(self._total_work_sec / 60, 1),
            "break_minutes": round(self._total_break_sec / 60, 1),
        }

    def increment_stat(self, key: str, amount: int = 1) -> None:
        """Track activity during focus (e.g. lines_written, commands_run)."""
        self._stats[key] = self._stats.get(key, 0) + amount

    def _save_session(self, session: FocusSession) -> None:
        history = self._load_history()
        entry = {
            "id": session.id,
            "started_at": session.started_at,
            "ended_at": session.ended_at,
            "pomodoro_count": session.pomodoro_count,
            "work_minutes": session.work_minutes,
            "break_minutes": session.break_minutes,
            "stats": session.stats,
        }
        history.append(entry)
        try:
            with open(self._history_path, "w") as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            logger.error("Failed to save focus history: %s", e)

    def _load_history(self) -> List[Dict]:
        if not os.path.exists(self._history_path):
            return []
        try:
            with open(self._history_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []

    def get_history(self, limit: int = 20) -> List[Dict]:
        """Return recent focus sessions from history."""
        history = self._load_history()
        return history[-limit:]
