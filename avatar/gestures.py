"""
IGRIS Avatar Gesture System.

Provides named gestures, context-based selection, an execution queue,
and random idle micro-gestures.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger("igris.avatar.gestures")


class Gesture(enum.Enum):
    POINT = "point"
    STOP = "stop"
    WAVE = "wave"
    HANDSHAKE = "handshake"
    THINKING = "thinking"
    ATTACK = "attack"
    SHIELD = "shield"
    BOW = "bow"
    NOD = "nod"
    # Idle micro-gestures
    BLINK = "blink"
    SUBTLE_MOVE = "subtle_move"
    HEAD_TILT = "head_tilt"


# Duration in ms for each gesture animation
GESTURE_DURATIONS: dict[Gesture, int] = {
    Gesture.POINT: 800,
    Gesture.STOP: 600,
    Gesture.WAVE: 1200,
    Gesture.HANDSHAKE: 1500,
    Gesture.THINKING: 2000,
    Gesture.ATTACK: 700,
    Gesture.SHIELD: 500,
    Gesture.BOW: 1400,
    Gesture.NOD: 600,
    Gesture.BLINK: 200,
    Gesture.SUBTLE_MOVE: 1000,
    Gesture.HEAD_TILT: 800,
}

# Context → gesture mapping
CONTEXT_GESTURES: dict[str, Gesture] = {
    "advice": Gesture.POINT,
    "explain": Gesture.POINT,
    "warning": Gesture.STOP,
    "danger": Gesture.STOP,
    "greeting": Gesture.WAVE,
    "hello": Gesture.WAVE,
    "morning": Gesture.WAVE,
    "agreement": Gesture.NOD,
    "done": Gesture.NOD,
    "accept": Gesture.NOD,
    "thinking": Gesture.THINKING,
    "analyzing": Gesture.THINKING,
    "combat": Gesture.ATTACK,
    "threat": Gesture.SHIELD,
    "respect": Gesture.BOW,
    "farewell": Gesture.BOW,
    "introduction": Gesture.HANDSHAKE,
}

IDLE_GESTURES = [Gesture.BLINK, Gesture.SUBTLE_MOVE, Gesture.HEAD_TILT]


@dataclass
class GestureEvent:
    gesture: Gesture
    priority: int = 0  # higher = more urgent
    timestamp: float = field(default_factory=time.time)


class GestureManager:
    """Manages gesture queue, context selection, and idle animations."""

    def __init__(self, idle_interval: float = 5.0) -> None:
        """
        Args:
            idle_interval: Seconds between random idle gestures.
        """
        self.idle_interval = idle_interval
        self._queue: asyncio.PriorityQueue[tuple[int, float, GestureEvent]] = asyncio.PriorityQueue()
        self._listeners: list[Callable[[dict[str, Any]], Any]] = []
        self._idle_task: Optional[asyncio.Task] = None
        self._running = False

    # -- lifecycle -----------------------------------------------------------

    def start_idle_loop(self) -> None:
        """Start emitting random idle gestures in the background."""
        if self._running:
            return
        self._running = True
        self._idle_task = asyncio.ensure_future(self._idle_loop())

    def stop(self) -> None:
        self._running = False
        if self._idle_task:
            self._idle_task.cancel()

    # -- public API ----------------------------------------------------------

    async def play(self, gesture: Gesture, priority: int = 0) -> None:
        """Enqueue a gesture for execution."""
        evt = GestureEvent(gesture=gesture, priority=priority)
        await self._queue.put((-priority, time.time(), evt))
        await self._process_queue()

    async def play_for_context(self, context: str, priority: int = 0) -> Optional[Gesture]:
        """Select and play a gesture based on semantic context."""
        gesture = CONTEXT_GESTURES.get(context.lower())
        if gesture:
            await self.play(gesture, priority)
            return gesture
        logger.debug("No gesture mapped for context: %s", context)
        return None

    def add_listener(self, fn: Callable[[dict[str, Any]], Any]) -> None:
        self._listeners.append(fn)

    # -- internal ------------------------------------------------------------

    async def _process_queue(self) -> None:
        while not self._queue.empty():
            _, _, evt = await self._queue.get()
            duration = GESTURE_DURATIONS.get(evt.gesture, 500)
            payload = {
                "type": "gesture",
                "gesture": evt.gesture.value,
                "duration_ms": duration,
                "timestamp": time.time(),
            }
            await self._notify(payload)
            # Wait for gesture to finish before next
            await asyncio.sleep(duration / 1000)

    async def _idle_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self.idle_interval + random.uniform(-1, 2))
            if self._queue.empty() and self._running:
                gesture = random.choice(IDLE_GESTURES)
                await self.play(gesture, priority=-1)

    async def _notify(self, payload: dict[str, Any]) -> None:
        for fn in self._listeners:
            try:
                result = fn(payload)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception("Gesture listener error")
