"""
IGRIS Avatar State Machine.

Manages avatar states (SLEEPING → IDLE → LISTENING → …), tracks durations,
and broadcasts state changes over WebSocket.
"""

from __future__ import annotations

import asyncio
import enum
import json
import logging
import time
from typing import Any, Callable, Optional

logger = logging.getLogger("igris.avatar.states")


class AvatarState(enum.Enum):
    SLEEPING = "sleeping"
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    WORKING = "working"


# Valid transitions: from → {to, …}
_TRANSITIONS: dict[AvatarState, set[AvatarState]] = {
    AvatarState.SLEEPING: {AvatarState.IDLE, AvatarState.LISTENING},
    AvatarState.IDLE: {AvatarState.LISTENING, AvatarState.THINKING, AvatarState.WORKING, AvatarState.SLEEPING},
    AvatarState.LISTENING: {AvatarState.THINKING, AvatarState.IDLE, AvatarState.SLEEPING},
    AvatarState.THINKING: {AvatarState.SPEAKING, AvatarState.WORKING, AvatarState.IDLE, AvatarState.LISTENING},
    AvatarState.SPEAKING: {AvatarState.IDLE, AvatarState.LISTENING, AvatarState.THINKING},
    AvatarState.WORKING: {AvatarState.SPEAKING, AvatarState.THINKING, AvatarState.IDLE},
}

# Animations triggered on entering a state
_ENTER_ANIMATIONS: dict[AvatarState, str] = {
    AvatarState.SLEEPING: "fade_out",
    AvatarState.IDLE: "breathe",
    AvatarState.LISTENING: "perk_up",
    AvatarState.THINKING: "hand_chin",
    AvatarState.SPEAKING: "talk_start",
    AvatarState.WORKING: "type_start",
}


class AvatarStateMachine:
    """
    Tracks avatar state with transition validation, duration logging,
    and WebSocket notification dispatch.
    """

    def __init__(self) -> None:
        self._state = AvatarState.SLEEPING
        self._state_entered_at: float = time.monotonic()
        self._listeners: list[Callable[[dict[str, Any]], Any]] = []
        self._history: list[tuple[AvatarState, float]] = []

    # -- properties ----------------------------------------------------------

    @property
    def state(self) -> AvatarState:
        return self._state

    @property
    def state_duration(self) -> float:
        """Seconds in the current state."""
        return time.monotonic() - self._state_entered_at

    @property
    def history(self) -> list[tuple[AvatarState, float]]:
        """List of (state, duration_seconds) for past states."""
        return list(self._history)

    # -- transitions ---------------------------------------------------------

    async def transition(self, new_state: AvatarState) -> bool:
        """
        Attempt a state transition. Returns True on success.
        Broadcasts a WebSocket notification on change.
        """
        if new_state == self._state:
            return True

        allowed = _TRANSITIONS.get(self._state, set())
        if new_state not in allowed:
            logger.warning(
                "Invalid transition %s → %s (allowed: %s)",
                self._state.value, new_state.value,
                [s.value for s in allowed],
            )
            return False

        duration = self.state_duration
        self._history.append((self._state, duration))

        old = self._state
        self._state = new_state
        self._state_entered_at = time.monotonic()

        animation = _ENTER_ANIMATIONS.get(new_state, "default")
        logger.info("State: %s → %s (was %.1fs, anim=%s)", old.value, new_state.value, duration, animation)

        await self._notify({
            "type": "state_change",
            "from": old.value,
            "to": new_state.value,
            "animation": animation,
            "previous_duration_ms": round(duration * 1000),
            "timestamp": time.time(),
        })
        return True

    # -- observer / WebSocket integration ------------------------------------

    def add_listener(self, fn: Callable[[dict[str, Any]], Any]) -> None:
        """Register a callback (sync or async) for state-change events."""
        self._listeners.append(fn)

    def remove_listener(self, fn: Callable) -> None:
        self._listeners = [l for l in self._listeners if l is not fn]

    async def _notify(self, payload: dict[str, Any]) -> None:
        for fn in self._listeners:
            try:
                result = fn(payload)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception("Listener error")

    # -- convenience ---------------------------------------------------------

    async def wake(self) -> bool:
        if self._state == AvatarState.SLEEPING:
            return await self.transition(AvatarState.IDLE)
        return True

    async def sleep(self) -> bool:
        return await self.transition(AvatarState.SLEEPING)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self._state.value,
            "duration_ms": round(self.state_duration * 1000),
            "animation": _ENTER_ANIMATIONS.get(self._state, "default"),
        }
