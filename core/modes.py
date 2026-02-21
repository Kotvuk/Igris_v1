"""Work Modes - Combat, Guard, Sleep, Learning."""

import asyncio
import time
import logging
from enum import Enum
from typing import Optional, Callable, Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict
from datetime import datetime, date

logger = logging.getLogger("igris.modes")


class Mode(Enum):
    COMBAT = "combat"
    GUARD = "guard"
    SLEEP = "sleep"
    LEARNING = "learning"


@dataclass
class ModeConfig:
    name: str
    description: str
    screen_observer: bool
    proactive_suggestions: bool
    clipboard_monitor: bool
    voice_responses: bool
    auto_confirm_safe: bool
    system_prompt_addition: str


MODE_CONFIGS: Dict[Mode, ModeConfig] = {
    Mode.COMBAT: ModeConfig(
        "Боевой", "Максимум фокуса и проактивности",
        True, True, True, True, True,
        "Ты в боевом режиме. Будь проактивен, предлагай улучшения, следи за экраном.",
    ),
    Mode.GUARD: ModeConfig(
        "Дежурный", "Фоновый мониторинг",
        True, False, True, False, False,
        "Ты в дежурном режиме. Мониторь, но не мешай. Алертуй только при важном.",
    ),
    Mode.SLEEP: ModeConfig(
        "Сон", "Только wake word",
        False, False, False, False, False,
        "Ты в режиме сна. Жди wake word.",
    ),
    Mode.LEARNING: ModeConfig(
        "Обучение", "Объясняй каждый шаг",
        True, True, True, True, False,
        "Ты в режиме обучения. Объясняй каждое действие подробно, как учитель.",
    ),
}

# Allowed transitions: from_mode -> set of allowed to_modes (None = all allowed)
VALID_TRANSITIONS: Dict[Mode, Optional[set]] = {
    Mode.COMBAT: None,
    Mode.GUARD: None,
    Mode.SLEEP: None,
    Mode.LEARNING: None,
}


class ModeManager:
    """Manages the current operating mode and tracks mode history."""

    def __init__(
        self,
        default_mode: Mode = Mode.GUARD,
        on_mode_change: Optional[Callable[[Mode, Mode], None]] = None,
    ):
        self._current_mode: Mode = default_mode
        self._on_mode_change = on_mode_change
        self._history: List[Tuple[Mode, float]] = [(default_mode, time.time())]
        self._daily_time: Dict[str, Dict[Mode, float]] = defaultdict(lambda: defaultdict(float))
        self._last_switch: float = time.time()
        logger.info("ModeManager initialized in %s mode", default_mode.value)

    @property
    def current_mode(self) -> Mode:
        return self._current_mode

    @property
    def mode_history(self) -> List[Tuple[Mode, float]]:
        """Return list of (mode, timestamp) transitions."""
        return list(self._history)

    def switch_mode(self, new_mode: Mode) -> ModeConfig:
        """Switch to a new mode. Returns the new mode's config.

        Raises ValueError if the mode enum value is invalid.
        """
        if not isinstance(new_mode, Mode):
            raise ValueError(f"Invalid mode: {new_mode}")

        old_mode = self._current_mode
        allowed = VALID_TRANSITIONS.get(old_mode)
        if allowed is not None and new_mode not in allowed:
            raise ValueError(
                f"Transition from {old_mode.value} to {new_mode.value} is not allowed"
            )

        # Accumulate time in the old mode
        now = time.time()
        elapsed = now - self._last_switch
        today_key = date.today().isoformat()
        self._daily_time[today_key][old_mode] += elapsed
        self._last_switch = now

        self._current_mode = new_mode
        self._history.append((new_mode, now))

        logger.info("Mode switched: %s -> %s", old_mode.value, new_mode.value)

        if self._on_mode_change:
            try:
                self._on_mode_change(old_mode, new_mode)
            except Exception as e:
                logger.error("Mode change callback error: %s", e)

        return self.get_config()

    def get_config(self) -> ModeConfig:
        """Return the ModeConfig for the current mode."""
        return MODE_CONFIGS[self._current_mode]

    def get_system_prompt_addition(self) -> str:
        """Return the system prompt addition for the current mode."""
        return MODE_CONFIGS[self._current_mode].system_prompt_addition

    def time_in_mode(self, mode: Optional[Mode] = None) -> Dict[str, float]:
        """Return seconds spent in each mode today (or for a specific mode).

        If mode is given, returns {"mode_name": seconds}. Otherwise returns all modes.
        """
        now = time.time()
        today_key = date.today().isoformat()

        # Flush current elapsed time
        current_elapsed = now - self._last_switch
        result: Dict[str, float] = {}

        for m in Mode:
            secs = self._daily_time[today_key].get(m, 0.0)
            if m == self._current_mode:
                secs += current_elapsed
            result[m.value] = round(secs, 1)

        if mode is not None:
            return {mode.value: result.get(mode.value, 0.0)}
        return result

    def get_dominant_mode_today(self) -> Tuple[Mode, float]:
        """Return the mode with the most time today and its duration."""
        times = self.time_in_mode()
        best_mode_val = max(times, key=times.get)  # type: ignore[arg-type]
        best_mode = Mode(best_mode_val)
        return best_mode, times[best_mode_val]
