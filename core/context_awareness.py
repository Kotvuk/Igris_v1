"""Window Context Awareness - detect active app and switch modes."""

import re
import platform
import asyncio
import subprocess
import time
import logging
from typing import Optional, Callable, Dict, List
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

logger = logging.getLogger("igris.context_awareness")


class AppContext(Enum):
    IDE = "ide"
    BROWSER = "browser"
    TERMINAL = "terminal"
    MESSENGER = "messenger"
    OFFICE = "office"
    MEDIA = "media"
    UNKNOWN = "unknown"


CONTEXT_PATTERNS: Dict[AppContext, List[str]] = {
    AppContext.IDE: [
        r'visual studio code', r'vscode', r'pycharm', r'intellij', r'sublime',
        r'atom', r'\bvim\b', r'\bnvim\b', r'\.py\s*[-–]', r'\.js\s*[-–]',
        r'webstorm', r'goland', r'rider', r'android studio',
    ],
    AppContext.BROWSER: [
        r'\bchrome\b', r'chromium', r'firefox', r'\bedge\b', r'\bsafari\b',
        r'\bopera\b', r'\bbrave\b', r'vivaldi',
    ],
    AppContext.TERMINAL: [
        r'cmd\.exe', r'powershell', r'windows terminal', r'\bterminal\b',
        r'iterm', r'\bbash\b', r'\bzsh\b', r'командная строка', r'konsole',
        r'gnome-terminal', r'alacritty', r'kitty', r'wezterm',
    ],
    AppContext.MESSENGER: [
        r'telegram', r'discord', r'\bslack\b', r'whatsapp', r'\bteams\b',
        r'signal', r'viber', r'skype',
    ],
    AppContext.OFFICE: [
        r'\bword\b', r'\bexcel\b', r'powerpoint', r'google docs', r'notion',
        r'obsidian', r'libreoffice', r'google sheets',
    ],
    AppContext.MEDIA: [
        r'photoshop', r'figma', r'premiere', r'after effects', r'blender',
        r'gimp', r'inkscape', r'davinci resolve', r'audacity',
    ],
}

MODE_SUGGESTIONS: Dict[AppContext, str] = {
    AppContext.IDE: "combat",
    AppContext.BROWSER: "guard",
    AppContext.TERMINAL: "combat",
    AppContext.MESSENGER: "guard",
    AppContext.OFFICE: "guard",
    AppContext.MEDIA: "guard",
    AppContext.UNKNOWN: "guard",
}


class ContextAwareness:
    """Detects the active application window and suggests operating modes."""

    def __init__(self, on_context_change: Optional[Callable[[AppContext, AppContext], None]] = None):
        self._on_context_change = on_context_change
        self._current_context: AppContext = AppContext.UNKNOWN
        self._monitoring: bool = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._time_per_context: Dict[AppContext, float] = defaultdict(float)
        self._last_switch_time: float = time.time()
        self._compiled_patterns: Dict[AppContext, List[re.Pattern]] = {
            ctx: [re.compile(p, re.IGNORECASE) for p in patterns]
            for ctx, patterns in CONTEXT_PATTERNS.items()
        }

    @property
    def current_context(self) -> AppContext:
        return self._current_context

    def get_active_window_title(self) -> str:
        """Return the title of the currently focused window."""
        system = platform.system()
        try:
            if system == "Windows":
                import ctypes
                hwnd = ctypes.windll.user32.GetForegroundWindow()
                buf = ctypes.create_unicode_buffer(512)
                ctypes.windll.user32.GetWindowTextW(hwnd, buf, 512)
                return buf.value
            elif system == "Linux":
                result = subprocess.run(
                    ["xdotool", "getactivewindow", "getwindowname"],
                    capture_output=True, text=True, timeout=3,
                )
                return result.stdout.strip() if result.returncode == 0 else ""
            elif system == "Darwin":
                script = (
                    'tell application "System Events" to get name of '
                    '(first application process whose frontmost is true)'
                )
                result = subprocess.run(
                    ["osascript", "-e", script],
                    capture_output=True, text=True, timeout=3,
                )
                return result.stdout.strip() if result.returncode == 0 else ""
        except Exception as e:
            logger.debug("Failed to get window title: %s", e)
        return ""

    def detect_context(self) -> AppContext:
        """Detect current app context from the active window title."""
        title = self.get_active_window_title()
        if not title:
            return AppContext.UNKNOWN
        for ctx, patterns in self._compiled_patterns.items():
            for pat in patterns:
                if pat.search(title):
                    return ctx
        return AppContext.UNKNOWN

    def get_suggested_mode(self, context: Optional[AppContext] = None) -> str:
        """Return the suggested Igris mode for the given (or current) context."""
        ctx = context if context is not None else self._current_context
        return MODE_SUGGESTIONS.get(ctx, "guard")

    def get_time_per_context(self) -> Dict[str, float]:
        """Return seconds spent in each context since tracking started."""
        self._flush_time()
        return {ctx.value: secs for ctx, secs in self._time_per_context.items()}

    def _flush_time(self) -> None:
        now = time.time()
        elapsed = now - self._last_switch_time
        self._time_per_context[self._current_context] += elapsed
        self._last_switch_time = now

    def _update_context(self, new_ctx: AppContext) -> None:
        if new_ctx != self._current_context:
            old = self._current_context
            self._flush_time()
            self._current_context = new_ctx
            logger.info("Context changed: %s -> %s", old.value, new_ctx.value)
            if self._on_context_change:
                try:
                    self._on_context_change(old, new_ctx)
                except Exception as e:
                    logger.error("Context change callback error: %s", e)

    async def start_monitoring(self, interval: float = 2.0) -> None:
        """Start background monitoring of the active window."""
        if self._monitoring:
            return
        self._monitoring = True
        self._last_switch_time = time.time()

        async def _loop() -> None:
            while self._monitoring:
                try:
                    ctx = self.detect_context()
                    self._update_context(ctx)
                except Exception as e:
                    logger.error("Monitoring error: %s", e)
                await asyncio.sleep(interval)

        self._monitor_task = asyncio.ensure_future(_loop())
        logger.info("Context monitoring started (interval=%.1fs)", interval)

    def stop(self) -> None:
        """Stop background monitoring."""
        self._monitoring = False
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
        self._flush_time()
        logger.info("Context monitoring stopped")
