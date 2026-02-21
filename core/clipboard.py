"""Clipboard Intelligence - monitor and suggest actions."""

import re
import time
import asyncio
import logging
from typing import Optional, Callable
from dataclasses import dataclass

logger = logging.getLogger("igris.clipboard")


@dataclass
class ClipboardEvent:
    content: str
    content_type: str  # "error", "url", "code", "path", "text"
    timestamp: float
    suggestion: str


class ClipboardMonitor:
    """Monitors clipboard for changes and suggests contextual actions."""

    SUGGESTIONS = {
        "error": "Проанализировать эту ошибку?",
        "url": "Открыть и проанализировать эту страницу?",
        "code": "Сделать ревью этого кода?",
        "path": "Показать информацию об этом файле?",
        "text": "",
    }

    def __init__(self, on_event: Optional[Callable[[ClipboardEvent], None]] = None) -> None:
        self.on_event = on_event
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_content: Optional[str] = None

    def detect_type(self, content: str) -> str:
        """Detect the type of clipboard content."""
        stripped = content.strip()

        # Error detection
        error_patterns = [
            r"Traceback \(most recent call last\)",
            r"Error:",
            r"Exception:",
            r"FAILED",
            r"panic:",
            r"Errno",
            r"stderr:",
            r"SyntaxError",
            r"TypeError",
            r"ValueError",
            r"KeyError",
            r"IndexError",
            r"AttributeError",
            r"ImportError",
            r"RuntimeError",
            r"FileNotFoundError",
            r"PermissionError",
        ]
        for pat in error_patterns:
            if re.search(pat, stripped):
                return "error"

        # URL detection
        if re.match(r"^https?://\S+$", stripped):
            return "url"

        # Path detection
        path_patterns = [
            r"^[A-Za-z]:\\",  # Windows path
            r"^/[\w./-]+$",  # Unix absolute path
            r"^~/.+$",  # Home-relative
            r"\.(py|js|ts|go|rs|java|c|cpp|h|rb|php|sh|yml|yaml|json|toml|md|txt|html|css)$",
        ]
        for pat in path_patterns:
            if re.search(pat, stripped):
                return "path"

        # Code detection
        code_indicators = [
            r"^\s*(def |class |import |from \S+ import)",
            r"^\s*(function |const |let |var |export )",
            r"^\s*(pub fn |fn |struct |impl |use )",
            r"^\s*(package |func |type |interface )",
            r"^\s*#include\s+[<\"]",
            r"\{\s*$",
            r"^\s{2,}\S",  # Indented code
        ]
        code_score = 0
        for pat in code_indicators:
            if re.search(pat, stripped, re.MULTILINE):
                code_score += 1
        if code_score >= 2 or (code_score >= 1 and len(stripped.splitlines()) > 3):
            return "code"

        return "text"

    def _make_event(self, content: str) -> ClipboardEvent:
        content_type = self.detect_type(content)
        suggestion = self.SUGGESTIONS.get(content_type, "")
        return ClipboardEvent(
            content=content,
            content_type=content_type,
            timestamp=time.time(),
            suggestion=suggestion,
        )

    async def start(self) -> None:
        """Start clipboard monitoring in background."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info("Clipboard monitor started")

    async def stop(self) -> None:
        """Stop clipboard monitoring."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        logger.info("Clipboard monitor stopped")

    async def _poll_loop(self) -> None:
        try:
            import pyperclip
        except ImportError:
            logger.warning("pyperclip not installed, clipboard monitoring disabled")
            self._running = False
            return

        while self._running:
            try:
                current = pyperclip.paste()
                if current and current != self._last_content:
                    self._last_content = current
                    event = self._make_event(current)
                    if event.suggestion and self.on_event:
                        self.on_event(event)
            except Exception:
                pass  # Clipboard access can fail transiently
            await asyncio.sleep(1.0)
