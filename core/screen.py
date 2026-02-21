"""Screen Observer - captures and analyzes screen content."""

import asyncio
import io
import hashlib
import time
import re
import sys
import logging
from typing import Optional, List, Callable
from dataclasses import dataclass

logger = logging.getLogger("igris.screen")


@dataclass
class ScreenCapture:
    image_bytes: bytes
    timestamp: float
    monitor_index: int
    hash: str
    width: int
    height: int


class PrivacyFilter:
    """Filters out sensitive window titles and text content."""

    BLOCKED_PATTERNS: List[str] = [
        r"(?i).*bank.*",
        r"(?i).*пароль.*",
        r"(?i).*password.*",
        r"(?i).*1password.*",
        r"(?i).*lastpass.*",
        r"(?i).*bitwarden.*",
        r"(?i).*incognito.*",
        r"(?i).*приват.*",
        r"(?i).*private\s*brows.*",
    ]

    SENSITIVE_PATTERNS: List[str] = [
        r"\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}",
        r"sk-[a-zA-Z0-9]{20,}",
        r"ghp_[a-zA-Z0-9]{36}",
        r"gsk_[a-zA-Z0-9]{20,}",
        r"(?i)password[:\s]=?\s*\S+",
        r"(?i)secret[:\s]=?\s*\S+",
        r"(?i)token[:\s]=?\s*\S+",
    ]

    def __init__(self) -> None:
        self._blocked_compiled = [re.compile(p) for p in self.BLOCKED_PATTERNS]
        self._sensitive_compiled = [re.compile(p) for p in self.SENSITIVE_PATTERNS]

    def check_window_allowed(self, title: str) -> bool:
        """Return True if the window title is safe to capture."""
        if not title:
            return True
        for pat in self._blocked_compiled:
            if pat.search(title):
                logger.info("Blocked window: %s", title[:60])
                return False
        return True

    def mask_sensitive_text(self, text: str) -> str:
        """Replace sensitive patterns in text with [MASKED]."""
        result = text
        for pat in self._sensitive_compiled:
            result = pat.sub("[MASKED]", result)
        return result


class ScreenObserver:
    """Captures screen content periodically with privacy filtering."""

    def __init__(
        self,
        interval: float = 5.0,
        privacy_filter: Optional[PrivacyFilter] = None,
        on_capture: Optional[Callable[[ScreenCapture], None]] = None,
        max_width: int = 1280,
    ) -> None:
        self.interval = interval
        self.privacy_filter = privacy_filter or PrivacyFilter()
        self.on_capture = on_capture
        self.max_width = max_width
        self.enabled = True
        self._task: Optional[asyncio.Task] = None
        self._last_hash: Optional[str] = None
        self._running = False

    async def start(self) -> None:
        """Start background screen capture loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("ScreenObserver started (interval=%.1fs)", self.interval)

    async def stop(self) -> None:
        """Stop background capture."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        logger.info("ScreenObserver stopped")

    async def _loop(self) -> None:
        while self._running:
            try:
                if self.enabled:
                    capture = self.capture_once()
                    if capture and self.on_capture:
                        self.on_capture(capture)
            except Exception:
                logger.exception("Screen capture error")
            await asyncio.sleep(self.interval)

    def capture_once(self, monitor_index: int = 1) -> Optional[ScreenCapture]:
        """Capture a single screenshot. Returns None if blocked or duplicate."""
        if not self.enabled:
            return None

        title = self.get_active_window_title()
        if not self.privacy_filter.check_window_allowed(title):
            return None

        try:
            import mss
            import mss.tools
        except ImportError:
            logger.warning("mss not available, cannot capture screen")
            return None

        try:
            with mss.mss() as sct:
                monitors = sct.monitors
                if monitor_index >= len(monitors):
                    monitor_index = 1 if len(monitors) > 1 else 0
                shot = sct.grab(monitors[monitor_index])
                raw = bytes(shot.rgb)
                img_hash = hashlib.md5(raw).hexdigest()

                if img_hash == self._last_hash:
                    return None
                self._last_hash = img_hash

                image_bytes = self._to_png_bytes(shot, raw)
                image_bytes = self._resize_if_needed(image_bytes, shot.width, shot.height)

                cap_hash = hashlib.sha256(image_bytes).hexdigest()[:16]
                return ScreenCapture(
                    image_bytes=image_bytes,
                    timestamp=time.time(),
                    monitor_index=monitor_index,
                    hash=cap_hash,
                    width=shot.width,
                    height=shot.height,
                )
        except Exception:
            logger.exception("mss capture failed")
            return None

    def _to_png_bytes(self, shot: Any, raw: bytes) -> bytes:
        """Convert mss screenshot to PNG bytes in memory."""
        try:
            from PIL import Image as PILImage
            img = PILImage.frombytes("RGB", (shot.width, shot.height), raw, "raw", "BGRX")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except ImportError:
            import mss.tools
            return mss.tools.to_png(raw, (shot.width, shot.height))

    def _resize_if_needed(self, image_bytes: bytes, width: int, height: int) -> bytes:
        """Resize image if wider than max_width to save LLM tokens."""
        if width <= self.max_width:
            return image_bytes
        try:
            from PIL import Image as PILImage
            img = PILImage.open(io.BytesIO(image_bytes))
            ratio = self.max_width / width
            new_size = (self.max_width, int(height * ratio))
            img = img.resize(new_size, PILImage.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except ImportError:
            return image_bytes

    @staticmethod
    def get_active_window_title() -> str:
        """Get the title of the currently active window."""
        system = sys.platform
        if system == "win32":
            try:
                import ctypes
                hwnd = ctypes.windll.user32.GetForegroundWindow()
                buf = ctypes.create_unicode_buffer(256)
                ctypes.windll.user32.GetWindowTextW(hwnd, buf, 256)
                return buf.value
            except Exception:
                return ""
        elif system == "linux":
            try:
                import subprocess
                result = subprocess.run(
                    ["xdotool", "getactivewindow", "getwindowname"],
                    capture_output=True, text=True, timeout=3,
                )
                return result.stdout.strip() if result.returncode == 0 else ""
            except Exception:
                return ""
        elif system == "darwin":
            try:
                import subprocess
                script = 'tell application "System Events" to get name of first process whose frontmost is true'
                result = subprocess.run(
                    ["osascript", "-e", script],
                    capture_output=True, text=True, timeout=3,
                )
                return result.stdout.strip() if result.returncode == 0 else ""
            except Exception:
                return ""
        return ""

    def capture_all_monitors(self) -> List[ScreenCapture]:
        """Capture all monitors and return list of captures."""
        captures = []
        try:
            import mss
            with mss.mss() as sct:
                for i in range(1, len(sct.monitors)):
                    cap = self.capture_once(monitor_index=i)
                    if cap:
                        captures.append(cap)
        except ImportError:
            logger.warning("mss not available")
        return captures
