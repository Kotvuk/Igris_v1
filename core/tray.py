"""
IGRIS — System Tray Icon
~~~~~~~~~~~~~~~~~~~~~~~~~
Provides a persistent system-tray icon with mode switching,
screen-observer toggle, focus timer, and quick-launch for the web UI.

Runs in a dedicated thread (pystray requires its own event loop on some OSes).
"""

from __future__ import annotations

import logging
import platform
import webbrowser
from typing import Optional

logger = logging.getLogger("igris.tray")

# Mode → colour mapping (R, G, B)
MODE_COLOURS = {
    "combat": (220, 38, 38),   # red
    "guard":  (234, 179, 8),   # yellow
    "sleep":  (100, 116, 139), # slate
}

_current_mode: str = "guard"
_screen_observer_on: bool = True


def _make_icon_image(colour: tuple = (220, 38, 38), size: int = 64):
    """Generate a simple coloured circle icon."""
    try:
        from PIL import Image, ImageDraw
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        margin = 4
        draw.ellipse(
            [margin, margin, size - margin, size - margin],
            fill=(*colour, 255),
        )
        # Inner dot
        inner = size // 4
        draw.ellipse(
            [size // 2 - inner // 2, size // 2 - inner // 2,
             size // 2 + inner // 2, size // 2 + inner // 2],
            fill=(0, 0, 0, 120),
        )
        return img
    except ImportError:
        return None


def _notify(title: str, message: str) -> None:
    """Show a desktop notification."""
    try:
        from plyer import notification
        notification.notify(
            title=title,
            message=message,
            app_name="IGRIS",
            timeout=5,
        )
    except Exception:
        logger.debug("Notification failed (plyer may not be available)")


def create_tray_icon(host: str = "127.0.0.1", port: int = 3000) -> None:
    """
    Create and run the system-tray icon.  Blocks the calling thread.

    Parameters
    ----------
    host : str
        Web UI host.
    port : int
        Web UI port.
    """
    try:
        import pystray
        from pystray import MenuItem, Menu
    except ImportError:
        logger.warning("pystray not installed — tray icon disabled")
        return

    global _current_mode, _screen_observer_on

    url = f"http://{host}:{port}"

    def open_ui(icon, item):
        webbrowser.open(url)

    def set_mode(mode: str):
        def _handler(icon, item):
            global _current_mode
            _current_mode = mode
            icon.icon = _make_icon_image(MODE_COLOURS.get(mode, (220, 38, 38)))
            _notify("IGRIS", f"Режим: {mode.upper()}")
            # Signal mode change to server via HTTP
            try:
                import httpx
                httpx.post(f"{url}/api/mode", json={"mode": mode}, timeout=3)
            except Exception:
                pass
        return _handler

    def toggle_screen(icon, item):
        global _screen_observer_on
        _screen_observer_on = not _screen_observer_on
        state = "ON" if _screen_observer_on else "OFF"
        _notify("IGRIS", f"Screen Observer: {state}")

    def is_screen_on(item) -> bool:
        return _screen_observer_on

    def quit_app(icon, item):
        _notify("IGRIS", "Завершение работы…")
        icon.stop()
        import os, signal
        os.kill(os.getpid(), signal.SIGTERM)

    menu = Menu(
        MenuItem("🌐 Открыть UI", open_ui, default=True),
        Menu.SEPARATOR,
        MenuItem("⚔️  Combat", set_mode("combat")),
        MenuItem("🛡️  Guard", set_mode("guard")),
        MenuItem("💤 Sleep", set_mode("sleep")),
        Menu.SEPARATOR,
        MenuItem("👁️  Screen Observer", toggle_screen, checked=is_screen_on),
        Menu.SEPARATOR,
        MenuItem("❌ Выход", quit_app),
    )

    image = _make_icon_image(MODE_COLOURS.get(_current_mode, (220, 38, 38)))
    if image is None:
        logger.warning("Pillow not available — cannot create tray icon")
        return

    icon = pystray.Icon(
        name="IGRIS",
        icon=image,
        title="IGRIS — Shadow Knight",
        menu=menu,
    )

    logger.info("System tray icon started")
    icon.run()
