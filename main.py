#!/usr/bin/env python3
"""
IGRIS — Entry Point
~~~~~~~~~~~~~~~~~~~~
Parses CLI arguments, prints the startup banner, initialises modules,
launches the FastAPI server, and optionally sets up a system-tray icon.

Usage
-----
    python main.py                        # defaults
    python main.py --port 8080 --debug    # custom port, debug logging
    python main.py --no-voice --no-screen # disable optional modules
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import platform
import signal
import sys
import threading
import webbrowser
from pathlib import Path

VERSION = "1.0.0"
BANNER = r"""
  ██╗ ██████╗ ██████╗ ██╗███████╗
  ██║██╔════╝ ██╔══██╗██║██╔════╝
  ██║██║  ███╗██████╔╝██║███████╗
  ██║██║   ██║██╔══██╗██║╚════██║
  ██║╚██████╔╝██║  ██║██║███████║
  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝╚══════╝
  Shadow Knight AI Assistant  v{version}
  ─────────────────────────────────
"""

BASE_DIR = Path(__file__).resolve().parent
FIRST_RUN_FLAG = BASE_DIR / ".first_run_done"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="igris",
        description="IGRIS — Shadow Knight AI Assistant",
    )
    parser.add_argument("--port", type=int, default=3000, help="Server port (default: 3000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind host")
    parser.add_argument("--no-voice", action="store_true", help="Disable voice module")
    parser.add_argument("--no-screen", action="store_true", help="Disable screen observer")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--no-tray", action="store_true", help="Disable system tray icon")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser on first run")
    return parser.parse_args()


def setup_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    fmt = "%(asctime)s │ %(name)-18s │ %(levelname)-7s │ %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S")
    # Quieten noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def print_banner(port: int, host: str) -> None:
    print(BANNER.format(version=VERSION))
    print(f"  🌐  http://{host}:{port}")
    print(f"  📂  {BASE_DIR}")
    print(f"  🖥️  {platform.system()} {platform.release()}")
    print(f"  🐍  Python {platform.python_version()}")
    print()


def open_browser_on_first_run(host: str, port: int) -> None:
    """Open the browser once after first install."""
    if FIRST_RUN_FLAG.exists():
        return
    try:
        FIRST_RUN_FLAG.touch()
        url = f"http://{host}:{port}"
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    except Exception:
        pass


def start_tray(host: str, port: int) -> threading.Thread:
    """Start the system tray icon in a background thread."""
    try:
        from core.tray import create_tray_icon
        t = threading.Thread(
            target=create_tray_icon,
            args=(host, port),
            daemon=True,
            name="igris-tray",
        )
        t.start()
        return t
    except ImportError:
        logging.getLogger("igris.main").warning(
            "System tray unavailable (pystray not installed)"
        )
    except Exception as exc:
        logging.getLogger("igris.main").warning("Tray setup failed: %s", exc)
    return None  # type: ignore[return-value]


def main() -> None:
    args = parse_args()
    setup_logging(args.debug)
    logger = logging.getLogger("igris.main")

    print_banner(args.port, args.host)

    # Graceful shutdown
    shutdown_event = asyncio.Event()

    def _signal_handler(sig, frame):
        logger.info("Received signal %s — shutting down…", sig)
        shutdown_event.set()
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # System tray
    if not args.no_tray and platform.system() in ("Windows", "Linux", "Darwin"):
        start_tray(args.host, args.port)

    # Open browser on first run
    if not args.no_browser:
        open_browser_on_first_run(args.host, args.port)

    # Environment hints for sub-modules
    if args.no_voice:
        os.environ["IGRIS_NO_VOICE"] = "1"
    if args.no_screen:
        os.environ["IGRIS_NO_SCREEN"] = "1"

    # Run server
    try:
        import uvicorn

        uvicorn.run(
            "server:app",
            host=args.host,
            port=args.port,
            log_level="debug" if args.debug else "info",
            reload=args.debug,
            workers=1,
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user.")
    except Exception as exc:
        logger.critical("Fatal error: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
