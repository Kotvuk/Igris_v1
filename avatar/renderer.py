"""
IGRIS Avatar WebSocket Server.

Real-time communication layer between the Python backend and a Three.js
(or any web) frontend for avatar state, gestures, and lip-sync data.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Callable, Optional, Set

logger = logging.getLogger("igris.avatar.renderer")

try:
    import websockets  # type: ignore
    from websockets.server import WebSocketServerProtocol  # type: ignore
except ImportError:
    websockets = None  # type: ignore


class AvatarWebSocketServer:
    """
    WebSocket server that:
    - Sends avatar state changes, gestures, and lip-sync frames to connected clients.
    - Receives interaction events (clicks, hover, etc.) from the frontend.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
    ) -> None:
        self.host = host
        self.port = port
        self._clients: Set[Any] = set()
        self._server: Optional[Any] = None
        self._event_handlers: dict[str, list[Callable[[dict], Any]]] = {}

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Start the WebSocket server."""
        if websockets is None:
            raise RuntimeError("websockets package is required: pip install websockets")
        self._server = await websockets.serve(
            self._handler,
            self.host,
            self.port,
        )
        logger.info("Avatar WebSocket server started on ws://%s:%d", self.host, self.port)

    async def stop(self) -> None:
        """Gracefully shut down."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            logger.info("Avatar WebSocket server stopped")

        for ws in list(self._clients):
            await ws.close()
        self._clients.clear()

    # -- broadcasting --------------------------------------------------------

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Send a JSON message to all connected clients."""
        if not self._clients:
            return
        data = json.dumps(message, ensure_ascii=False)
        disconnected = set()
        for ws in self._clients:
            try:
                await ws.send(data)
            except Exception:
                disconnected.add(ws)
        self._clients -= disconnected

    async def send_state(self, state_data: dict[str, Any]) -> None:
        """Broadcast a state-change event."""
        await self.broadcast({"category": "state", **state_data})

    async def send_gesture(self, gesture_data: dict[str, Any]) -> None:
        """Broadcast a gesture event."""
        await self.broadcast({"category": "gesture", **gesture_data})

    async def send_lip_sync(self, frames: list[dict[str, Any]]) -> None:
        """Broadcast lip-sync keyframe data."""
        await self.broadcast({
            "category": "lip_sync",
            "type": "lip_sync_data",
            "frames": frames,
            "timestamp": time.time(),
        })

    async def send_custom(self, event_type: str, data: dict[str, Any]) -> None:
        """Broadcast an arbitrary custom event."""
        await self.broadcast({"category": "custom", "type": event_type, **data})

    # -- event handling (from frontend) --------------------------------------

    def on(self, event_type: str, handler: Callable[[dict], Any]) -> None:
        """Register a handler for frontend events of a given type."""
        self._event_handlers.setdefault(event_type, []).append(handler)

    async def _dispatch(self, event: dict[str, Any]) -> None:
        event_type = event.get("type", "")
        handlers = self._event_handlers.get(event_type, [])
        for h in handlers:
            try:
                result = h(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception("Error in event handler for %s", event_type)

    # -- connection handler --------------------------------------------------

    async def _handler(self, ws: Any) -> None:
        """Handle a single WebSocket connection."""
        self._clients.add(ws)
        remote = ws.remote_address
        logger.info("Client connected: %s", remote)

        # Send welcome / current state
        await ws.send(json.dumps({
            "category": "system",
            "type": "connected",
            "message": "IGRIS avatar stream ready",
            "timestamp": time.time(),
        }))

        try:
            async for raw in ws:
                try:
                    event = json.loads(raw)
                    logger.debug("Received from %s: %s", remote, event)
                    await self._dispatch(event)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON from %s: %s", remote, raw[:100])
        except Exception:
            logger.debug("Client disconnected: %s", remote)
        finally:
            self._clients.discard(ws)

    # -- utility -------------------------------------------------------------

    @property
    def client_count(self) -> int:
        return len(self._clients)
