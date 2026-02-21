"""
IGRIS — FastAPI + WebSocket Server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Serves the web UI, REST API endpoints, and real-time WebSocket channels
for chat streaming and avatar state updates.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.agent import AgentCore

logger = logging.getLogger("igris.server")

# ---------------------------------------------------------------------------
# Globals (set during lifespan)
# ---------------------------------------------------------------------------

agent: Optional[AgentCore] = None
_config: Dict[str, Any] = {}
_start_time: float = 0.0
_active_ws: List[WebSocket] = []
_avatar_ws: List[WebSocket] = []

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
STATIC_DIR = BASE_DIR / "static"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def load_config() -> Dict[str, Any]:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    example = BASE_DIR / "config.example.json"
    if example.exists():
        with open(example, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_config(cfg: Dict[str, Any]) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent, _config, _start_time
    _start_time = time.time()
    _config = load_config()

    # Extract Groq keys from config into flat list for AgentCore
    groq = _config.get("groq", {})
    groq_keys = []
    for key_name in ["key1_primary", "key2_backup", "key3_vision", "key4_fast", "key5_emergency"]:
        k = groq.get(key_name, "")
        if k:
            groq_keys.append(k)
    _config["groq_api_keys"] = groq_keys
    # Set default model from config
    if not _config.get("default_model"):
        _config["default_model"] = groq.get("key1_model", "llama-3.3-70b-versatile")

    agent = AgentCore(_config)
    agent.register_builtin_tools()
    await agent.start()

    logger.info("IGRIS server started.")
    yield

    if agent:
        await agent.shutdown()
    logger.info("IGRIS server stopped.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="IGRIS",
    description="AI Knight Assistant",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception):
    logger.error("Unhandled error: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "message": str(exc)},
    )


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str
    stream: bool = False


class ModeRequest(BaseModel):
    mode: str


class FocusRequest(BaseModel):
    minutes: int = 25


class SettingsUpdate(BaseModel):
    settings: Dict[str, Any]


class MediaRequest(BaseModel):
    prompt: str
    type: str = "image"
    provider: Optional[str] = None


# ---------------------------------------------------------------------------
# WebSocket — Chat
# ---------------------------------------------------------------------------


@app.websocket("/ws")
async def ws_chat(ws: WebSocket):
    await ws.accept()
    _active_ws.append(ws)
    logger.info("WebSocket chat client connected (%d total)", len(_active_ws))
    try:
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type", "message")

            if msg_type == "message" and agent:
                user_text = data.get("content", "")
                if not user_text.strip():
                    continue

                async for event in agent.process_message_stream(user_text):
                    await ws.send_json(event)

            elif msg_type == "confirmation" and agent:
                conf_id = data.get("id", "")
                approved = data.get("approved", False)
                agent.resolve_confirmation(conf_id, approved)
                await ws.send_json({"type": "confirmation_ack", "id": conf_id})

            elif msg_type == "ping":
                await ws.send_json({"type": "pong"})

    except WebSocketDisconnect:
        logger.info("WebSocket chat client disconnected")
    except Exception as exc:
        logger.error("WebSocket error: %s", exc)
    finally:
        if ws in _active_ws:
            _active_ws.remove(ws)


# ---------------------------------------------------------------------------
# WebSocket — Avatar
# ---------------------------------------------------------------------------


@app.websocket("/ws/avatar")
async def ws_avatar(ws: WebSocket):
    await ws.accept()
    _avatar_ws.append(ws)
    try:
        while True:
            data = await ws.receive_json()
            # Broadcast state to all avatar clients
            for client in _avatar_ws:
                if client != ws:
                    try:
                        await client.send_json(data)
                    except Exception:
                        pass
    except WebSocketDisconnect:
        pass
    finally:
        if ws in _avatar_ws:
            _avatar_ws.remove(ws)


# ---------------------------------------------------------------------------
# Helper: broadcast avatar state
# ---------------------------------------------------------------------------


async def broadcast_avatar_state(state: Dict[str, Any]) -> None:
    for ws in _avatar_ws:
        try:
            await ws.send_json(state)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# REST — Chat
# ---------------------------------------------------------------------------


@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    if not agent:
        raise HTTPException(503, "Agent not initialized")
    result = await agent.process_message(req.message)
    return {
        "response": result.text,
        "tool_results": [
            {"name": r.name, "content": r.content, "success": r.success}
            for r in result.tool_results
        ],
        "error": result.error,
    }


# ---------------------------------------------------------------------------
# REST — History
# ---------------------------------------------------------------------------


@app.get("/api/history")
async def api_history(limit: int = 50):
    if not agent:
        return {"messages": []}
    messages = []
    for m in agent.conversation[-limit:]:
        if m.role.value == "system":
            continue
        messages.append({
            "id": m.id,
            "role": m.role.value,
            "content": m.content,
            "timestamp": m.timestamp,
        })
    return {"messages": messages}


# ---------------------------------------------------------------------------
# REST — Settings
# ---------------------------------------------------------------------------


@app.get("/api/settings")
async def api_get_settings():
    cfg = load_config()
    # Mask API keys
    safe = json.loads(json.dumps(cfg))
    for i, key in enumerate(safe.get("groq_api_keys", [])):
        if key:
            safe["groq_api_keys"][i] = key[:8] + "…" + key[-4:] if len(key) > 12 else "***"
    return {"settings": safe}


@app.post("/api/settings")
async def api_save_settings(req: SettingsUpdate):
    global _config
    _config.update(req.settings)
    save_config(_config)
    if agent:
        agent.config = _config
    return {"status": "ok"}


@app.post("/api/settings/test")
async def api_test_settings():
    """Test API connection with current keys."""
    if not agent:
        raise HTTPException(503, "Agent not initialized")
    try:
        import httpx
        key = agent._active_key()
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {key}"},
            )
            if resp.status_code == 200:
                models = resp.json().get("data", [])
                return {"status": "ok", "models": len(models)}
            return {"status": "error", "code": resp.status_code}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


# ---------------------------------------------------------------------------
# REST — Analytics
# ---------------------------------------------------------------------------


@app.get("/api/analytics")
async def api_analytics():
    if agent and agent.analytics:
        data = await agent.analytics.get_summary()
        return {"analytics": data}
    return {"analytics": {"messages_today": 0, "tools_used": 0}}


# ---------------------------------------------------------------------------
# REST — Gallery
# ---------------------------------------------------------------------------


@app.get("/api/gallery")
async def api_gallery():
    if agent and agent.media:
        items = await agent.media.list_gallery()
        return {"gallery": items}
    return {"gallery": []}


@app.post("/api/media/generate")
async def api_media_generate(req: MediaRequest):
    if not agent or not agent.media:
        raise HTTPException(503, "Media module not available")
    result = await agent.media.generate(req.prompt, req.type, req.provider)
    return {"result": result}


# ---------------------------------------------------------------------------
# REST — Status
# ---------------------------------------------------------------------------


@app.get("/api/status")
async def api_status():
    if not agent:
        return {"status": "initializing"}
    status = agent.get_status()
    status["server_uptime"] = time.time() - _start_time
    status["ws_clients"] = len(_active_ws)
    return status


# ---------------------------------------------------------------------------
# REST — Mode
# ---------------------------------------------------------------------------


@app.post("/api/mode")
async def api_mode(req: ModeRequest):
    if not agent:
        raise HTTPException(503, "Agent not initialized")
    result = agent.set_mode(req.mode)
    await broadcast_avatar_state({"type": "mode", "mode": agent.mode})
    return {"message": result, "mode": agent.mode}


# ---------------------------------------------------------------------------
# REST — Focus Timer
# ---------------------------------------------------------------------------


@app.post("/api/focus/start")
async def api_focus_start(req: FocusRequest):
    if agent and agent.focus:
        await agent.focus.start(req.minutes)
        return {"status": "started", "minutes": req.minutes}
    return {"status": "unavailable"}


@app.post("/api/focus/stop")
async def api_focus_stop():
    if agent and agent.focus:
        await agent.focus.stop()
        return {"status": "stopped"}
    return {"status": "unavailable"}


# ---------------------------------------------------------------------------
# REST — Snippets
# ---------------------------------------------------------------------------


@app.get("/api/snippets")
async def api_snippets(q: str = ""):
    if agent and agent.snippets:
        results = await agent.snippets.search(q)
        return {"snippets": results}
    return {"snippets": []}


# ---------------------------------------------------------------------------
# Root redirect
# ---------------------------------------------------------------------------


@app.get("/")
async def root():
    if STATIC_DIR.exists() and (STATIC_DIR / "index.html").exists():
        from fastapi.responses import FileResponse
        return FileResponse(STATIC_DIR / "index.html")
    return {"message": "IGRIS is running. Deploy static files to /static."}
