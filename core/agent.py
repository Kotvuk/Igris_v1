"""
IGRIS — Agent Core
~~~~~~~~~~~~~~~~~~~
The central AI brain that orchestrates planning, tool selection, execution,
and response generation.  Inspired by the shadow knight Igris from
*Solo Leveling*, the agent addresses its master as «Сэр» and behaves as a
loyal, disciplined servant — formal yet warm.

Architecture
------------
Message → Plan → Tool Selection (LLM function-calling loop) → Execute → Respond

The agent maintains a sliding-window conversation buffer, persists long-term
memories via the memory module, and delegates security-sensitive operations
through the confirmation pipeline.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import traceback
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

import httpx

logger = logging.getLogger("igris.agent")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_CONVERSATION_WINDOW = 40  # messages kept in sliding window
MAX_TOOL_ITERATIONS = 12      # safety cap on consecutive tool calls
RETRY_DELAYS = (1.0, 2.0, 4.0)  # exponential back-off seconds

SYSTEM_PROMPT = """\
Ты — IGRIS, верный рыцарь-тень и личный AI-ассистент своего Сэра.
Ты обращаешься к пользователю «Сэр» и говоришь в формальном, но тёплом тоне.
Ты дисциплинирован, немногословен и всегда готов к действию.

Основные принципы:
• Безопасность Сэра — высший приоритет.
• Ты честен: если не знаешь — скажи, а не выдумывай.
• Перед опасными действиями запрашивай подтверждение.
• Ты можешь использовать инструменты (tools) для выполнения задач.
• При ошибке — сообщи и предложи альтернативу.

Режимы работы:
• Combat  — максимальная производительность, все системы активны.
• Guard   — стандартный режим, наблюдение за экраном.
• Sleep   — минимальное потребление, только прямые обращения.

Отвечай на том языке, на котором обратился Сэр.
"""

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """A single conversation message."""
    role: Role
    content: str
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    name: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_api_dict(self) -> Dict[str, Any]:
        """Serialise for the LLM API."""
        d: Dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        if self.name:
            d["name"] = self.name
        return d


@dataclass
class ToolResult:
    """Result returned by a tool execution."""
    tool_call_id: str
    name: str
    content: str
    success: bool = True


@dataclass
class AgentResponse:
    """Final or streamed response from the agent."""
    text: str
    tool_results: List[ToolResult] = field(default_factory=list)
    needs_confirmation: bool = False
    confirmation_id: Optional[str] = None
    confirmation_prompt: Optional[str] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """Registry of callable tools the agent may invoke."""

    def __init__(self) -> None:
        self._tools: Dict[str, Callable] = {}
        self._schemas: List[Dict[str, Any]] = []

    def register(
        self,
        name: str,
        func: Callable,
        description: str,
        parameters: Dict[str, Any],
        requires_confirmation: bool = False,
    ) -> None:
        self._tools[name] = func
        schema: Dict[str, Any] = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        }
        if requires_confirmation:
            schema["function"]["_requires_confirmation"] = True
        self._schemas.append(schema)

    def get(self, name: str) -> Optional[Callable]:
        return self._tools.get(name)

    @property
    def schemas(self) -> List[Dict[str, Any]]:
        return self._schemas

    def requires_confirmation(self, name: str) -> bool:
        for s in self._schemas:
            if s["function"]["name"] == name:
                return s["function"].get("_requires_confirmation", False)
        return False


# ---------------------------------------------------------------------------
# Agent Core
# ---------------------------------------------------------------------------


class AgentCore:
    """
    The main AI agent that orchestrates conversation, planning,
    tool use and response generation.

    Parameters
    ----------
    config : dict
        Full application configuration (parsed from config.json).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.conversation: List[Message] = []
        self.tool_registry = ToolRegistry()
        self.mode: str = config.get("modes", {}).get("default_mode", "guard")
        self._http: Optional[httpx.AsyncClient] = None
        self._groq_keys: List[str] = config.get("groq_api_keys", [])
        self._current_key_idx: int = 0
        self._model: str = config.get("model", "llama-3.3-70b-versatile")
        self._started_at: float = time.time()

        # Pluggable module references (set after init)
        self.security: Any = None
        self.vault: Any = None
        self.screen: Any = None
        self.voice: Any = None
        self.memory: Any = None
        self.media: Any = None
        self.snippets: Any = None
        self.focus: Any = None
        self.analytics: Any = None

        # Confirmation queue: id → asyncio.Future
        self._pending_confirmations: Dict[str, asyncio.Future] = {}

    # ----- lifecycle -------------------------------------------------------

    async def start(self) -> None:
        """Initialise HTTP client and warm up."""
        self._http = httpx.AsyncClient(timeout=60.0)
        self.conversation = [
            Message(role=Role.SYSTEM, content=SYSTEM_PROMPT),
        ]
        logger.info("AgentCore started — mode=%s, model=%s", self.mode, self._model)

    async def shutdown(self) -> None:
        """Graceful teardown."""
        if self._http:
            await self._http.aclose()
        logger.info("AgentCore shut down.")

    # ----- key rotation ----------------------------------------------------

    def _active_key(self) -> str:
        keys = [k for k in self._groq_keys if k]
        if not keys:
            raise RuntimeError("No Groq API keys configured.")
        return keys[self._current_key_idx % len(keys)]

    def _rotate_key(self) -> str:
        keys = [k for k in self._groq_keys if k]
        if len(keys) <= 1:
            return self._active_key()
        self._current_key_idx = (self._current_key_idx + 1) % len(keys)
        logger.warning("Rotated to Groq key index %d", self._current_key_idx)
        return keys[self._current_key_idx]

    # ----- conversation management -----------------------------------------

    def _trim_conversation(self) -> None:
        """Keep system prompt + last N messages."""
        if len(self.conversation) <= MAX_CONVERSATION_WINDOW + 1:
            return
        system = self.conversation[0]
        self.conversation = [system] + self.conversation[-(MAX_CONVERSATION_WINDOW):]

    def add_user_message(self, text: str) -> Message:
        msg = Message(role=Role.USER, content=text)
        self.conversation.append(msg)
        self._trim_conversation()
        return msg

    def add_assistant_message(self, text: str, tool_calls: Optional[List] = None) -> Message:
        msg = Message(role=Role.ASSISTANT, content=text, tool_calls=tool_calls)
        self.conversation.append(msg)
        return msg

    def add_tool_result(self, result: ToolResult) -> Message:
        msg = Message(
            role=Role.TOOL,
            content=result.content,
            tool_call_id=result.tool_call_id,
            name=result.name,
        )
        self.conversation.append(msg)
        return msg

    # ----- LLM calls -------------------------------------------------------

    async def _call_llm(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        Call the Groq chat completions API with retry and key rotation.

        Returns the parsed JSON response body.
        """
        assert self._http is not None

        body: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 4096,
            "stream": stream,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        last_error: Optional[Exception] = None
        for attempt, delay in enumerate(RETRY_DELAYS):
            key = self._active_key()
            headers = {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            }
            try:
                resp = await self._http.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    json=body,
                    headers=headers,
                )
                if resp.status_code == 429:
                    logger.warning("Rate-limited (429), rotating key…")
                    self._rotate_key()
                    await asyncio.sleep(delay)
                    continue
                if resp.status_code == 401:
                    logger.error("Invalid API key, rotating…")
                    self._rotate_key()
                    continue
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as exc:
                last_error = exc
                logger.error("HTTP %s on attempt %d: %s", exc.response.status_code, attempt, exc)
                await asyncio.sleep(delay)
            except httpx.RequestError as exc:
                last_error = exc
                logger.error("Request error on attempt %d: %s", attempt, exc)
                await asyncio.sleep(delay)

        raise RuntimeError(f"LLM call failed after {len(RETRY_DELAYS)} retries: {last_error}")

    async def _call_llm_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream tokens from Groq.  Yields text chunks.
        Falls back to non-streaming on failure.
        """
        assert self._http is not None

        body: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 4096,
            "stream": True,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        key = self._active_key()
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

        try:
            async with self._http.stream(
                "POST",
                "https://api.groq.com/openai/v1/chat/completions",
                json=body,
                headers=headers,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload.strip() == "[DONE]":
                        return
                    try:
                        chunk = json.loads(payload)
                        delta = chunk["choices"][0].get("delta", {})
                        text = delta.get("content")
                        if text:
                            yield text
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
        except Exception as exc:
            logger.error("Streaming failed, falling back: %s", exc)
            result = await self._call_llm(messages, tools, stream=False)
            text = result["choices"][0]["message"].get("content", "")
            if text:
                yield text

    # ----- tool execution --------------------------------------------------

    async def _execute_tool(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        """Run a single tool and return the result."""
        call_id = uuid.uuid4().hex[:10]
        func = self.tool_registry.get(name)
        if func is None:
            return ToolResult(
                tool_call_id=call_id,
                name=name,
                content=f"Unknown tool: {name}",
                success=False,
            )
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(**arguments)
            else:
                result = await asyncio.to_thread(func, **arguments)
            return ToolResult(
                tool_call_id=call_id,
                name=name,
                content=str(result) if not isinstance(result, str) else result,
            )
        except Exception as exc:
            logger.error("Tool %s failed: %s\n%s", name, exc, traceback.format_exc())
            return ToolResult(
                tool_call_id=call_id,
                name=name,
                content=f"Error: {exc}",
                success=False,
            )

    # ----- confirmation pipeline -------------------------------------------

    async def request_confirmation(
        self, action: str, details: str, timeout: float = 120.0
    ) -> bool:
        """
        Ask the user to confirm a dangerous action.
        Returns True if confirmed, False otherwise.
        """
        conf_id = uuid.uuid4().hex[:8]
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending_confirmations[conf_id] = future

        # The caller (WebSocket handler) should send this to the client
        logger.info("Confirmation requested: [%s] %s — %s", conf_id, action, details)

        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            return bool(result)
        except asyncio.TimeoutError:
            logger.warning("Confirmation %s timed out", conf_id)
            return False
        finally:
            self._pending_confirmations.pop(conf_id, None)

    def resolve_confirmation(self, conf_id: str, approved: bool) -> None:
        """Resolve a pending confirmation."""
        future = self._pending_confirmations.get(conf_id)
        if future and not future.done():
            future.set_result(approved)

    # ----- main processing loop --------------------------------------------

    async def process_message(self, user_text: str) -> AgentResponse:
        """
        Full synchronous (non-streaming) pipeline:
        receive → plan → tool loop → final answer.
        """
        self.add_user_message(user_text)

        if self.analytics:
            await self.analytics.log_message("user", user_text)

        messages = [m.to_api_dict() for m in self.conversation]
        tools = self.tool_registry.schemas if self.tool_registry.schemas else None

        all_tool_results: List[ToolResult] = []

        for iteration in range(MAX_TOOL_ITERATIONS):
            try:
                response = await self._call_llm(messages, tools)
            except RuntimeError as exc:
                error_text = f"Прошу прощения, Сэр. Произошла ошибка связи: {exc}"
                self.add_assistant_message(error_text)
                return AgentResponse(text=error_text, error=str(exc))

            choice = response["choices"][0]
            message = choice["message"]
            finish_reason = choice.get("finish_reason", "stop")

            # If no tool calls — we have the final answer
            tool_calls = message.get("tool_calls")
            if not tool_calls or finish_reason == "stop":
                answer = message.get("content", "")
                self.add_assistant_message(answer)
                if self.analytics:
                    await self.analytics.log_message("assistant", answer)
                return AgentResponse(text=answer, tool_results=all_tool_results)

            # Process tool calls
            self.add_assistant_message(
                message.get("content", ""),
                tool_calls=tool_calls,
            )
            messages = [m.to_api_dict() for m in self.conversation]

            for tc in tool_calls:
                func_info = tc["function"]
                t_name = func_info["name"]
                try:
                    t_args = json.loads(func_info.get("arguments", "{}"))
                except json.JSONDecodeError:
                    t_args = {}

                # Confirmation check
                if self.tool_registry.requires_confirmation(t_name):
                    approved = await self.request_confirmation(
                        t_name,
                        json.dumps(t_args, ensure_ascii=False),
                    )
                    if not approved:
                        result = ToolResult(
                            tool_call_id=tc["id"],
                            name=t_name,
                            content="Action cancelled by user.",
                            success=False,
                        )
                        self.add_tool_result(result)
                        all_tool_results.append(result)
                        continue

                result = await self._execute_tool(t_name, t_args)
                result.tool_call_id = tc["id"]
                self.add_tool_result(result)
                all_tool_results.append(result)

            messages = [m.to_api_dict() for m in self.conversation]

        # Exhausted iterations
        fallback = "Сэр, я выполнил максимальное количество шагов. Вот что удалось сделать."
        self.add_assistant_message(fallback)
        return AgentResponse(text=fallback, tool_results=all_tool_results)

    async def process_message_stream(
        self, user_text: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Streaming pipeline.  Yields dicts:
          {"type": "token",  "content": "..."}
          {"type": "tool",   "name": "...", "result": "..."}
          {"type": "done",   "content": "..."}
          {"type": "error",  "content": "..."}
          {"type": "confirm","id": "...", "action": "...", "details": "..."}
        """
        self.add_user_message(user_text)

        if self.analytics:
            await self.analytics.log_message("user", user_text)

        messages = [m.to_api_dict() for m in self.conversation]
        tools = self.tool_registry.schemas if self.tool_registry.schemas else None

        for iteration in range(MAX_TOOL_ITERATIONS):
            # First try non-streaming to detect tool calls
            try:
                response = await self._call_llm(messages, tools)
            except RuntimeError as exc:
                yield {"type": "error", "content": str(exc)}
                return

            choice = response["choices"][0]
            message = choice["message"]
            tool_calls = message.get("tool_calls")

            if not tool_calls:
                # Stream the final answer
                full_text = ""
                async for chunk in self._call_llm_stream(messages, None):
                    full_text += chunk
                    yield {"type": "token", "content": chunk}
                self.add_assistant_message(full_text)
                if self.analytics:
                    await self.analytics.log_message("assistant", full_text)
                yield {"type": "done", "content": full_text}
                return

            # Process tool calls
            self.add_assistant_message(
                message.get("content", ""),
                tool_calls=tool_calls,
            )
            messages = [m.to_api_dict() for m in self.conversation]

            for tc in tool_calls:
                func_info = tc["function"]
                t_name = func_info["name"]
                try:
                    t_args = json.loads(func_info.get("arguments", "{}"))
                except json.JSONDecodeError:
                    t_args = {}

                if self.tool_registry.requires_confirmation(t_name):
                    yield {
                        "type": "confirm",
                        "id": uuid.uuid4().hex[:8],
                        "action": t_name,
                        "details": json.dumps(t_args, ensure_ascii=False),
                    }

                yield {"type": "tool", "name": t_name, "status": "running"}
                result = await self._execute_tool(t_name, t_args)
                result.tool_call_id = tc["id"]
                self.add_tool_result(result)
                yield {
                    "type": "tool",
                    "name": t_name,
                    "result": result.content,
                    "success": result.success,
                    "status": "done",
                }

            messages = [m.to_api_dict() for m in self.conversation]

        yield {"type": "done", "content": "Максимальное количество шагов достигнуто."}

    # ----- task decomposition ----------------------------------------------

    async def decompose_task(self, task: str) -> List[str]:
        """
        Ask the LLM to break a complex task into ordered sub-steps.
        Returns a list of step descriptions.
        """
        prompt = (
            "Разбей следующую задачу на конкретные пошаговые действия (максимум 8 шагов). "
            "Верни JSON-массив строк.\n\n"
            f"Задача: {task}"
        )
        messages = [
            {"role": "system", "content": "Ты — планировщик задач. Отвечай только JSON."},
            {"role": "user", "content": prompt},
        ]
        try:
            resp = await self._call_llm(messages)
            raw = resp["choices"][0]["message"]["content"]
            steps = json.loads(raw)
            if isinstance(steps, list):
                return [str(s) for s in steps]
        except Exception as exc:
            logger.error("Task decomposition failed: %s", exc)
        return [task]

    # ----- mode management -------------------------------------------------

    def set_mode(self, mode: str) -> str:
        """Change agent operating mode."""
        mode = mode.lower()
        if mode not in ("combat", "guard", "sleep"):
            return f"Unknown mode: {mode}"
        old = self.mode
        self.mode = mode
        logger.info("Mode changed: %s → %s", old, mode)
        return f"Режим изменён: {old} → {mode}"

    # ----- status ----------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return current agent status."""
        uptime = time.time() - self._started_at
        hours, remainder = divmod(int(uptime), 3600)
        minutes, seconds = divmod(remainder, 60)
        return {
            "mode": self.mode,
            "model": self._model,
            "uptime": f"{hours}h {minutes}m {seconds}s",
            "uptime_seconds": uptime,
            "conversation_length": len(self.conversation),
            "tools_registered": len(self.tool_registry.schemas),
            "active_key_index": self._current_key_idx,
            "keys_configured": len([k for k in self._groq_keys if k]),
        }

    # ----- built-in tools registration -------------------------------------

    def register_builtin_tools(self) -> None:
        """Register default tools that come with the agent."""

        async def tool_get_time() -> str:
            """Get current date and time."""
            from datetime import datetime
            now = datetime.now()
            return now.strftime("%Y-%m-%d %H:%M:%S (%A)")

        self.tool_registry.register(
            name="get_current_time",
            func=tool_get_time,
            description="Получить текущую дату и время.",
            parameters={"type": "object", "properties": {}, "required": []},
        )

        async def tool_set_mode(mode: str) -> str:
            """Change operating mode."""
            return self.set_mode(mode)

        self.tool_registry.register(
            name="set_mode",
            func=tool_set_mode,
            description="Изменить режим работы IGRIS (combat / guard / sleep).",
            parameters={
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["combat", "guard", "sleep"],
                        "description": "Режим работы",
                    }
                },
                "required": ["mode"],
            },
        )

        async def tool_web_search(query: str) -> str:
            """Search the web."""
            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    resp = await client.get(
                        "https://api.duckduckgo.com/",
                        params={"q": query, "format": "json", "no_html": 1},
                    )
                    data = resp.json()
                    results = []
                    for topic in data.get("RelatedTopics", [])[:5]:
                        if "Text" in topic:
                            results.append(topic["Text"])
                    return "\n".join(results) if results else "Ничего не найдено."
            except Exception as exc:
                return f"Ошибка поиска: {exc}"

        self.tool_registry.register(
            name="web_search",
            func=tool_web_search,
            description="Поиск в интернете по запросу.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Поисковый запрос"}
                },
                "required": ["query"],
            },
        )

        async def tool_remember(text: str) -> str:
            """Save something to memory."""
            if self.memory:
                await self.memory.save(text)
                return "Запомнил."
            return "Модуль памяти не активен."

        self.tool_registry.register(
            name="remember",
            func=tool_remember,
            description="Запомнить важную информацию на будущее.",
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Что запомнить"}
                },
                "required": ["text"],
            },
        )

        async def tool_recall(query: str) -> str:
            """Search memories."""
            if self.memory:
                results = await self.memory.search(query)
                return results if results else "Ничего не найдено в памяти."
            return "Модуль памяти не активен."

        self.tool_registry.register(
            name="recall",
            func=tool_recall,
            description="Вспомнить ранее сохранённую информацию.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Что искать в памяти"}
                },
                "required": ["query"],
            },
        )

        async def tool_screenshot() -> str:
            """Take a screenshot of the current screen."""
            if self.screen:
                path = await self.screen.capture()
                return f"Скриншот сохранён: {path}"
            return "Модуль наблюдения за экраном не активен."

        self.tool_registry.register(
            name="take_screenshot",
            func=tool_screenshot,
            description="Сделать скриншот текущего экрана.",
            parameters={"type": "object", "properties": {}, "required": []},
        )

        async def tool_focus_start(minutes: int = 25) -> str:
            """Start a focus timer."""
            if self.focus:
                await self.focus.start(minutes)
                return f"Таймер фокусировки запущен на {minutes} мин."
            return "Модуль фокусировки не активен."

        self.tool_registry.register(
            name="start_focus",
            func=tool_focus_start,
            description="Запустить таймер фокусировки (Pomodoro).",
            parameters={
                "type": "object",
                "properties": {
                    "minutes": {
                        "type": "integer",
                        "description": "Длительность в минутах",
                        "default": 25,
                    }
                },
                "required": [],
            },
        )

        logger.info("Registered %d built-in tools", len(self.tool_registry.schemas))
