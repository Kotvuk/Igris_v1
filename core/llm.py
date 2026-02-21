"""
IGRIS LLM Client — Groq API with 5-key rotation, health tracking, token budgets, and async support.
"""

import asyncio
import base64
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Optional

import httpx

logger = logging.getLogger("igris.llm")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


class KeyRole(Enum):
    PRIMARY = "primary"
    BACKUP = "backup"
    VISION = "vision"
    FAST = "fast"
    EMERGENCY = "emergency"


@dataclass
class KeyHealth:
    """Tracks health metrics for a single API key."""
    total_requests: int = 0
    successful: int = 0
    failed: int = 0
    rate_limited: int = 0
    total_latency_ms: float = 0.0
    last_error: Optional[str] = None
    last_error_time: Optional[float] = None
    last_success_time: Optional[float] = None
    consecutive_failures: int = 0
    cooldown_until: float = 0.0

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return self.successful / self.total_requests

    @property
    def avg_latency_ms(self) -> float:
        if self.successful == 0:
            return 0.0
        return self.total_latency_ms / self.successful

    @property
    def is_healthy(self) -> bool:
        if time.time() < self.cooldown_until:
            return False
        if self.consecutive_failures >= 5:
            return False
        return True

    def record_success(self, latency_ms: float) -> None:
        self.total_requests += 1
        self.successful += 1
        self.total_latency_ms += latency_ms
        self.last_success_time = time.time()
        self.consecutive_failures = 0

    def record_failure(self, error: str, is_rate_limit: bool = False) -> None:
        self.total_requests += 1
        self.failed += 1
        self.last_error = error
        self.last_error_time = time.time()
        self.consecutive_failures += 1
        if is_rate_limit:
            self.rate_limited += 1
            self.cooldown_until = time.time() + min(60 * self.consecutive_failures, 300)
        elif self.consecutive_failures >= 3:
            self.cooldown_until = time.time() + 30 * self.consecutive_failures


@dataclass
class KeyConfig:
    """Configuration for a single API key."""
    api_key: str
    role: KeyRole
    model: str
    health: KeyHealth = field(default_factory=KeyHealth)
    priority: int = 0
    supports_vision: bool = False


@dataclass
class TokenBudget:
    """Track token usage against daily/hourly limits."""
    hourly_limit: int = 100_000
    daily_limit: int = 1_000_000
    _hourly_used: int = 0
    _daily_used: int = 0
    _hour_start: float = field(default_factory=time.time)
    _day_start: float = field(default_factory=time.time)

    def _reset_if_needed(self) -> None:
        now = time.time()
        if now - self._hour_start >= 3600:
            self._hourly_used = 0
            self._hour_start = now
        if now - self._day_start >= 86400:
            self._daily_used = 0
            self._day_start = now

    def can_spend(self, tokens: int) -> bool:
        self._reset_if_needed()
        return (
            self._hourly_used + tokens <= self.hourly_limit
            and self._daily_used + tokens <= self.daily_limit
        )

    def spend(self, tokens: int) -> None:
        self._reset_if_needed()
        self._hourly_used += tokens
        self._daily_used += tokens

    @property
    def hourly_remaining(self) -> int:
        self._reset_if_needed()
        return max(0, self.hourly_limit - self._hourly_used)

    @property
    def daily_remaining(self) -> int:
        self._reset_if_needed()
        return max(0, self.daily_limit - self._daily_used)


class LLMClient:
    """Groq LLM client with multi-key rotation and auto-fallback."""

    DEFAULT_MODELS = {
        KeyRole.PRIMARY: "gpt-oss-120b",
        KeyRole.BACKUP: "llama-3.3-70b-versatile",
        KeyRole.VISION: "llama-4-scout-17b-16e-instruct",
        KeyRole.FAST: "gpt-oss-20b",
        KeyRole.EMERGENCY: "qwen3-32b",
    }

    def __init__(
        self,
        api_keys: Optional[dict[KeyRole, str]] = None,
        hourly_limit: int = 100_000,
        daily_limit: int = 1_000_000,
        timeout: float = 120.0,
    ):
        self._keys: dict[KeyRole, KeyConfig] = {}
        self._budget = TokenBudget(hourly_limit=hourly_limit, daily_limit=daily_limit)
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        self._fallback_order = [
            KeyRole.PRIMARY,
            KeyRole.BACKUP,
            KeyRole.FAST,
            KeyRole.EMERGENCY,
        ]
        self._vision_key = KeyRole.VISION

        if api_keys:
            for role, key in api_keys.items():
                self.add_key(role, key)

    def add_key(self, role: KeyRole, api_key: str, model: Optional[str] = None) -> None:
        """Register an API key for a specific role."""
        self._keys[role] = KeyConfig(
            api_key=api_key,
            role=role,
            model=model or self.DEFAULT_MODELS[role],
            priority=list(KeyRole).index(role),
            supports_vision=(role == KeyRole.VISION),
        )

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    def _select_key(self, need_vision: bool = False) -> Optional[KeyConfig]:
        """Select the best available key based on health and role."""
        if need_vision and KeyRole.VISION in self._keys:
            cfg = self._keys[KeyRole.VISION]
            if cfg.health.is_healthy:
                return cfg

        for role in self._fallback_order:
            if role in self._keys and self._keys[role].health.is_healthy:
                return self._keys[role]

        # Last resort: return any key ignoring cooldown
        for role in self._fallback_order:
            if role in self._keys:
                return self._keys[role]

        return None

    async def chat(
        self,
        messages: list[dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stream: bool = False,
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[str] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send a chat completion request with automatic key rotation and retry."""
        has_images = any(
            isinstance(m.get("content"), list)
            and any(p.get("type") == "image_url" for p in m["content"])
            for m in messages
        )

        estimated_tokens = sum(len(str(m.get("content", ""))) // 4 for m in messages) + max_tokens
        if not self._budget.can_spend(estimated_tokens):
            raise TokenBudgetExceeded(
                f"Token budget exceeded. Hourly remaining: {self._budget.hourly_remaining}, "
                f"Daily remaining: {self._budget.daily_remaining}"
            )

        last_error: Optional[Exception] = None
        tried_keys: set[KeyRole] = set()

        for attempt in range(len(self._keys) + 2):
            key_cfg = self._select_key(need_vision=has_images)
            if key_cfg is None:
                break
            if key_cfg.role in tried_keys:
                # All keys tried, do exponential backoff on primary
                key_cfg = self._keys.get(KeyRole.PRIMARY) or key_cfg
                backoff = min(2 ** attempt, 30)
                logger.warning(f"All keys tried, backing off {backoff}s before retry")
                await asyncio.sleep(backoff)

            tried_keys.add(key_cfg.role)
            use_model = model or key_cfg.model

            payload: dict[str, Any] = {
                "model": use_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream,
            }
            if tools:
                payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = tool_choice
            payload.update(kwargs)

            headers = {
                "Authorization": f"Bearer {key_cfg.api_key}",
                "Content-Type": "application/json",
            }

            start = time.time()
            try:
                client = await self._get_client()
                if stream:
                    return await self._stream_request(client, headers, payload, key_cfg, start)

                resp = await client.post(GROQ_API_URL, json=payload, headers=headers)

                if resp.status_code == 429:
                    key_cfg.health.record_failure("rate_limited", is_rate_limit=True)
                    logger.warning(f"Rate limited on {key_cfg.role.value}, rotating...")
                    last_error = RateLimitError(f"429 on {key_cfg.role.value}")
                    continue

                if resp.status_code >= 500:
                    key_cfg.health.record_failure(f"server_error_{resp.status_code}")
                    last_error = LLMError(f"Server error {resp.status_code}")
                    continue

                resp.raise_for_status()
                data = resp.json()
                latency = (time.time() - start) * 1000
                key_cfg.health.record_success(latency)

                usage = data.get("usage", {})
                total_tokens = usage.get("total_tokens", 0)
                self._budget.spend(total_tokens)

                return data

            except httpx.TimeoutException:
                key_cfg.health.record_failure("timeout")
                last_error = LLMError("Request timed out")
                continue
            except httpx.HTTPStatusError as e:
                key_cfg.health.record_failure(str(e))
                last_error = LLMError(str(e))
                continue
            except Exception as e:
                key_cfg.health.record_failure(str(e))
                last_error = e
                continue

        raise last_error or LLMError("No API keys available")

    async def _stream_request(
        self,
        client: httpx.AsyncClient,
        headers: dict,
        payload: dict,
        key_cfg: KeyConfig,
        start: float,
    ) -> dict[str, Any]:
        """Handle streaming response, collecting chunks into a single result."""
        collected_content = []
        collected_tool_calls: list[dict] = []
        finish_reason = None
        model_name = ""

        async with client.stream("POST", GROQ_API_URL, json=payload, headers=headers) as resp:
            if resp.status_code == 429:
                key_cfg.health.record_failure("rate_limited", is_rate_limit=True)
                raise RateLimitError("429 rate limited")
            resp.raise_for_status()

            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                model_name = chunk.get("model", model_name)
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta", {})
                    if "content" in delta and delta["content"]:
                        collected_content.append(delta["content"])
                    if "tool_calls" in delta:
                        for tc in delta["tool_calls"]:
                            idx = tc.get("index", 0)
                            while len(collected_tool_calls) <= idx:
                                collected_tool_calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                            if "id" in tc:
                                collected_tool_calls[idx]["id"] = tc["id"]
                            fn = tc.get("function", {})
                            if "name" in fn:
                                collected_tool_calls[idx]["function"]["name"] = fn["name"]
                            if "arguments" in fn:
                                collected_tool_calls[idx]["function"]["arguments"] += fn["arguments"]
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]

        latency = (time.time() - start) * 1000
        key_cfg.health.record_success(latency)

        message: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(collected_content) or None,
        }
        if collected_tool_calls:
            message["tool_calls"] = collected_tool_calls

        return {
            "model": model_name,
            "choices": [{"message": message, "finish_reason": finish_reason}],
            "usage": {"total_tokens": len("".join(collected_content)) // 4},
        }

    async def vision(
        self,
        prompt: str,
        image_data: bytes,
        image_mime: str = "image/png",
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send an image to the vision model for analysis."""
        b64 = base64.b64encode(image_data).decode()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{image_mime};base64,{b64}"},
                    },
                ],
            }
        ]
        return await self.chat(messages, max_tokens=max_tokens, **kwargs)

    def get_health_report(self) -> dict[str, Any]:
        """Return health metrics for all keys."""
        report = {}
        for role, cfg in self._keys.items():
            h = cfg.health
            report[role.value] = {
                "model": cfg.model,
                "total_requests": h.total_requests,
                "success_rate": round(h.success_rate, 3),
                "avg_latency_ms": round(h.avg_latency_ms, 1),
                "rate_limited": h.rate_limited,
                "consecutive_failures": h.consecutive_failures,
                "is_healthy": h.is_healthy,
                "last_error": h.last_error,
            }
        report["budget"] = {
            "hourly_remaining": self._budget.hourly_remaining,
            "daily_remaining": self._budget.daily_remaining,
        }
        return report

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()


class LLMError(Exception):
    pass


class RateLimitError(LLMError):
    pass


class TokenBudgetExceeded(LLMError):
    pass
