"""
IGRIS Settings Manager — JSON config with defaults, validation, hot-reload.
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import httpx

logger = logging.getLogger("igris.settings")

CONFIG_PATH = Path.home() / ".igris" / "config.json"

DEFAULT_CONFIG: dict[str, Any] = {
    "version": 1,
    "ai": {
        "groq_key_1": "",  # Primary: gpt-oss-120b
        "groq_key_2": "",  # Backup: llama-3.3-70b
        "groq_key_3": "",  # Vision: llama-4-scout
        "groq_key_4": "",  # Fast: gpt-oss-20b
        "groq_key_5": "",  # Emergency: qwen3-32b
        "openai_key": "",
        "anthropic_key": "",
        "default_temperature": 0.7,
        "max_tokens": 4096,
        "hourly_token_limit": 100000,
        "daily_token_limit": 1000000,
    },
    "voice": {
        "elevenlabs_key": "",
        "elevenlabs_voice_id": "",
        "whisper_model": "base",
        "tts_enabled": False,
        "stt_enabled": False,
    },
    "search": {
        "brave_api_key": "",
        "serpapi_key": "",
        "duckduckgo_fallback": True,
    },
    "google": {
        "api_key": "",
        "cx_id": "",
        "credentials_path": "",
    },
    "db": {
        "path": str(Path.home() / ".igris" / "igris.db"),
        "backup_enabled": True,
        "backup_interval_hours": 24,
    },
    "dev": {
        "github_token": "",
        "gitlab_token": "",
        "docker_enabled": False,
    },
    "messengers": {
        "telegram_bot_token": "",
        "telegram_chat_id": "",
        "discord_webhook": "",
    },
    "media": {
        "huggingface_token": "",
        "replicate_token": "",
        "together_ai_key": "",
        "default_image_model": "stabilityai/stable-diffusion-xl-base-1.0",
        "gallery_path": str(Path.home() / ".igris" / "gallery"),
    },
    "security": {
        "default_access_level": "Safe",
        "inactivity_timeout": 300,
        "max_actions_per_minute": 60,
        "confirmation_mode": "Medium",
    },
    "screen": {
        "enabled": False,
        "interval_seconds": 10,
        "privacy_filter": True,
    },
    "general": {
        "language": "ru",
        "theme": "dark",
        "log_level": "INFO",
        "workspace_dir": str(Path.home() / ".igris" / "workspace"),
    },
}


class Settings:
    """Manages IGRIS configuration with hot-reload and validation."""

    def __init__(self, config_path: Optional[Path] = None):
        self._path = config_path or CONFIG_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._config: dict[str, Any] = {}
        self._last_modified: float = 0
        self._watchers: list[callable] = []
        self._watch_task: Optional[asyncio.Task] = None
        self.load()

    def load(self) -> dict[str, Any]:
        """Load config from disk, merging with defaults."""
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    user_config = json.load(f)
                self._config = self._deep_merge(DEFAULT_CONFIG, user_config)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load config: {e}, using defaults")
                self._config = dict(DEFAULT_CONFIG)
        else:
            self._config = dict(DEFAULT_CONFIG)
            self.save()

        self._last_modified = self._path.stat().st_mtime if self._path.exists() else 0
        self._apply_env_overrides()
        return self._config

    def save(self) -> None:
        """Persist current config to disk."""
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._config, f, indent=2, ensure_ascii=False)
        self._last_modified = self._path.stat().st_mtime

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        """Deep merge override into base."""
        result = dict(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = Settings._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides (IGRIS_AI_GROQ_KEY_1, etc.)."""
        env_map = {
            "IGRIS_GROQ_KEY_1": ("ai", "groq_key_1"),
            "IGRIS_GROQ_KEY_2": ("ai", "groq_key_2"),
            "IGRIS_GROQ_KEY_3": ("ai", "groq_key_3"),
            "IGRIS_GROQ_KEY_4": ("ai", "groq_key_4"),
            "IGRIS_GROQ_KEY_5": ("ai", "groq_key_5"),
            "IGRIS_BRAVE_API_KEY": ("search", "brave_api_key"),
            "IGRIS_HF_TOKEN": ("media", "huggingface_token"),
            "IGRIS_REPLICATE_TOKEN": ("media", "replicate_token"),
            "IGRIS_TOGETHER_KEY": ("media", "together_ai_key"),
            "IGRIS_TELEGRAM_TOKEN": ("messengers", "telegram_bot_token"),
            "IGRIS_ELEVENLABS_KEY": ("voice", "elevenlabs_key"),
            "IGRIS_GITHUB_TOKEN": ("dev", "github_token"),
        }
        for env_var, (section, key) in env_map.items():
            value = os.environ.get(env_var)
            if value:
                self._config.setdefault(section, {})[key] = value

    def get(self, *keys: str, default: Any = None) -> Any:
        """Get a nested config value: settings.get('ai', 'groq_key_1')."""
        current = self._config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current

    def set(self, *keys_and_value: Any) -> None:
        """Set a nested config value. Last arg is the value."""
        if len(keys_and_value) < 2:
            raise ValueError("Need at least one key and a value")
        *keys, value = keys_and_value
        current = self._config
        for key in keys[:-1]:
            current = current.setdefault(key, {})
        current[keys[-1]] = value
        self.save()
        self._notify_watchers(keys, value)

    def validate(self) -> list[str]:
        """Validate config and return list of warnings."""
        warnings = []
        ai = self._config.get("ai", {})
        if not any(ai.get(f"groq_key_{i}") for i in range(1, 6)):
            warnings.append("No Groq API keys configured — LLM will not work")
        if not self._config.get("search", {}).get("brave_api_key"):
            warnings.append("No Brave API key — web search will use DuckDuckGo only")
        if self._config.get("ai", {}).get("daily_token_limit", 0) < 10000:
            warnings.append("Daily token limit seems very low")
        return warnings

    def on_change(self, callback: callable) -> None:
        """Register a callback for config changes."""
        self._watchers.append(callback)

    def _notify_watchers(self, keys: list[str], value: Any) -> None:
        for cb in self._watchers:
            try:
                cb(keys, value)
            except Exception as e:
                logger.error(f"Config watcher error: {e}")

    async def start_hot_reload(self, interval: float = 5.0) -> None:
        """Watch config file for external changes."""
        if self._watch_task:
            return
        self._watch_task = asyncio.create_task(self._watch_loop(interval))

    async def stop_hot_reload(self) -> None:
        if self._watch_task:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass
            self._watch_task = None

    async def _watch_loop(self, interval: float) -> None:
        while True:
            await asyncio.sleep(interval)
            if self._path.exists():
                mtime = self._path.stat().st_mtime
                if mtime > self._last_modified:
                    logger.info("Config file changed, reloading...")
                    self.load()
                    for cb in self._watchers:
                        try:
                            cb(["__reload__"], None)
                        except Exception as e:
                            logger.error(f"Reload watcher error: {e}")

    async def test_connection(self, service: str) -> tuple[bool, str]:
        """Test connectivity to a service."""
        tests = {
            "groq": self._test_groq,
            "brave": self._test_brave,
            "huggingface": self._test_huggingface,
            "telegram": self._test_telegram,
        }
        test_fn = tests.get(service)
        if not test_fn:
            return False, f"Unknown service: {service}"
        return await test_fn()

    async def _test_groq(self) -> tuple[bool, str]:
        key = self.get("ai", "groq_key_1")
        if not key:
            return False, "No Groq API key configured"
        async with httpx.AsyncClient(timeout=10) as c:
            resp = await c.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {key}"},
            )
            if resp.status_code == 200:
                return True, "Groq API connected"
            return False, f"Groq API error: {resp.status_code}"

    async def _test_brave(self) -> tuple[bool, str]:
        key = self.get("search", "brave_api_key")
        if not key:
            return False, "No Brave API key configured"
        async with httpx.AsyncClient(timeout=10) as c:
            resp = await c.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": "test"},
                headers={"X-Subscription-Token": key},
            )
            if resp.status_code == 200:
                return True, "Brave Search connected"
            return False, f"Brave error: {resp.status_code}"

    async def _test_huggingface(self) -> tuple[bool, str]:
        token = self.get("media", "huggingface_token")
        if not token:
            return False, "No HuggingFace token configured"
        async with httpx.AsyncClient(timeout=10) as c:
            resp = await c.get(
                "https://huggingface.co/api/whoami-v2",
                headers={"Authorization": f"Bearer {token}"},
            )
            if resp.status_code == 200:
                return True, "HuggingFace connected"
            return False, f"HuggingFace error: {resp.status_code}"

    async def _test_telegram(self) -> tuple[bool, str]:
        token = self.get("messengers", "telegram_bot_token")
        if not token:
            return False, "No Telegram bot token configured"
        async with httpx.AsyncClient(timeout=10) as c:
            resp = await c.get(f"https://api.telegram.org/bot{token}/getMe")
            if resp.status_code == 200:
                return True, "Telegram bot connected"
            return False, f"Telegram error: {resp.status_code}"
