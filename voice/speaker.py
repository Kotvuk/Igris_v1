"""
IGRIS Text-to-Speech & Personality Phrases Module.

Uses Edge TTS for high-quality free synthesis with audio caching,
SSML support, lip-sync viseme generation, and a queued playback pipeline.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import struct
import time
from pathlib import Path
from typing import Any, Optional

import edge_tts  # type: ignore

logger = logging.getLogger("igris.voice.speaker")

# ---------------------------------------------------------------------------
# Igris personality phrases (30+ by situation)
# ---------------------------------------------------------------------------

PHRASES: dict[str, list[str]] = {
    "wake": [
        "Да, Сэр. Ваши пожелания?",
        "Слушаю, Сэр.",
        "Я здесь, Сэр. Чем могу помочь?",
    ],
    "accept": [
        "Будет исполнено.",
        "Принято, Сэр.",
        "Как прикажете.",
    ],
    "done": [
        "Выполнено, Сэр.",
        "Задача завершена.",
        "Готово, Сэр. Что-нибудь ещё?",
    ],
    "error": [
        "Прошу прощения, Сэр. Возникла проблема.",
        "Сэр, произошла ошибка. Анализирую причины.",
        "К сожалению, выполнить не удалось.",
    ],
    "warning": [
        "Сэр, позвольте предупредить…",
        "Внимание, Сэр. Обнаружен потенциальный риск.",
        "Сэр, рекомендую проявить осторожность.",
    ],
    "danger": [
        "Сэр, это может быть опасно. Подтверждаете?",
        "Обнаружена серьёзная угроза. Продолжить?",
        "Сэр, уровень риска высок. Жду подтверждения.",
    ],
    "sleep": [
        "Буду ждать вашего зова.",
        "Перехожу в режим ожидания.",
        "Засыпаю. Позовите, когда понадоблюсь.",
    ],
    "combat": [
        "Боевой режим активирован.",
        "Готов к бою, Сэр.",
        "Включаю боевые протоколы.",
    ],
    "morning": [
        "Доброе утро, Сэр.",
        "Доброе утро. Готов к работе.",
        "С добрым утром, Сэр. Новый день — новые задачи.",
    ],
    "thinking": [
        "Начинаю анализ…",
        "Обрабатываю информацию…",
        "Думаю, Сэр. Одну секунду.",
    ],
    "good_idea": [
        "Отличная идея, Сэр.",
        "Превосходное решение.",
        "Блестящая мысль, Сэр.",
    ],
    "threat": [
        "Обнаружена угроза.",
        "Сэр, зафиксирована подозрительная активность.",
        "Внимание: потенциальная угроза безопасности.",
    ],
}

import random as _random


def get_phrase(situation: str) -> str:
    """Return a random phrase for the given situation."""
    options = PHRASES.get(situation)
    if not options:
        raise ValueError(f"Unknown situation: {situation}")
    return _random.choice(options)


# ---------------------------------------------------------------------------
# Audio cache
# ---------------------------------------------------------------------------

DEFAULT_CACHE_DIR = Path.home() / ".igris" / "tts_cache"


class AudioCache:
    """Disk cache keyed by text+voice hash → mp3 file."""

    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, text: str, voice: str) -> str:
        return hashlib.sha256(f"{voice}::{text}".encode()).hexdigest()[:24]

    def get(self, text: str, voice: str) -> Optional[Path]:
        p = self.cache_dir / f"{self._key(text, voice)}.mp3"
        return p if p.exists() else None

    def put(self, text: str, voice: str, data: bytes) -> Path:
        p = self.cache_dir / f"{self._key(text, voice)}.mp3"
        p.write_bytes(data)
        return p

    def clear(self) -> int:
        count = 0
        for f in self.cache_dir.glob("*.mp3"):
            f.unlink()
            count += 1
        return count


# ---------------------------------------------------------------------------
# SSML helpers
# ---------------------------------------------------------------------------

def wrap_ssml(text: str, rate: str = "+0%", pitch: str = "+0Hz") -> str:
    """Wrap plain text in SSML with optional rate/pitch adjustments."""
    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # Add slight pause after commas for natural delivery
    escaped = escaped.replace(",", ',<break time="200ms"/>')
    escaped = escaped.replace("…", '<break time="400ms"/>')
    escaped = escaped.replace("...", '<break time="400ms"/>')
    return (
        f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="ru-RU">'
        f'<prosody rate="{rate}" pitch="{pitch}">{escaped}</prosody>'
        f"</speak>"
    )


# ---------------------------------------------------------------------------
# Lip-sync viseme generation
# ---------------------------------------------------------------------------

# Simplified phoneme→viseme mapping for Russian
_VISEME_MAP: dict[str, int] = {
    "а": 1, "о": 2, "у": 3, "э": 4, "и": 5, "ы": 5,
    "б": 6, "п": 6, "м": 6,
    "в": 7, "ф": 7,
    "д": 8, "т": 8, "н": 8, "л": 8,
    "г": 9, "к": 9, "х": 9,
    "ж": 10, "ш": 10, "щ": 10, "ч": 10, "ц": 10,
    "р": 11, "с": 12, "з": 12,
    "й": 5, "е": 5, "ё": 2, "ю": 3, "я": 1,
}


def generate_lip_sync(text: str, duration_ms: float) -> list[dict[str, Any]]:
    """
    Generate approximate viseme timeline from text and expected audio duration.

    Returns list of ``{"time_ms": int, "viseme": int, "weight": float}``.
    """
    chars = [c.lower() for c in text if c.isalpha()]
    if not chars:
        return []

    interval = duration_ms / len(chars)
    result: list[dict[str, Any]] = []
    for i, ch in enumerate(chars):
        viseme = _VISEME_MAP.get(ch, 0)
        result.append({
            "time_ms": round(i * interval),
            "viseme": viseme,
            "weight": 0.8 if viseme > 0 else 0.1,
        })
    return result


# ---------------------------------------------------------------------------
# Speaker (TTS engine + playback queue)
# ---------------------------------------------------------------------------

DEFAULT_VOICE = "ru-RU-DmitryNeural"  # deep male voice


class Speaker:
    """
    Igris TTS speaker with caching, SSML, lip-sync data, and queued playback.
    """

    def __init__(
        self,
        voice: str = DEFAULT_VOICE,
        cache_dir: Optional[Path] = None,
        rate: str = "+0%",
        pitch: str = "-5Hz",
    ) -> None:
        self.voice = voice
        self.rate = rate
        self.pitch = pitch
        self._cache = AudioCache(cache_dir or DEFAULT_CACHE_DIR)
        self._queue: asyncio.Queue[tuple[Path, list[dict]]] = asyncio.Queue()
        self._playing = False
        self._playback_task: Optional[asyncio.Task] = None

    # -- public API ----------------------------------------------------------

    async def say(self, text: str, situation: Optional[str] = None) -> list[dict[str, Any]]:
        """
        Synthesise *text* (or a random phrase for *situation*) and enqueue for playback.

        Returns lip-sync viseme data.
        """
        if situation and not text:
            text = get_phrase(situation)

        cached = self._cache.get(text, self.voice)
        if cached:
            logger.debug("Cache hit for: %s", text[:40])
            audio_path = cached
            duration_ms = _estimate_duration(text)
        else:
            audio_path, duration_ms = await self._synthesise(text)

        lip_sync = generate_lip_sync(text, duration_ms)
        await self._queue.put((audio_path, lip_sync))
        self._ensure_playback_loop()
        return lip_sync

    async def say_phrase(self, situation: str) -> list[dict[str, Any]]:
        """Shortcut: speak a random phrase for the given situation."""
        return await self.say("", situation=situation)

    def stop(self) -> None:
        """Clear the queue and stop current playback."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._playing = False

    # -- synthesis -----------------------------------------------------------

    async def _synthesise(self, text: str) -> tuple[Path, float]:
        """Run Edge TTS and cache the result. Returns (path, duration_ms)."""
        ssml = wrap_ssml(text, rate=self.rate, pitch=self.pitch)
        communicate = edge_tts.Communicate(text, self.voice)
        chunks: list[bytes] = []
        duration_ms = 0.0

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                chunks.append(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                offset = chunk.get("offset", 0)
                dur = chunk.get("duration", 0)
                end = (offset + dur) / 10_000  # 100-ns units → ms
                if end > duration_ms:
                    duration_ms = end

        audio_data = b"".join(chunks)
        path = self._cache.put(text, self.voice, audio_data)
        if duration_ms == 0:
            duration_ms = _estimate_duration(text)
        logger.info("Synthesised %d bytes (%.1f ms): %s", len(audio_data), duration_ms, text[:40])
        return path, duration_ms

    # -- playback ------------------------------------------------------------

    def _ensure_playback_loop(self) -> None:
        if self._playback_task is None or self._playback_task.done():
            self._playback_task = asyncio.ensure_future(self._playback_loop())

    async def _playback_loop(self) -> None:
        """Sequentially play queued audio files."""
        self._playing = True
        while self._playing:
            try:
                audio_path, _ = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                if self._queue.empty():
                    break
                continue
            await self._play_file(audio_path)
        self._playing = False

    @staticmethod
    async def _play_file(path: Path) -> None:
        """Play an mp3 file using pygame.mixer (non-blocking wait)."""
        try:
            import pygame  # type: ignore

            if not pygame.mixer.get_init():
                pygame.mixer.init()
            pygame.mixer.music.load(str(path))
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                await asyncio.sleep(0.05)
        except ImportError:
            # Fallback: use system player
            proc = await asyncio.create_subprocess_exec(
                "ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path),
            )
            await proc.wait()


def _estimate_duration(text: str) -> float:
    """Rough duration estimate: ~80 ms per character for Russian speech."""
    return max(500.0, len(text) * 80.0)
