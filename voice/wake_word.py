"""
IGRIS Wake Word Detection Module.

Supports Picovoice Porcupine for custom wake word "Восстань Игрис"
with fallback to keyword spotting via speech_recognition.
"""

from __future__ import annotations

import logging
import struct
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger("igris.voice.wake_word")

# ---------------------------------------------------------------------------
# Porcupine-based detector (preferred)
# ---------------------------------------------------------------------------

class PorcupineWakeWord:
    """Wake word detection using Picovoice Porcupine engine."""

    def __init__(
        self,
        access_key: str,
        keyword_path: str,
        sensitivity: float = 0.6,
        device_index: int = -1,
    ) -> None:
        """
        Args:
            access_key: Picovoice access key.
            keyword_path: Path to .ppn keyword file for "Восстань Игрис".
            sensitivity: Detection sensitivity 0.0–1.0.
            device_index: Audio input device index (-1 = default).
        """
        self.access_key = access_key
        self.keyword_path = keyword_path
        self.sensitivity = max(0.0, min(1.0, sensitivity))
        self.device_index = device_index

        self._porcupine = None
        self._recorder = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._callback: Optional[Callable[[], None]] = None

    # -- lifecycle -----------------------------------------------------------

    def start(self, callback: Callable[[], None]) -> None:
        """Begin listening for the wake word in a background thread."""
        if self._running:
            logger.warning("PorcupineWakeWord already running")
            return

        import pvporcupine  # type: ignore
        import pvrecorder  # type: ignore

        self._callback = callback
        self._porcupine = pvporcupine.create(
            access_key=self.access_key,
            keyword_paths=[self.keyword_path],
            sensitivities=[self.sensitivity],
        )
        self._recorder = pvrecorder.PvRecorder(
            frame_length=self._porcupine.frame_length,
            device_index=self.device_index,
        )
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        logger.info("Porcupine wake-word listener started (sensitivity=%.2f)", self.sensitivity)

    def stop(self) -> None:
        """Stop listening and release resources."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        if self._recorder:
            self._recorder.stop()
            self._recorder.delete()
            self._recorder = None
        if self._porcupine:
            self._porcupine.delete()
            self._porcupine = None
        logger.info("Porcupine wake-word listener stopped")

    # -- internal ------------------------------------------------------------

    def _listen_loop(self) -> None:
        assert self._porcupine and self._recorder
        self._recorder.start()
        while self._running:
            try:
                pcm = self._recorder.read()
                keyword_index = self._porcupine.process(pcm)
                if keyword_index >= 0:
                    logger.info("Wake word detected (porcupine)")
                    if self._callback:
                        self._callback()
            except Exception:
                logger.exception("Error in porcupine listen loop")
                time.sleep(0.5)


# ---------------------------------------------------------------------------
# Fallback detector using speech_recognition
# ---------------------------------------------------------------------------

_WAKE_KEYWORDS = [
    "восстань игрис",
    "игрис",
    "встань игрис",
    "эй игрис",
]


class FallbackWakeWord:
    """Simple keyword-spotting fallback using Google/Vosk speech recognition."""

    def __init__(
        self,
        keywords: Optional[list[str]] = None,
        energy_threshold: int = 300,
        pause_threshold: float = 0.8,
        device_index: Optional[int] = None,
        recognizer_language: str = "ru-RU",
    ) -> None:
        self.keywords = [k.lower() for k in (keywords or _WAKE_KEYWORDS)]
        self.energy_threshold = energy_threshold
        self.pause_threshold = pause_threshold
        self.device_index = device_index
        self.language = recognizer_language

        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._callback: Optional[Callable[[], None]] = None

    def start(self, callback: Callable[[], None]) -> None:
        if self._running:
            return
        self._callback = callback
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        logger.info("Fallback wake-word listener started (keywords=%s)", self.keywords)

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("Fallback wake-word listener stopped")

    def _listen_loop(self) -> None:
        import speech_recognition as sr  # type: ignore

        recognizer = sr.Recognizer()
        recognizer.energy_threshold = self.energy_threshold
        recognizer.pause_threshold = self.pause_threshold

        mic = sr.Microphone(device_index=self.device_index)

        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)

        while self._running:
            try:
                with mic as source:
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=4)
                text = recognizer.recognize_google(audio, language=self.language).lower()
                logger.debug("Heard: %s", text)
                if any(kw in text for kw in self.keywords):
                    logger.info("Wake word detected (fallback): %s", text)
                    if self._callback:
                        self._callback()
            except Exception:
                # timeout / recognition failure — just keep going
                continue


# ---------------------------------------------------------------------------
# Unified detector
# ---------------------------------------------------------------------------

class WakeWordDetector:
    """Unified wake-word detector: tries Porcupine, falls back to speech_recognition."""

    def __init__(
        self,
        porcupine_access_key: Optional[str] = None,
        porcupine_keyword_path: Optional[str] = None,
        sensitivity: float = 0.6,
        device_index: Optional[int] = None,
        fallback_keywords: Optional[list[str]] = None,
    ) -> None:
        self._detector: PorcupineWakeWord | FallbackWakeWord
        if porcupine_access_key and porcupine_keyword_path:
            try:
                self._detector = PorcupineWakeWord(
                    access_key=porcupine_access_key,
                    keyword_path=porcupine_keyword_path,
                    sensitivity=sensitivity,
                    device_index=device_index or -1,
                )
                logger.info("Using Porcupine wake-word engine")
                return
            except Exception:
                logger.warning("Porcupine init failed, falling back")
        self._detector = FallbackWakeWord(
            keywords=fallback_keywords,
            device_index=device_index,
        )

    def start(self, callback: Callable[[], None]) -> None:
        self._detector.start(callback)

    def stop(self) -> None:
        self._detector.stop()
