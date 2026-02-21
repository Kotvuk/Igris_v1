"""
IGRIS Speech Recognition Module.

Captures microphone audio with VAD (webrtcvad), detects silence to segment
utterances, and transcribes via Groq Whisper V3 Turbo API.
"""

from __future__ import annotations

import io
import logging
import os
import struct
import tempfile
import time
import wave
from typing import Optional

import httpx
import numpy as np

logger = logging.getLogger("igris.voice.listener")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16_000
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit
FRAME_DURATION_MS = 30  # webrtcvad supports 10/20/30 ms
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)  # 480 samples

# Silence detection
SILENCE_FRAMES_THRESHOLD = 40  # ~1.2 s of silence → end of utterance
MIN_VOICED_FRAMES = 10  # ignore very short bursts


# ---------------------------------------------------------------------------
# VAD helper
# ---------------------------------------------------------------------------

class VoiceActivityDetector:
    """Wraps webrtcvad for frame-level voice activity detection."""

    def __init__(self, aggressiveness: int = 2) -> None:
        """
        Args:
            aggressiveness: 0–3 (3 = most aggressive filtering of non-speech).
        """
        import webrtcvad  # type: ignore

        self.vad = webrtcvad.Vad(aggressiveness)

    def is_speech(self, frame: bytes) -> bool:
        """Return True if *frame* contains speech."""
        return self.vad.is_speech(frame, SAMPLE_RATE)


# ---------------------------------------------------------------------------
# Microphone capture
# ---------------------------------------------------------------------------

class MicrophoneStream:
    """Capture raw PCM from the microphone using sounddevice."""

    def __init__(self, device: Optional[int] = None) -> None:
        self.device = device
        self._buffer: list[bytes] = []
        self._stream = None

    def start(self) -> None:
        import sounddevice as sd  # type: ignore

        self._buffer.clear()
        self._stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=FRAME_SIZE,
            dtype="int16",
            channels=CHANNELS,
            device=self.device,
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def read_frame(self) -> Optional[bytes]:
        """Return one VAD-sized frame or None."""
        if self._buffer:
            return self._buffer.pop(0)
        return None

    def _audio_callback(self, indata: bytes, frames: int, time_info, status) -> None:  # noqa: ANN001
        if status:
            logger.debug("sounddevice status: %s", status)
        self._buffer.append(bytes(indata))


# ---------------------------------------------------------------------------
# Groq Whisper transcription
# ---------------------------------------------------------------------------

GROQ_API_URL = "https://api.groq.com/openai/v1/audio/transcriptions"


async def transcribe_audio(
    audio_bytes: bytes,
    api_key: Optional[str] = None,
    language: str = "ru",
    model: str = "whisper-large-v3-turbo",
    prompt: str = "Игрис, сэр, команда",
) -> str:
    """
    Send WAV audio to Groq Whisper and return transcription text.

    Args:
        audio_bytes: Raw 16-bit PCM at 16 kHz mono.
        api_key: Groq API key (falls back to ``GROQ_API_KEY`` env var).
        language: ISO language hint.
        model: Whisper model identifier on Groq.
        prompt: Prompt/context hint for better recognition.

    Returns:
        Transcribed text string.
    """
    key = api_key or os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set")

    wav_buf = _pcm_to_wav(audio_bytes)

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {key}"},
            files={"file": ("audio.wav", wav_buf, "audio/wav")},
            data={
                "model": model,
                "language": language,
                "prompt": prompt,
                "response_format": "json",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("text", "").strip()


def _pcm_to_wav(pcm: bytes) -> bytes:
    """Wrap raw PCM in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# High-level listener
# ---------------------------------------------------------------------------

class SpeechListener:
    """
    Listens on the microphone, uses VAD to detect speech boundaries,
    and transcribes complete utterances via Groq Whisper.
    """

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        vad_aggressiveness: int = 2,
        device: Optional[int] = None,
        silence_threshold: int = SILENCE_FRAMES_THRESHOLD,
        min_voiced: int = MIN_VOICED_FRAMES,
        language: str = "ru",
    ) -> None:
        self.api_key = groq_api_key
        self.language = language
        self.silence_threshold = silence_threshold
        self.min_voiced = min_voiced

        self._vad = VoiceActivityDetector(aggressiveness=vad_aggressiveness)
        self._mic = MicrophoneStream(device=device)

    async def listen_once(self, timeout: float = 15.0) -> Optional[str]:
        """
        Block until an utterance is captured, then transcribe it.

        Returns:
            Transcribed text, or ``None`` on timeout / empty speech.
        """
        self._mic.start()
        try:
            pcm = self._capture_utterance(timeout)
        finally:
            self._mic.stop()

        if pcm is None:
            return None

        logger.info("Captured %.1f s of audio, transcribing…", len(pcm) / (SAMPLE_RATE * SAMPLE_WIDTH))
        text = await transcribe_audio(pcm, api_key=self.api_key, language=self.language)
        logger.info("Transcription: %s", text)
        return text or None

    def _capture_utterance(self, timeout: float) -> Optional[bytes]:
        """
        Collect voiced frames from the mic until silence is detected.

        Returns:
            Raw PCM bytes of the utterance, or None on timeout.
        """
        voiced_frames: list[bytes] = []
        silent_count = 0
        voiced_count = 0
        started = False
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            frame = self._mic.read_frame()
            if frame is None:
                time.sleep(0.005)
                continue

            is_speech = self._vad.is_speech(frame)

            if is_speech:
                voiced_frames.append(frame)
                voiced_count += 1
                silent_count = 0
                started = True
            elif started:
                voiced_frames.append(frame)  # keep trailing silence for natural end
                silent_count += 1
                if silent_count >= self.silence_threshold:
                    break
            # else: not started yet, discard silence

        if voiced_count < self.min_voiced:
            logger.debug("Too few voiced frames (%d), ignoring", voiced_count)
            return None

        return b"".join(voiced_frames)
