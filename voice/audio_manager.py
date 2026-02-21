"""
IGRIS Audio I/O Manager.

Handles microphone and speaker device selection, audio playback via
pygame.mixer or sounddevice, and volume control.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("igris.voice.audio_manager")


@dataclass
class AudioDevice:
    """Describes an audio device."""
    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_sample_rate: float
    is_input: bool = False
    is_output: bool = False


class AudioManager:
    """
    Central audio I/O controller.

    Provides device enumeration, selection, playback, and volume control.
    """

    def __init__(self) -> None:
        self._input_device: Optional[int] = None
        self._output_device: Optional[int] = None
        self._volume: float = 0.8  # 0.0–1.0
        self._pygame_initialized: bool = False

    # -- device enumeration --------------------------------------------------

    @staticmethod
    def list_devices() -> list[AudioDevice]:
        """Return all available audio devices."""
        import sounddevice as sd  # type: ignore

        devices: list[AudioDevice] = []
        for i, d in enumerate(sd.query_devices()):
            devices.append(AudioDevice(
                index=i,
                name=d["name"],
                max_input_channels=d["max_input_channels"],
                max_output_channels=d["max_output_channels"],
                default_sample_rate=d["default_samplerate"],
                is_input=d["max_input_channels"] > 0,
                is_output=d["max_output_channels"] > 0,
            ))
        return devices

    def list_input_devices(self) -> list[AudioDevice]:
        """Return only input (microphone) devices."""
        return [d for d in self.list_devices() if d.is_input]

    def list_output_devices(self) -> list[AudioDevice]:
        """Return only output (speaker) devices."""
        return [d for d in self.list_devices() if d.is_output]

    # -- device selection ----------------------------------------------------

    def select_input(self, device_index: int) -> None:
        """Select microphone device by index."""
        devs = self.list_input_devices()
        if not any(d.index == device_index for d in devs):
            raise ValueError(f"No input device with index {device_index}")
        self._input_device = device_index
        logger.info("Selected input device %d", device_index)

    def select_output(self, device_index: int) -> None:
        """Select speaker device by index."""
        devs = self.list_output_devices()
        if not any(d.index == device_index for d in devs):
            raise ValueError(f"No output device with index {device_index}")
        self._output_device = device_index
        logger.info("Selected output device %d", device_index)

    @property
    def input_device(self) -> Optional[int]:
        return self._input_device

    @property
    def output_device(self) -> Optional[int]:
        return self._output_device

    # -- volume --------------------------------------------------------------

    @property
    def volume(self) -> float:
        return self._volume

    @volume.setter
    def volume(self, value: float) -> None:
        self._volume = max(0.0, min(1.0, value))
        if self._pygame_initialized:
            try:
                import pygame  # type: ignore
                pygame.mixer.music.set_volume(self._volume)
            except Exception:
                pass
        logger.debug("Volume set to %.0f%%", self._volume * 100)

    # -- playback ------------------------------------------------------------

    def _ensure_pygame(self) -> None:
        if not self._pygame_initialized:
            import pygame  # type: ignore
            pygame.mixer.init()
            pygame.mixer.music.set_volume(self._volume)
            self._pygame_initialized = True

    def play_file(self, path: str | Path, blocking: bool = True) -> None:
        """
        Play an audio file (mp3/wav/ogg).

        Args:
            path: Path to the audio file.
            blocking: If True, wait until playback finishes.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        try:
            self._play_pygame(path, blocking)
        except ImportError:
            self._play_sounddevice(path, blocking)

    def _play_pygame(self, path: Path, blocking: bool) -> None:
        import pygame  # type: ignore

        self._ensure_pygame()
        pygame.mixer.music.load(str(path))
        pygame.mixer.music.play()
        if blocking:
            import time
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)

    def _play_sounddevice(self, path: Path, blocking: bool) -> None:
        import sounddevice as sd  # type: ignore
        import soundfile as sf  # type: ignore

        data, samplerate = sf.read(str(path), dtype="float32")
        data = data * self._volume
        sd.play(data, samplerate, device=self._output_device)
        if blocking:
            sd.wait()

    def play_array(
        self,
        audio: np.ndarray,
        samplerate: int = 16_000,
        blocking: bool = True,
    ) -> None:
        """Play a numpy audio array."""
        import sounddevice as sd  # type: ignore

        scaled = (audio * self._volume).astype(audio.dtype)
        sd.play(scaled, samplerate, device=self._output_device)
        if blocking:
            sd.wait()

    def stop(self) -> None:
        """Stop any current playback."""
        try:
            import pygame  # type: ignore
            if self._pygame_initialized:
                pygame.mixer.music.stop()
        except ImportError:
            pass
        try:
            import sounddevice as sd  # type: ignore
            sd.stop()
        except ImportError:
            pass
