"""
IGRIS Lip Synchronization Module.

Maps Russian phonemes to visemes, analyses audio amplitude for mouth movement,
and produces timing data consumable by a Three.js frontend.
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger("igris.avatar.lip_sync")


# ---------------------------------------------------------------------------
# Viseme definitions (compatible with Oculus/ARKit standard)
# ---------------------------------------------------------------------------

class Viseme:
    SILENT = 0   # mouth closed
    AA = 1       # а, я
    OO = 2       # о, ё
    UU = 3       # у, ю
    EE = 4       # э, е
    II = 5       # и, ы, й
    PP = 6       # п, б, м (bilabial)
    FF = 7       # ф, в (labiodental)
    TH = 8       # т, д, н, л (dental/alveolar)
    KK = 9       # к, г, х (velar)
    SH = 10      # ш, ж, щ, ч, ц (postalveolar/affricate)
    RR = 11      # р (trill)
    SS = 12      # с, з (sibilant)


RUSSIAN_PHONEME_MAP: dict[str, int] = {
    # Vowels
    "а": Viseme.AA, "я": Viseme.AA,
    "о": Viseme.OO, "ё": Viseme.OO,
    "у": Viseme.UU, "ю": Viseme.UU,
    "э": Viseme.EE, "е": Viseme.EE,
    "и": Viseme.II, "ы": Viseme.II, "й": Viseme.II,
    # Consonants
    "п": Viseme.PP, "б": Viseme.PP, "м": Viseme.PP,
    "ф": Viseme.FF, "в": Viseme.FF,
    "т": Viseme.TH, "д": Viseme.TH, "н": Viseme.TH, "л": Viseme.TH,
    "к": Viseme.KK, "г": Viseme.KK, "х": Viseme.KK,
    "ш": Viseme.SH, "ж": Viseme.SH, "щ": Viseme.SH, "ч": Viseme.SH, "ц": Viseme.SH,
    "р": Viseme.RR,
    "с": Viseme.SS, "з": Viseme.SS,
    # Soft/hard signs → silent
    "ь": Viseme.SILENT, "ъ": Viseme.SILENT,
}


@dataclass
class VisemeFrame:
    """Single viseme keyframe for the frontend."""
    time_ms: float
    viseme: int
    weight: float  # 0.0–1.0 mouth openness


# ---------------------------------------------------------------------------
# Text-based viseme generation
# ---------------------------------------------------------------------------

def visemes_from_text(text: str, duration_ms: float) -> list[dict[str, Any]]:
    """
    Generate a viseme timeline from Russian text and expected duration.

    Distributes characters evenly across the duration and maps each to a viseme.
    Spaces produce SILENT frames for natural pauses.
    """
    frames: list[dict[str, Any]] = []
    chars = list(text.lower())
    if not chars:
        return frames

    interval = duration_ms / len(chars) if chars else 0

    for i, ch in enumerate(chars):
        t = round(i * interval, 1)
        if ch == " ":
            frames.append({"time_ms": t, "viseme": Viseme.SILENT, "weight": 0.05})
        elif ch in RUSSIAN_PHONEME_MAP:
            v = RUSSIAN_PHONEME_MAP[ch]
            # Vowels get higher weight (mouth opens more)
            w = 0.85 if v <= Viseme.II else 0.55
            frames.append({"time_ms": t, "viseme": v, "weight": w})
        # Non-mapped characters (punctuation etc.) skipped

    # Close mouth at the end
    frames.append({"time_ms": round(duration_ms, 1), "viseme": Viseme.SILENT, "weight": 0.0})
    return frames


# ---------------------------------------------------------------------------
# Audio amplitude-based lip sync
# ---------------------------------------------------------------------------

def visemes_from_audio(
    pcm_data: bytes,
    sample_rate: int = 16_000,
    frame_ms: int = 30,
    sample_width: int = 2,
) -> list[dict[str, Any]]:
    """
    Analyse raw PCM audio amplitude to produce mouth-movement data.

    Returns a list of ``{"time_ms", "viseme", "weight"}`` where viseme is
    always ``Viseme.AA`` (open mouth) and weight reflects amplitude.
    """
    samples_per_frame = int(sample_rate * frame_ms / 1000)
    fmt = f"<{samples_per_frame}h"
    frame_bytes = samples_per_frame * sample_width

    frames: list[dict[str, Any]] = []
    offset = 0
    frame_idx = 0

    # Find max amplitude for normalisation
    total_samples = len(pcm_data) // sample_width
    if total_samples == 0:
        return frames
    all_samples = np.frombuffer(pcm_data, dtype=np.int16)
    max_amp = float(np.max(np.abs(all_samples))) or 1.0

    while offset + frame_bytes <= len(pcm_data):
        chunk = pcm_data[offset : offset + frame_bytes]
        samples = np.frombuffer(chunk, dtype=np.int16)
        rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
        weight = min(1.0, rms / max_amp)

        t = frame_idx * frame_ms
        viseme = Viseme.SILENT if weight < 0.05 else Viseme.AA
        frames.append({"time_ms": t, "viseme": viseme, "weight": round(weight, 3)})

        offset += frame_bytes
        frame_idx += 1

    return frames


# ---------------------------------------------------------------------------
# Helpers for Three.js integration
# ---------------------------------------------------------------------------

def to_threejs_morphs(viseme_timeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convert viseme timeline to Three.js morph-target keyframe format.

    Output: list of ``{"time": <seconds>, "morphs": {"viseme_<N>": weight}}``
    """
    keyframes: list[dict[str, Any]] = []
    for f in viseme_timeline:
        morphs = {f"viseme_{i}": 0.0 for i in range(13)}
        morphs[f"viseme_{f['viseme']}"] = f["weight"]
        keyframes.append({
            "time": round(f["time_ms"] / 1000, 4),
            "morphs": morphs,
        })
    return keyframes
