"""
IGRIS Media Generation Skill.

Prompt engineering for images, style presets, batch generation,
and image description/analysis.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import httpx

logger = logging.getLogger("igris.skills.media")


# ---------------------------------------------------------------------------
# Style presets
# ---------------------------------------------------------------------------

class StylePreset(Enum):
    SOLO_LEVELING = "solo_leveling"
    ANIME = "anime"
    PHOTOREALISTIC = "photorealistic"
    CYBERPUNK = "cyberpunk"
    FANTASY = "fantasy"
    MINIMALIST = "minimalist"
    WATERCOLOR = "watercolor"
    PIXEL_ART = "pixel_art"
    COMIC = "comic"
    DARK_FANTASY = "dark_fantasy"


STYLE_PROMPTS: dict[StylePreset, str] = {
    StylePreset.SOLO_LEVELING: (
        "in the style of Solo Leveling manhwa, dramatic purple-blue shadows, "
        "glowing eyes, dark aura particles, Korean manhwa art style, "
        "high contrast, dynamic lighting, detailed armor"
    ),
    StylePreset.ANIME: (
        "anime art style, vibrant colors, clean linework, "
        "detailed eyes, soft shading, Studio Ghibli quality"
    ),
    StylePreset.PHOTOREALISTIC: (
        "photorealistic, ultra-detailed, 8K resolution, "
        "natural lighting, shallow depth of field, DSLR quality"
    ),
    StylePreset.CYBERPUNK: (
        "cyberpunk aesthetic, neon lights, rain-soaked streets, "
        "holographic displays, dark futuristic atmosphere, Blade Runner style"
    ),
    StylePreset.FANTASY: (
        "epic fantasy art, detailed environment, magical atmosphere, "
        "volumetric lighting, concept art quality, Greg Rutkowski style"
    ),
    StylePreset.MINIMALIST: (
        "minimalist design, clean lines, limited color palette, "
        "negative space, modern aesthetic, elegant simplicity"
    ),
    StylePreset.WATERCOLOR: (
        "watercolor painting style, soft edges, color bleeding, "
        "paper texture, delicate brushstrokes, artistic"
    ),
    StylePreset.PIXEL_ART: (
        "pixel art style, 16-bit aesthetic, limited palette, "
        "crisp pixels, retro gaming style, detailed sprites"
    ),
    StylePreset.COMIC: (
        "comic book style, bold outlines, halftone dots, "
        "dynamic composition, speech bubbles, vibrant colors"
    ),
    StylePreset.DARK_FANTASY: (
        "dark fantasy art, ominous atmosphere, gothic architecture, "
        "muted colors with accent highlights, Berserk manga influence"
    ),
}

# Quality boosters appended to all prompts
_QUALITY_SUFFIX = (
    "masterpiece, best quality, highly detailed, "
    "sharp focus, professional artwork"
)

# Negative prompt for filtering
_NEGATIVE_PROMPT = (
    "low quality, blurry, deformed, ugly, bad anatomy, "
    "bad hands, missing fingers, watermark, text, signature, "
    "jpeg artifacts, cropped"
)


# ---------------------------------------------------------------------------
# Prompt engineering
# ---------------------------------------------------------------------------

def engineer_prompt(
    description: str,
    style: Optional[StylePreset] = None,
    extra_tags: Optional[list[str]] = None,
    negative: bool = True,
) -> dict[str, str]:
    """
    Build an optimised image generation prompt.

    Returns dict with ``"prompt"`` and optionally ``"negative_prompt"``.
    """
    parts = [description.strip()]

    if style and style in STYLE_PROMPTS:
        parts.append(STYLE_PROMPTS[style])

    if extra_tags:
        parts.extend(extra_tags)

    parts.append(_QUALITY_SUFFIX)
    prompt = ", ".join(parts)

    result: dict[str, str] = {"prompt": prompt}
    if negative:
        result["negative_prompt"] = _NEGATIVE_PROMPT
    return result


# ---------------------------------------------------------------------------
# Generation (OpenAI-compatible API)
# ---------------------------------------------------------------------------

@dataclass
class GeneratedImage:
    url: Optional[str] = None
    b64_data: Optional[str] = None
    revised_prompt: str = ""
    local_path: Optional[Path] = None


async def generate_image(
    prompt: str,
    negative_prompt: str = "",
    api_key: Optional[str] = None,
    api_url: str = "https://api.openai.com/v1/images/generations",
    model: str = "dall-e-3",
    size: str = "1024x1024",
    quality: str = "standard",
    n: int = 1,
) -> list[GeneratedImage]:
    """
    Generate images using an OpenAI-compatible API.
    """
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")

    full_prompt = prompt
    if negative_prompt:
        full_prompt += f"\n\nAvoid: {negative_prompt}"

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            api_url,
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": model,
                "prompt": full_prompt,
                "n": n,
                "size": size,
                "quality": quality,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    results: list[GeneratedImage] = []
    for item in data.get("data", []):
        results.append(GeneratedImage(
            url=item.get("url"),
            b64_data=item.get("b64_json"),
            revised_prompt=item.get("revised_prompt", ""),
        ))
    return results


async def batch_generate(
    description: str,
    style: Optional[StylePreset] = None,
    variations: int = 3,
    **kwargs: Any,
) -> list[GeneratedImage]:
    """
    Generate multiple variations of an image with slight prompt tweaks.
    """
    base = engineer_prompt(description, style)
    variation_tags = [
        ["different angle", "alternative composition"],
        ["close-up view", "detailed focus"],
        ["wide shot", "environmental context"],
        ["dramatic lighting", "high contrast"],
        ["soft lighting", "pastel tones"],
    ]

    tasks = []
    for i in range(variations):
        extra = variation_tags[i % len(variation_tags)]
        prompt_data = engineer_prompt(description, style, extra_tags=extra)
        tasks.append(generate_image(
            prompt=prompt_data["prompt"],
            negative_prompt=prompt_data.get("negative_prompt", ""),
            n=1,
            **kwargs,
        ))

    results_nested = await asyncio.gather(*tasks, return_exceptions=True)
    images: list[GeneratedImage] = []
    for r in results_nested:
        if isinstance(r, list):
            images.extend(r)
        elif isinstance(r, Exception):
            logger.error("Batch generation error: %s", r)
    return images


# ---------------------------------------------------------------------------
# Image analysis / description
# ---------------------------------------------------------------------------

async def describe_image(
    image_url: str,
    api_key: Optional[str] = None,
    model: str = "gpt-4o",
    detail: str = "high",
) -> str:
    """
    Analyse and describe an image using a vision model.
    """
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image in detail. Include style, mood, composition, colors, and notable elements."},
                            {"type": "image_url", "image_url": {"url": image_url, "detail": detail}},
                        ],
                    }
                ],
                "max_tokens": 1000,
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
