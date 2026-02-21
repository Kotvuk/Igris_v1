"""Media Generation - images and video via API."""

import httpx
import json
import base64
import os
import uuid
import time
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field

logger = logging.getLogger("igris.media")


@dataclass
class GeneratedMedia:
    id: str
    type: str  # "image" or "video"
    path: str
    prompt: str
    enhanced_prompt: str
    provider: str
    model: str
    width: int
    height: int
    created_at: float
    metadata: Dict


class ImageGenerator:
    """Generate images via various API providers."""

    @staticmethod
    def generate_huggingface(
        prompt: str,
        model: str = "black-forest-labs/FLUX.1-schnell",
        api_key: Optional[str] = None,
    ) -> bytes:
        """Generate image via HuggingFace Inference API."""
        key = api_key or os.environ.get("HUGGINGFACE_API_KEY", "")
        if not key:
            raise ValueError("HUGGINGFACE_API_KEY not set")
        url = f"https://api-inference.huggingface.co/models/{model}"
        resp = httpx.post(
            url,
            headers={"Authorization": f"Bearer {key}"},
            json={"inputs": prompt},
            timeout=120,
        )
        if resp.status_code == 503:
            estimated = resp.json().get("estimated_time", 60)
            logger.info("Model loading, waiting %.0fs...", estimated)
            time.sleep(min(estimated, 120))
            resp = httpx.post(
                url,
                headers={"Authorization": f"Bearer {key}"},
                json={"inputs": prompt},
                timeout=180,
            )
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if "json" in content_type:
            data = resp.json()
            if isinstance(data, list) and data and "generated_text" in data[0]:
                raise RuntimeError(f"Model returned text instead of image: {data[0]['generated_text'][:100]}")
            raise RuntimeError(f"Unexpected JSON response: {str(data)[:200]}")
        return resp.content

    @staticmethod
    def generate_replicate(
        prompt: str,
        model: str = "stability-ai/sdxl:c221b2b8ef527988fb59bf24a8b97c4561f1c671f73bd389f866bfb27c061316",
        api_key: Optional[str] = None,
    ) -> bytes:
        """Generate image via Replicate API."""
        key = api_key or os.environ.get("REPLICATE_API_KEY", "")
        if not key:
            raise ValueError("REPLICATE_API_KEY not set")
        create_resp = httpx.post(
            "https://api.replicate.com/v1/predictions",
            headers={"Authorization": f"Token {key}", "Content-Type": "application/json"},
            json={"version": model.split(":")[-1] if ":" in model else model, "input": {"prompt": prompt}},
            timeout=30,
        )
        create_resp.raise_for_status()
        prediction = create_resp.json()
        poll_url = prediction.get("urls", {}).get("get", f"https://api.replicate.com/v1/predictions/{prediction['id']}")

        for _ in range(120):
            time.sleep(2)
            poll_resp = httpx.get(poll_url, headers={"Authorization": f"Token {key}"}, timeout=15)
            poll_resp.raise_for_status()
            status_data = poll_resp.json()
            status = status_data.get("status")
            if status == "succeeded":
                output = status_data.get("output")
                img_url = output[0] if isinstance(output, list) else output
                img_resp = httpx.get(img_url, timeout=60, follow_redirects=True)
                img_resp.raise_for_status()
                return img_resp.content
            elif status == "failed":
                raise RuntimeError(f"Replicate prediction failed: {status_data.get('error', 'unknown')}")
            elif status == "canceled":
                raise RuntimeError("Replicate prediction was canceled")
        raise TimeoutError("Replicate prediction timed out after 240s")

    @staticmethod
    def generate_together(
        prompt: str,
        model: str = "stabilityai/stable-diffusion-xl-base-1.0",
        api_key: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
    ) -> bytes:
        """Generate image via Together AI API."""
        key = api_key or os.environ.get("TOGETHER_API_KEY", "")
        if not key:
            raise ValueError("TOGETHER_API_KEY not set")
        resp = httpx.post(
            "https://api.together.xyz/v1/images/generations",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": model, "prompt": prompt, "width": width, "height": height, "n": 1, "response_format": "b64_json"},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        b64_data = data["data"][0]["b64_json"]
        return base64.b64decode(b64_data)


class VideoGenerator:
    """Generate video via API providers."""

    @staticmethod
    def generate_replicate(
        prompt: str,
        image_path: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = "wan-video/wan2.1:latest",
    ) -> bytes:
        """Generate video via Replicate (Wan2.1 model)."""
        key = api_key or os.environ.get("REPLICATE_API_KEY", "")
        if not key:
            raise ValueError("REPLICATE_API_KEY not set")

        input_data: Dict[str, Any] = {"prompt": prompt}
        if image_path:
            img_path = Path(image_path).expanduser().resolve()
            if img_path.exists():
                img_bytes = img_path.read_bytes()
                b64 = base64.b64encode(img_bytes).decode()
                mime = "image/png" if img_path.suffix == ".png" else "image/jpeg"
                input_data["image"] = f"data:{mime};base64,{b64}"

        version = model.split(":")[-1] if ":" in model else model
        create_resp = httpx.post(
            "https://api.replicate.com/v1/predictions",
            headers={"Authorization": f"Token {key}", "Content-Type": "application/json"},
            json={"version": version, "input": input_data},
            timeout=30,
        )
        create_resp.raise_for_status()
        prediction = create_resp.json()
        poll_url = prediction.get("urls", {}).get("get", f"https://api.replicate.com/v1/predictions/{prediction['id']}")

        for _ in range(300):
            time.sleep(3)
            poll_resp = httpx.get(poll_url, headers={"Authorization": f"Token {key}"}, timeout=15)
            poll_resp.raise_for_status()
            status_data = poll_resp.json()
            status = status_data.get("status")
            if status == "succeeded":
                output = status_data.get("output")
                vid_url = output[0] if isinstance(output, list) else output
                vid_resp = httpx.get(vid_url, timeout=120, follow_redirects=True)
                vid_resp.raise_for_status()
                return vid_resp.content
            elif status in ("failed", "canceled"):
                raise RuntimeError(f"Video generation {status}: {status_data.get('error', 'unknown')}")
        raise TimeoutError("Video generation timed out")


class MediaManager:
    """High-level media generation manager with gallery support."""

    def __init__(self, gallery_dir: str = "~/.igris/gallery") -> None:
        self.gallery_dir = Path(gallery_dir).expanduser().resolve()
        self.gallery_dir.mkdir(parents=True, exist_ok=True)
        self.image_gen = ImageGenerator()
        self.video_gen = VideoGenerator()

    def generate_image(
        self,
        prompt: str,
        provider: str = "huggingface",
        settings: Optional[Dict] = None,
    ) -> GeneratedMedia:
        """Generate an image and save to gallery."""
        settings = settings or {}
        api_key = settings.get("api_key")
        model = settings.get("model", "black-forest-labs/FLUX.1-schnell")
        enhanced = settings.get("enhanced_prompt", prompt)

        if provider == "huggingface":
            img_bytes = self.image_gen.generate_huggingface(enhanced, model=model, api_key=api_key)
        elif provider == "replicate":
            img_bytes = self.image_gen.generate_replicate(enhanced, model=model, api_key=api_key)
        elif provider == "together":
            img_bytes = self.image_gen.generate_together(
                enhanced, model=model, api_key=api_key,
                width=settings.get("width", 1024), height=settings.get("height", 1024),
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

        media_id = str(uuid.uuid4())[:12]
        ext = ".png"
        img_path = self.gallery_dir / f"{media_id}{ext}"
        img_path.write_bytes(img_bytes)

        width, height = self._get_image_dimensions(img_bytes)
        media = GeneratedMedia(
            id=media_id, type="image", path=str(img_path),
            prompt=prompt, enhanced_prompt=enhanced,
            provider=provider, model=model,
            width=width, height=height,
            created_at=time.time(), metadata=settings,
        )
        self._save_metadata(media)
        return media

    def generate_video(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        provider: str = "replicate",
        settings: Optional[Dict] = None,
    ) -> GeneratedMedia:
        """Generate a video and save to gallery."""
        settings = settings or {}
        api_key = settings.get("api_key")
        model = settings.get("model", "wan-video/wan2.1:latest")
        enhanced = settings.get("enhanced_prompt", prompt)

        vid_bytes = self.video_gen.generate_replicate(enhanced, image_path=image_path, api_key=api_key, model=model)

        media_id = str(uuid.uuid4())[:12]
        vid_path = self.gallery_dir / f"{media_id}.mp4"
        vid_path.write_bytes(vid_bytes)

        media = GeneratedMedia(
            id=media_id, type="video", path=str(vid_path),
            prompt=prompt, enhanced_prompt=enhanced,
            provider=provider, model=model,
            width=0, height=0,
            created_at=time.time(), metadata=settings,
        )
        self._save_metadata(media)
        return media

    def enhance_prompt(self, prompt: str, llm_client: Any) -> str:
        """Use an LLM to enhance the image generation prompt."""
        system = (
            "You are an expert at writing prompts for image generation models. "
            "Enhance the following prompt to be more detailed and descriptive. "
            "Add style, lighting, composition details. Keep it concise (under 200 words). "
            "Return ONLY the enhanced prompt, nothing else."
        )
        try:
            if hasattr(llm_client, "chat"):
                resp = llm_client.chat(messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ])
                return resp if isinstance(resp, str) else str(resp)
            return prompt
        except Exception:
            logger.warning("Prompt enhancement failed, using original")
            return prompt

    def list_gallery(self, filter_type: Optional[str] = None, limit: int = 50) -> List[GeneratedMedia]:
        """List items in the gallery."""
        items: List[GeneratedMedia] = []
        for meta_file in sorted(self.gallery_dir.glob("*.meta.json"), reverse=True):
            if len(items) >= limit:
                break
            try:
                data = json.loads(meta_file.read_text())
                media = GeneratedMedia(**data)
                if filter_type and media.type != filter_type:
                    continue
                items.append(media)
            except Exception:
                continue
        return items

    def _save_metadata(self, media: GeneratedMedia) -> None:
        meta_path = self.gallery_dir / f"{media.id}.meta.json"
        meta_path.write_text(json.dumps({
            "id": media.id, "type": media.type, "path": media.path,
            "prompt": media.prompt, "enhanced_prompt": media.enhanced_prompt,
            "provider": media.provider, "model": media.model,
            "width": media.width, "height": media.height,
            "created_at": media.created_at, "metadata": media.metadata,
        }, ensure_ascii=False, indent=2))

    @staticmethod
    def _get_image_dimensions(img_bytes: bytes) -> tuple:
        try:
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(img_bytes))
            return img.size
        except Exception:
            return (0, 0)
