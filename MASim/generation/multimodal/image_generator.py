"""Text-to-image generation interface (MAGID step 2).

Abstract base class for image generation backends.
Supports SDXL, FLUX, or any diffusion model with a common interface.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from MASim.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class ImageGenRequest:
    """Request for image generation."""
    turn_id: str
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    seed: int = 42


@dataclass
class ImageGenResult:
    """Result from image generation."""
    turn_id: str
    image_path: Optional[Path] = None
    success: bool = False
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class ImageGeneratorBase(abc.ABC):
    """Abstract base class for image generation backends."""

    @abc.abstractmethod
    def generate(self, request: ImageGenRequest) -> ImageGenResult:
        """Generate a single image."""
        ...

    def generate_batch(self, requests: List[ImageGenRequest]) -> List[ImageGenResult]:
        """Generate images for a batch of requests. Override for parallel impl."""
        return [self.generate(req) for req in requests]


class DummyImageGenerator(ImageGeneratorBase):
    """Placeholder generator for testing / dry-run mode."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, request: ImageGenRequest) -> ImageGenResult:
        # Create a placeholder file
        path = self.output_dir / f"{request.turn_id}.txt"
        path.write_text(f"[PLACEHOLDER IMAGE]\nPrompt: {request.prompt}\n")
        return ImageGenResult(
            turn_id=request.turn_id,
            image_path=path,
            success=True,
            metadata={"generator": "dummy"},
        )


class DiffusionImageGenerator(ImageGeneratorBase):
    """Image generator using a diffusion model API (SDXL/FLUX).

    Expects a running inference server with an OpenAI-compatible
    image generation endpoint, or a local diffusers pipeline.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        model: str = "stabilityai/stable-diffusion-xl-base-1.0",
        output_dir: Path = Path("images"),
        device: str = "cuda",
    ):
        self.endpoint = endpoint
        self.model_name = model
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self._pipeline = None

    def _load_pipeline(self):
        """Lazy-load the diffusion pipeline."""
        if self._pipeline is not None:
            return
        try:
            from diffusers import AutoPipelineForText2Image
            import torch

            self._pipeline = AutoPipelineForText2Image.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
            ).to(self.device)
            log.info("Loaded diffusion pipeline: %s", self.model_name)
        except ImportError:
            log.error("diffusers package not installed. Install with: pip install diffusers")
            raise

    def generate(self, request: ImageGenRequest) -> ImageGenResult:
        try:
            if self.endpoint:
                return self._generate_via_endpoint(request)
            self._load_pipeline()
            import torch

            with torch.no_grad():
                image = self._pipeline(
                    prompt=request.prompt,
                    negative_prompt=request.negative_prompt or "blurry, low quality",
                    width=request.width,
                    height=request.height,
                    generator=torch.Generator(self.device).manual_seed(request.seed),
                ).images[0]

            path = self.output_dir / f"{request.turn_id}.png"
            image.save(path)
            return ImageGenResult(
                turn_id=request.turn_id,
                image_path=path,
                success=True,
                metadata={"model": self.model_name},
            )
        except Exception as e:
            log.error("Image generation failed for %s: %s", request.turn_id, e)
            return ImageGenResult(
                turn_id=request.turn_id,
                success=False,
                error=str(e),
            )

    def _generate_via_endpoint(self, request: ImageGenRequest) -> ImageGenResult:
        """Generate via the running HTTP server (serve_flux2.py protocol)."""
        import base64
        import requests as req_lib

        payload = {
            "prompt": request.prompt,
            "width": request.width,
            "height": request.height,
            "seed": request.seed,
        }
        resp = req_lib.post(f"{self.endpoint}/generate", json=payload, timeout=120)
        resp.raise_for_status()
        images_b64 = resp.json()["images_b64"]
        img_bytes = base64.b64decode(images_b64[0])
        path = self.output_dir / f"{request.turn_id}.png"
        path.write_bytes(img_bytes)
        return ImageGenResult(
            turn_id=request.turn_id,
            image_path=path,
            success=True,
            metadata={"endpoint": self.endpoint},
        )
