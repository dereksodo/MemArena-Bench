"""Inline image tag parsing and FLUX.2 image generation client."""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from MASim.utils.logging import get_logger

log = get_logger(__name__)

_IMAGE_TAG_RE = re.compile(r'\[IMAGE:\s*(.+?)\]')

# Style suffix appended to brief prompts to improve FLUX.1-schnell output quality.
# Applied when the raw description is under 15 words.
_FLUX_STYLE_SUFFIX = (
    ", candid smartphone photograph, natural lighting, realistic, "
    "everyday life, sharp focus, high quality"
)


def _enrich_flux_prompt(description: str) -> str:
    """Enhance a brief image description into a FLUX.1-schnell-friendly prompt.

    If the description is already detailed (>=15 words) it is returned as-is.
    Short descriptions get a photographic style suffix.
    """
    if len(description.split()) >= 15:
        return description
    return description + _FLUX_STYLE_SUFFIX


def parse_image_tags(text: str) -> List[Dict[str, str]]:
    """Extract ``[IMAGE: description]`` tags from text.

    Returns a list of dicts with ``raw`` (full match) and ``description``.
    """
    results = []
    for m in _IMAGE_TAG_RE.finditer(text):
        results.append({"raw": m.group(0), "description": m.group(1).strip()})
    return results


def strip_image_tags(text: str) -> str:
    """Remove all ``[IMAGE: ...]`` tags from text."""
    return _IMAGE_TAG_RE.sub("", text).strip()


@dataclass
class FluxImageResult:
    """Result of a FLUX.2 image generation request."""
    turn_id: str
    success: bool
    image_path: Optional[str] = None
    error: Optional[str] = None


class FluxImageClient:
    """Client for a FLUX.2 image generation endpoint (OpenAI-compatible or raw)."""

    def __init__(
        self,
        endpoint: str = "http://localhost:8100",
        output_dir: str | Path = "media/inline_images",
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        guidance: float = 7.5,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        self.steps = steps
        self.guidance = guidance

    def generate(
        self,
        turn_id: str,
        prompt: str,
        seed: int = 42,
    ) -> FluxImageResult:
        """Generate an image via FLUX.2 and save to disk."""
        payload = {
            "prompt": _enrich_flux_prompt(prompt),
            "width": self.width,
            "height": self.height,
            "num_inference_steps": self.steps,
            "guidance_scale": self.guidance,
            "seed": seed,
        }
        try:
            resp = requests.post(
                f"{self.endpoint}/generate",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()

            # Decode response: FLUX.2 may return JSON with base64 or raw bytes
            out_path = self.output_dir / f"{turn_id}.png"
            image_bytes = self._extract_image_bytes(resp)
            out_path.write_bytes(image_bytes)
            log.debug("Generated image for %s -> %s", turn_id, out_path)
            return FluxImageResult(turn_id=turn_id, success=True, image_path=str(out_path))

        except Exception as e:
            log.warning("FLUX.2 generation failed for %s: %s", turn_id, e)
            return FluxImageResult(turn_id=turn_id, success=False, error=str(e))

    @staticmethod
    def _extract_image_bytes(resp: requests.Response) -> bytes:
        """Extract raw PNG bytes from a FLUX.2 response.

        Handles two formats:
          1. JSON with ``images_b64`` array → decode first entry from base64
          2. Raw binary PNG → return as-is
        """
        content_type = resp.headers.get("content-type", "")
        if "json" in content_type or resp.content[:1] == b"{":
            data = resp.json()
            b64_list = data.get("images_b64") or data.get("images", [])
            if b64_list:
                return base64.b64decode(b64_list[0])
        # Fallback: raw image bytes
        return resp.content

    @staticmethod
    def check_health(endpoint: str, timeout: float = 3.0) -> bool:
        """Quick connectivity check — returns True if the endpoint responds."""
        try:
            resp = requests.get(
                f"{endpoint.rstrip('/')}/health",
                timeout=timeout,
            )
            return resp.status_code < 500
        except Exception:
            # Try a simple TCP connect as fallback
            try:
                resp = requests.get(endpoint.rstrip("/"), timeout=timeout)
                return resp.status_code < 500
            except Exception:
                return False
