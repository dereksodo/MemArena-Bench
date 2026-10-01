"""Qwen3-TTS voice generation client (OpenAI-compatible vllm-omni API)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

from MASim.utils.logging import get_logger

log = get_logger(__name__)

# Default model ID used by the vllm-omni container
_DEFAULT_MODEL = "/models/Qwen3-TTS-12Hz-1.7B-CustomVoice"

# Voices supported by this Qwen3-TTS deployment
SUPPORTED_VOICES = [
    "aiden", "dylan", "eric", "ono_anna", "ryan",
    "serena", "sohee", "uncle_fu", "vivian",
]


@dataclass
class TTSResult:
    """Result of a TTS generation request."""
    turn_id: str
    success: bool
    audio_path: Optional[str] = None
    error: Optional[str] = None


class Qwen3TTSClient:
    """Client for a Qwen3-TTS endpoint served via vllm-omni (OpenAI-compatible)."""

    def __init__(
        self,
        endpoint: str = "http://localhost:8200",
        output_dir: str | Path = "media/voice_audio",
        model: str = _DEFAULT_MODEL,
        language: str = "English",
    ):
        self.endpoint = endpoint.rstrip("/")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.language = language

    @staticmethod
    def _map_speaker(speaker: str) -> str:
        """Map any speaker name/slug to a supported voice via stable hash."""
        idx = int(hashlib.md5(speaker.lower().encode()).hexdigest(), 16) % len(SUPPORTED_VOICES)
        return SUPPORTED_VOICES[idx]

    def generate(
        self,
        turn_id: str,
        text: str,
        speaker: str = "vivian",
        instruct: str = "",
    ) -> TTSResult:
        """Generate a WAV file from text via OpenAI-compatible /v1/audio/speech."""
        mapped_speaker = self._map_speaker(speaker)
        payload = {
            "model": self.model,
            "input": text,
            "voice": mapped_speaker,
        }
        try:
            resp = requests.post(
                f"{self.endpoint}/v1/audio/speech",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()

            # Detect JSON error bodies (some servers return 200 with {"error": ...})
            content_type = resp.headers.get("content-type", "")
            if "json" in content_type or (resp.content and resp.content[:1] == b"{"):
                try:
                    data = resp.json()
                    if "error" in data:
                        err = data["error"]
                        msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                        raise ValueError(f"TTS API error: {msg}")
                except (json.JSONDecodeError, ValueError) as exc:
                    if "TTS API error" in str(exc):
                        raise

            out_path = self.output_dir / f"{turn_id}.wav"
            out_path.write_bytes(resp.content)
            log.debug("Generated TTS for %s (voice=%s) -> %s (%d bytes)",
                      turn_id, mapped_speaker, out_path, len(resp.content))
            return TTSResult(turn_id=turn_id, success=True, audio_path=str(out_path))

        except Exception as e:
            log.warning("Qwen3-TTS generation failed for %s: %s", turn_id, e)
            return TTSResult(turn_id=turn_id, success=False, error=str(e))

    @staticmethod
    def check_health(endpoint: str, timeout: float = 3.0) -> bool:
        """Quick connectivity check — returns True if the endpoint responds."""
        try:
            resp = requests.get(
                f"{endpoint.rstrip('/')}/v1/models",
                timeout=timeout,
            )
            return resp.status_code < 500
        except Exception:
            try:
                resp = requests.get(
                    f"{endpoint.rstrip('/')}/health",
                    timeout=timeout,
                )
                return resp.status_code < 500
            except Exception:
                return False
