"""TTS generation with persona voice profiles.

Abstract base class for text-to-speech backends,
with persona-specific voice characteristics.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from MASim.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class VoiceProfile:
    """Voice characteristics for a persona."""
    agent_id: str
    speaker_id: str = ""      # model-specific speaker identifier
    language: str = "en"
    pitch: float = 1.0        # relative pitch adjustment
    speed: float = 1.0        # speaking rate
    style: str = "neutral"    # emotional style


@dataclass
class AudioGenRequest:
    """Request for audio generation."""
    turn_id: str
    text: str
    voice_profile: VoiceProfile
    sample_rate: int = 22050


@dataclass
class AudioGenResult:
    """Result from audio generation."""
    turn_id: str
    audio_path: Optional[Path] = None
    duration_seconds: float = 0.0
    success: bool = False
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class AudioGeneratorBase(abc.ABC):
    """Abstract base class for TTS backends."""

    @abc.abstractmethod
    def generate(self, request: AudioGenRequest) -> AudioGenResult:
        """Generate audio for a single request."""
        ...

    def generate_batch(self, requests: List[AudioGenRequest]) -> List[AudioGenResult]:
        """Generate audio for a batch of requests."""
        return [self.generate(req) for req in requests]


class DummyAudioGenerator(AudioGeneratorBase):
    """Placeholder TTS for testing / dry-run mode."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, request: AudioGenRequest) -> AudioGenResult:
        path = self.output_dir / f"{request.turn_id}.txt"
        path.write_text(
            f"[PLACEHOLDER AUDIO]\n"
            f"Text: {request.text}\n"
            f"Voice: {request.voice_profile.agent_id}\n"
        )
        return AudioGenResult(
            turn_id=request.turn_id,
            audio_path=path,
            duration_seconds=len(request.text.split()) * 0.4,  # rough estimate
            success=True,
            metadata={"generator": "dummy"},
        )
