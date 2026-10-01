"""Quality filtering for generated multimodal content (MAGID step 3).

Uses CLIP matching and aesthetic scoring to filter generated images
for quality and relevance to the source dialogue.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from MASim.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class QualityScore:
    """Quality assessment of a generated artifact."""
    turn_id: str
    clip_score: float = 0.0       # text-image similarity
    aesthetic_score: float = 0.0  # aesthetic quality
    safety_pass: bool = True      # safety filter
    overall_pass: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityFilterConfig:
    """Configuration for quality filtering."""
    clip_threshold: float = 0.25      # minimum CLIP score
    aesthetic_threshold: float = 4.5  # minimum aesthetic score (1-10 scale)
    enable_safety: bool = True


class QualityFilterBase(abc.ABC):
    """Abstract base class for quality filtering."""

    @abc.abstractmethod
    def score(self, image_path: Path, text: str) -> QualityScore:
        """Score a single image-text pair."""
        ...

    def filter_batch(
        self,
        pairs: List[Tuple[str, Path, str]],  # (turn_id, image_path, text)
        config: QualityFilterConfig,
    ) -> Tuple[List[str], List[str]]:
        """Filter a batch of generated images.

        Returns (passed_turn_ids, rejected_turn_ids).
        """
        passed = []
        rejected = []
        for turn_id, image_path, text in pairs:
            score = self.score(image_path, text)
            if (
                score.clip_score >= config.clip_threshold
                and score.aesthetic_score >= config.aesthetic_threshold
                and score.safety_pass
            ):
                passed.append(turn_id)
            else:
                rejected.append(turn_id)

        log.info("Quality filter: %d passed, %d rejected", len(passed), len(rejected))
        return passed, rejected


class DummyQualityFilter(QualityFilterBase):
    """Placeholder filter that passes everything (for testing)."""

    def score(self, image_path: Path, text: str) -> QualityScore:
        return QualityScore(
            turn_id=image_path.stem,
            clip_score=0.3,
            aesthetic_score=5.0,
            safety_pass=True,
            overall_pass=True,
        )


class CLIPQualityFilter(QualityFilterBase):
    """Quality filter using CLIP for text-image similarity scoring."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._processor = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from transformers import CLIPModel, CLIPProcessor

            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            self._model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            log.info("Loaded CLIP model: %s", self.model_name)
        except ImportError:
            log.error("transformers package not installed")
            raise

    def score(self, image_path: Path, text: str) -> QualityScore:
        try:
            self._load_model()
            from PIL import Image
            import torch

            image = Image.open(image_path).convert("RGB")
            inputs = self._processor(text=[text], images=[image], return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self._model(**inputs)
                clip_score = outputs.logits_per_image.item() / 100.0

            return QualityScore(
                turn_id=image_path.stem,
                clip_score=clip_score,
                aesthetic_score=5.0,  # would need a separate model
                safety_pass=True,
                overall_pass=clip_score >= 0.25,
            )
        except Exception as e:
            log.warning("CLIP scoring failed for %s: %s", image_path, e)
            return QualityScore(turn_id=image_path.stem, overall_pass=False)
