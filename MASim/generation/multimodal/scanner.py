"""Identify augmentable utterances for multimodal generation (MAGID step 1).

Scans dialogue turns to find utterances that would naturally be
accompanied by images, audio, or other media.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from MASim.core.schema import DialogueCorpus, DialogueTurn
from MASim.ground_truth.json_parser import parse_json_object
from MASim.prompts import MULTIMODAL_SCANNER_SYSTEM as SCANNER_SYSTEM_PROMPT
from MASim.prompts import MULTIMODAL_SCANNER_USER as SCANNER_USER_TEMPLATE
from MASim.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class AugmentableUtterance:
    """A dialogue turn identified as suitable for multimodal augmentation."""
    turn_id: str
    session_id: str
    text: str
    modality: str  # "image", "audio"
    image_prompt: str = ""
    reason: str = ""


class MultimodalScanner:
    """Scan corpus for turns suitable for multimodal augmentation."""

    def __init__(self, llm_client: Any):
        self.llm_client = llm_client

    def scan(
        self,
        corpus: DialogueCorpus,
        max_candidates: int = 500,
        dry_run: bool = False,
    ) -> List[AugmentableUtterance]:
        """Identify augmentable utterances in the corpus.

        Returns list of AugmentableUtterance candidates.
        """
        # Collect all non-trivial turns
        all_turns: List[DialogueTurn] = []
        for session in corpus.sessions:
            for turn in session.turns:
                if len(turn.text.split()) >= 5:
                    all_turns.append(turn)

        # Limit to max_candidates
        if len(all_turns) > max_candidates:
            import numpy as np
            rng = np.random.default_rng(42)
            indices = rng.choice(len(all_turns), size=max_candidates, replace=False)
            all_turns = [all_turns[i] for i in sorted(indices)]

        if dry_run:
            # Return a subset as placeholder candidates
            return [
                AugmentableUtterance(
                    turn_id=t.turn_id,
                    session_id=t.session_id,
                    text=t.text,
                    modality="image",
                    image_prompt=f"[DRY RUN] image for: {t.text[:50]}",
                )
                for t in all_turns[:10]
            ]

        # Batch LLM scan
        tasks = [
            {
                "system": SCANNER_SYSTEM_PROMPT,
                "user": SCANNER_USER_TEMPLATE.format(
                    speaker=turn.speaker_id, text=turn.text
                ),
                "tags": {"phase": "multimodal_scan"},
            }
            for turn in all_turns
        ]

        log.info("Scanning %d turns for multimodal augmentation...", len(tasks))
        responses = self.llm_client.generate_batch(tasks)

        candidates = []
        for turn, resp in zip(all_turns, responses):
            try:
                data = parse_json_object(resp)
                if data.get("augmentable"):
                    candidates.append(AugmentableUtterance(
                        turn_id=turn.turn_id,
                        session_id=turn.session_id,
                        text=turn.text,
                        modality=data.get("modality", "image"),
                        image_prompt=data.get("image_prompt", ""),
                        reason=data.get("reason", ""),
                    ))
            except (ValueError, TypeError):
                continue

        log.info("Found %d augmentable utterances from %d turns", len(candidates), len(all_turns))
        return candidates
