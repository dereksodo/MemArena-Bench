"""ActivityTopicSelector: persona-aware topic sampling for person-agent narration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from MASim.core.schema import PersonaCard
from MASim.utils.io import read_yaml
from MASim.utils.logging import get_logger

log = get_logger(__name__)

# Default path for the activity topics pool
_DEFAULT_TOPICS_PATH = Path(__file__).resolve().parent.parent / "data" / "activity_topics.yaml"


class ActivityTopicSelector:
    """Load an activity-topic pool and sample topics weighted by persona fit."""

    def __init__(
        self,
        topics_path: Path | str = _DEFAULT_TOPICS_PATH,
        seed: int = 42,
    ) -> None:
        doc = read_yaml(Path(topics_path))
        self.topics: List[Dict[str, Any]] = doc.get("topics", [])
        self._rng = np.random.default_rng(seed)
        if not self.topics:
            log.warning("Activity topic pool is empty — %s", topics_path)

    def select_topics(self, persona: PersonaCard, n: int = 1) -> List[Dict[str, Any]]:
        """Score topics against persona and softmax-sample *n* unique topics.

        Scoring heuristic:
          - Base score 0.1 for every topic (ensures coverage).
          - +1.0 if any topic category keyword appears in persona hobbies.
          - +0.5 if any topic category keyword appears in persona occupation.
          - +0.3 if any topic category keyword appears in persona values.
          - Topics with an occupation_hint that matches persona occupation get +1.0.
        """
        if not self.topics:
            return []

        n = min(n, len(self.topics))
        scores = np.full(len(self.topics), 0.1)

        hobbies_lower = " ".join(h.lower() for h in (persona.hobbies or []))
        occ_lower = (persona.occupation or "").lower()
        values_lower = " ".join(v.lower() for v in (persona.values or []))

        for i, topic in enumerate(self.topics):
            cats = topic.get("categories", [])
            for cat in cats:
                cat_l = cat.lower()
                if cat_l in hobbies_lower:
                    scores[i] += 1.0
                if cat_l in occ_lower:
                    scores[i] += 0.5
                if cat_l in values_lower:
                    scores[i] += 0.3
            occ_hint = topic.get("occupation_hint", "")
            if occ_hint and occ_hint.lower() in occ_lower:
                scores[i] += 1.0

        # Softmax for sampling probabilities
        exp_scores = np.exp(scores - scores.max())
        probs = exp_scores / exp_scores.sum()

        indices = self._rng.choice(
            len(self.topics), size=n, replace=False, p=probs,
        )
        return [self.topics[i] for i in indices]
