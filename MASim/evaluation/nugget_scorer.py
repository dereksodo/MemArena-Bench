"""Nugget-based decomposition and coverage scoring.

Decomposes reference answers into atomic nuggets (facts), then
measures how many nuggets the system response covers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from MASim.ground_truth.json_parser import parse_llm_json
from MASim.prompts import (
    NUGGET_DECOMPOSE_SYSTEM as DECOMPOSE_SYSTEM_PROMPT,
    NUGGET_DECOMPOSE_USER as DECOMPOSE_USER_TEMPLATE,
    NUGGET_MATCH_SYSTEM as MATCH_SYSTEM_PROMPT,
    NUGGET_MATCH_USER as MATCH_USER_TEMPLATE,
)
from MASim.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class NuggetResult:
    """Result of nugget-based scoring."""
    nuggets: List[str] = field(default_factory=list)
    matched: List[bool] = field(default_factory=list)
    recall: float = 0.0
    precision: float = 0.0
    f1: float = 0.0


class NuggetScorer:
    """Score responses by decomposing into atomic nuggets and checking coverage."""

    def __init__(self, llm_client: Any):
        self.llm_client = llm_client

    def score(self, response: str, reference: str) -> NuggetResult:
        """Score a response against a reference using nugget decomposition.

        Steps:
        1. Decompose reference into atomic nuggets
        2. Check each nugget against the response
        3. Compute recall, precision, F1
        """
        nuggets = self._decompose(reference)
        if not nuggets:
            return NuggetResult(recall=1.0, precision=1.0, f1=1.0)

        matched = self._match_nuggets(nuggets, response)
        return self._compute_result(nuggets, matched)

    def score_batch(self, responses: List[str], references: List[str]) -> List[NuggetResult]:
        """Batch score: decompose all references in one batch, then match all nuggets in one batch."""
        # Phase 1: batch decompose all references
        decompose_tasks = [
            {
                "system": DECOMPOSE_SYSTEM_PROMPT,
                "user": DECOMPOSE_USER_TEMPLATE.format(reference=ref),
                "tags": {"phase": "eval_nugget_decompose"},
            }
            for ref in references
        ]
        log.info("Batch-decomposing %d references into nuggets...", len(decompose_tasks))
        decompose_responses = self.llm_client.generate_batch(decompose_tasks)

        all_nuggets: List[List[str]] = []
        for i, (resp, ref) in enumerate(zip(decompose_responses, references)):
            nuggets = self._parse_nuggets(resp, ref)
            all_nuggets.append(nuggets)

        # Phase 2: batch match all nuggets across all instances
        match_tasks = []
        match_map: List[tuple] = []  # (instance_idx, nugget_idx)
        for i, (nuggets, response) in enumerate(zip(all_nuggets, responses)):
            for j, nugget in enumerate(nuggets):
                match_tasks.append({
                    "system": MATCH_SYSTEM_PROMPT,
                    "user": MATCH_USER_TEMPLATE.format(nugget=nugget, response=response),
                    "tags": {"phase": "eval_nugget_match"},
                })
                match_map.append((i, j))

        if match_tasks:
            log.info("Batch-matching %d nuggets...", len(match_tasks))
            match_responses = self.llm_client.generate_batch(match_tasks)
        else:
            match_responses = []

        # Reconstruct per-instance matched lists
        matched_per_instance: List[List[bool]] = [[] for _ in range(len(responses))]
        for (inst_idx, nug_idx), resp in zip(match_map, match_responses):
            matched_per_instance[inst_idx].append(resp.strip().lower().startswith("yes"))

        # Compute results
        results = []
        for nuggets, matched in zip(all_nuggets, matched_per_instance):
            if not nuggets:
                results.append(NuggetResult(recall=1.0, precision=1.0, f1=1.0))
            else:
                results.append(self._compute_result(nuggets, matched))

        return results

    def _decompose(self, reference: str) -> List[str]:
        """Decompose a reference answer into atomic nuggets."""
        prompt = DECOMPOSE_USER_TEMPLATE.format(reference=reference)
        response = self.llm_client.generate(DECOMPOSE_SYSTEM_PROMPT, prompt, tags={"phase": "eval_nugget_decompose"})
        return self._parse_nuggets(response, reference)

    def _parse_nuggets(self, response: str, reference: str) -> List[str]:
        """Parse LLM response into a list of nugget strings."""
        try:
            nuggets = parse_llm_json(response)
            if isinstance(nuggets, list):
                return [str(n) for n in nuggets]
        except (ValueError, TypeError):
            log.warning("Failed to decompose reference into nuggets")
            return [s.strip() for s in reference.split(".") if s.strip()]
        return []

    def _match_nuggets(self, nuggets: List[str], response: str) -> List[bool]:
        """Check which nuggets are present in the response."""
        tasks = [
            {
                "system": MATCH_SYSTEM_PROMPT,
                "user": MATCH_USER_TEMPLATE.format(nugget=nugget, response=response),
                "tags": {"phase": "eval_nugget_match"},
            }
            for nugget in nuggets
        ]

        responses = self.llm_client.generate_batch(tasks)
        return [r.strip().lower().startswith("yes") for r in responses]

    @staticmethod
    def _compute_result(nuggets: List[str], matched: List[bool]) -> NuggetResult:
        n_matched = sum(matched)
        recall = n_matched / len(nuggets) if nuggets else 0.0
        precision = n_matched / max(n_matched, 1)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        return NuggetResult(
            nuggets=nuggets,
            matched=matched,
            recall=recall,
            precision=precision,
            f1=f1,
        )
