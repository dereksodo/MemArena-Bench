"""ProbeSelector: select probe-worthy facts from agent memory for memory-probe PA sessions.

Scores facts for probe-worthiness, assigns probe types (conflict_probe,
temporal_probe, fact_recall), and samples ProbeSpec objects via softmax.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np

from MASim.core.knowledge_state import KnowledgeState, KnownFact, KnownFeeling
from MASim.core.schema import is_assistant as _is_assistant
from MASim.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class ProbeSpec:
    """Specification for a single memory-probe PA session."""
    probe_type: str                          # "conflict_probe" | "temporal_probe" | "fact_recall"
    dimension_hint: str                      # "D1" | "D7" | "D8"
    source_facts: List[KnownFact]            # the facts being probed
    source_feelings: List[KnownFeeling]      # optional emotional context
    hint_entities: List[str]                  # entities to weave into the question
    question_seed: str                       # natural-language opener for assistant
    score: float = 0.0                       # probe-worthiness score


class ProbeSelector:
    """Build and sample ProbeSpec objects from an agent's KnowledgeState."""

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)

    def select_probes(
        self,
        agent_id: str,
        knowledge: KnowledgeState,
        n: int = 1,
    ) -> List[ProbeSpec]:
        """Score facts for probe-worthiness and softmax-sample n ProbeSpecs.

        Scoring weights:
          - Cross-person facts (heard from someone else)       +2.0
          - Emotionally charged (co-occurring feeling exists)  +1.5
          - Facts with multiple entity mentions                +1.0
          - Older facts (more likely to be misremembered)      +0.5
          - Facts appearing in multiple sessions               +2.0
        """
        if not knowledge.facts:
            return []

        # Only probe conversation-derived facts from real people — skip:
        # - social-knowledge seed facts (session_id="", source_agent="world")
        # - facts sourced from __assistant__ (PA assistant turns, not real people)
        fact_list = [
            f for f in knowledge.facts.values()
            if f.session_id and f.source_agent not in ("world", "self") and not _is_assistant(f.source_agent)
        ]
        if not fact_list:
            return []
        scores = np.zeros(len(fact_list))

        # Build helper lookups
        feeling_sessions: Set[str] = {f.session_id for f in knowledge.feelings.values()}
        session_fact_count: Dict[str, int] = {}
        for f in fact_list:
            session_fact_count[f.session_id] = session_fact_count.get(f.session_id, 0) + 1

        # Find facts that appear to cover similar topics (content overlap)
        # Simple heuristic: group by session, facts from sessions with many facts
        # are likely richer targets
        content_by_session: Dict[str, List[str]] = {}
        for f in fact_list:
            content_by_session.setdefault(f.session_id, []).append(f.content.lower())

        # Timestamps for age scoring
        if fact_list:
            max_ts = max(f.timestamp for f in fact_list)
            min_ts = min(f.timestamp for f in fact_list)
            ts_range = max_ts - min_ts if max_ts > min_ts else 1.0

        for i, fact in enumerate(fact_list):
            # Cross-person: heard from someone else
            if fact.source_agent != agent_id and fact.source_agent != "self":
                scores[i] += 2.0

            # Emotionally charged: a feeling exists for the same session
            if fact.session_id in feeling_sessions:
                scores[i] += 1.5

            # Entity richness: facts with many entity index entries
            entity_count = sum(
                1 for eset in knowledge.entity_index.values()
                if fact.fact_id in eset
            )
            if entity_count >= 2:
                scores[i] += 1.0

            # Older facts are better probe targets
            age_frac = (max_ts - fact.timestamp) / ts_range if ts_range > 0 else 0
            scores[i] += 0.5 * age_frac

            # Facts from sessions with many facts (richer conversations)
            if session_fact_count.get(fact.session_id, 0) >= 3:
                scores[i] += 1.0

        # Filter out very low-scoring facts (base threshold)
        viable_mask = scores > 0.5
        if not viable_mask.any():
            # Fall back to all facts with a minimum score bump
            scores += 1.0
            viable_mask = np.ones(len(fact_list), dtype=bool)

        viable_indices = np.where(viable_mask)[0]
        viable_scores = scores[viable_mask]

        n = min(n, len(viable_indices))
        if n <= 0:
            return []

        # Softmax sample
        exp_s = np.exp(viable_scores - viable_scores.max())
        probs = exp_s / exp_s.sum()
        chosen = self._rng.choice(len(viable_indices), size=n, replace=False, p=probs)

        probes: List[ProbeSpec] = []
        for ci in chosen:
            fact_idx = viable_indices[ci]
            fact = fact_list[fact_idx]
            probe = self._build_probe_spec(
                agent_id, fact, knowledge, scores[fact_idx],
            )
            probes.append(probe)

        return probes

    def _build_probe_spec(
        self,
        agent_id: str,
        fact: KnownFact,
        knowledge: KnowledgeState,
        score: float,
    ) -> ProbeSpec:
        """Build a ProbeSpec from a selected fact, choosing type based on characteristics."""
        # Gather related facts from same session or same entities
        related_facts = [fact]
        fact_entities: List[str] = []
        for entity, fids in knowledge.entity_index.items():
            if fact.fact_id in fids:
                fact_entities.append(entity)
                for fid in fids:
                    if fid != fact.fact_id and fid in knowledge.facts:
                        other = knowledge.facts[fid]
                        if other not in related_facts:
                            related_facts.append(other)
                            if len(related_facts) >= 4:
                                break

        # Gather related feelings
        related_feelings: List[KnownFeeling] = []
        for f in knowledge.feelings.values():
            if f.session_id == fact.session_id:
                related_feelings.append(f)
                if len(related_feelings) >= 2:
                    break

        # Determine probe type
        probe_type, dim_hint, question_seed = self._classify_probe(
            agent_id, fact, related_facts, fact_entities, knowledge,
        )

        return ProbeSpec(
            probe_type=probe_type,
            dimension_hint=dim_hint,
            source_facts=related_facts,
            source_feelings=related_feelings,
            hint_entities=fact_entities[:5],
            question_seed=question_seed,
            score=score,
        )

    @staticmethod
    def _classify_probe(
        agent_id: str,
        primary_fact: KnownFact,
        related_facts: List[KnownFact],
        entities: List[str],
        knowledge: KnowledgeState,
    ) -> tuple:
        """Determine probe type and build a question seed.

        Returns (probe_type, dimension_hint, question_seed).
        """
        from MASim.core.knowledge_state import _slug_to_display_name

        source_name = _slug_to_display_name(primary_fact.source_agent)

        # Check for conflicting versions: different source agents, same entities
        different_sources = [
            f for f in related_facts
            if f.source_agent != primary_fact.source_agent
            and f.fact_id != primary_fact.fact_id
        ]
        if different_sources:
            other = different_sources[0]
            other_name = _slug_to_display_name(other.source_agent)
            entity_label = _slug_to_display_name(entities[0]) if entities else "that topic"
            return (
                "conflict_probe",
                "D1",
                f"I noticed something — you heard about {entity_label} from "
                f"{source_name}, but {other_name} seemed to have a different take. "
                f"What do you think actually happened?",
            )

        # Check for temporal markers: facts from different timestamps
        timestamps = sorted(set(f.timestamp for f in related_facts))
        if len(timestamps) >= 2 and len(entities) >= 1:
            entity_label = _slug_to_display_name(entities[0])
            return (
                "temporal_probe",
                "D8",
                f"I'm trying to piece together the timeline — when "
                f"{source_name} told you about {entity_label}, "
                f"was that early on or more recently?",
            )

        # Default: fact recall
        entity_label = _slug_to_display_name(entities[0]) if entities else "something"
        return (
            "fact_recall",
            "D7",
            f"I was curious — you were chatting with {source_name} about "
            f"{entity_label}. What was that about exactly?",
        )
