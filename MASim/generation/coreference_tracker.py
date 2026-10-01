"""Track and annotate cross-session entity references.

Identifies when entities mentioned in one session are referenced again
in later sessions, enabling D2 (cross-session anaphora) evaluation.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple

from MASim.core.schema import AnaphoraGT, DialogueCorpus, DialogueTurn, Session
from MASim.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class EntityMention:
    """A single entity mention in the corpus."""
    entity: str
    session_id: str
    turn_id: str
    speaker_id: str
    timestamp: float


@dataclass
class CrossReference:
    """A cross-session reference pair."""
    entity: str
    antecedent: EntityMention
    reference: EntityMention


class CoreferenceTracker:
    """Track entity mentions across sessions and annotate cross-references."""

    def __init__(self, min_session_gap: int = 1):
        """
        Args:
            min_session_gap: Minimum number of sessions between antecedent and reference.
        """
        self.min_session_gap = min_session_gap

    def track(self, corpus: DialogueCorpus) -> Tuple[List[CrossReference], List[AnaphoraGT]]:
        """Find all cross-session entity references in the corpus.

        Returns:
            (cross_references, anaphora_ground_truths)
        """
        # Step 1: Build entity mention index
        mentions = self._extract_mentions(corpus)

        # Step 2: Group by entity
        entity_groups: Dict[str, List[EntityMention]] = defaultdict(list)
        for m in mentions:
            entity_groups[m.entity].append(m)

        # Step 3: Find cross-session pairs
        session_order = {
            s.session_id: i for i, s in enumerate(corpus.sessions)
        }

        cross_refs: List[CrossReference] = []
        ground_truths: List[AnaphoraGT] = []

        for entity, entity_mentions in entity_groups.items():
            if len(entity_mentions) < 2:
                continue

            # Sort by timestamp
            sorted_mentions = sorted(entity_mentions, key=lambda m: m.timestamp)

            for i, antecedent in enumerate(sorted_mentions):
                for reference in sorted_mentions[i + 1:]:
                    # Must be in different sessions
                    if antecedent.session_id == reference.session_id:
                        continue

                    # Check session gap
                    ant_idx = session_order.get(antecedent.session_id, 0)
                    ref_idx = session_order.get(reference.session_id, 0)
                    if ref_idx - ant_idx < self.min_session_gap:
                        continue

                    xref = CrossReference(
                        entity=entity,
                        antecedent=antecedent,
                        reference=reference,
                    )
                    cross_refs.append(xref)

                    gt = AnaphoraGT(
                        referent=entity,
                        antecedent_session=antecedent.session_id,
                        antecedent_turn=antecedent.turn_id,
                        reference_session=reference.session_id,
                        reference_turn=reference.turn_id,
                    )
                    ground_truths.append(gt)

                    # Only track first cross-reference per antecedent to avoid explosion
                    break

        log.info(
            "Found %d cross-session references across %d entities",
            len(cross_refs), len(entity_groups),
        )
        return cross_refs, ground_truths

    def _extract_mentions(self, corpus: DialogueCorpus) -> List[EntityMention]:
        """Extract all entity mentions from the corpus using turn annotations.

        Filters:
        - Skip assistant/bot turns (speaker_id starting with 'assistant').
        - Only include turns where the entity name actually appears in the text,
          to avoid NER false-positives producing meaningless gold answers.
        """
        mentions = []
        for session in corpus.sessions:
            for turn in session.turns:
                # Skip assistant turns — gold answers should come from humans
                if turn.speaker_id.startswith("assistant"):
                    continue
                for entity in turn.entities_mentioned:
                    # Only include if the entity name is actually in the turn text
                    if entity.lower() not in turn.text.lower():
                        continue
                    mentions.append(EntityMention(
                        entity=entity,
                        session_id=session.session_id,
                        turn_id=turn.turn_id,
                        speaker_id=turn.speaker_id,
                        timestamp=turn.timestamp,
                    ))
        return mentions

    def annotate_corpus(
        self,
        corpus: DialogueCorpus,
        cross_refs: List[CrossReference],
    ) -> DialogueCorpus:
        """Add cross-reference annotations to turns in the corpus."""
        # Build lookup: turn_id -> list of cross-references
        ref_by_turn: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        for xref in cross_refs:
            ref_by_turn[xref.reference.turn_id].append({
                "entity": xref.entity,
                "antecedent_session": xref.antecedent.session_id,
                "antecedent_turn": xref.antecedent.turn_id,
            })

        # Annotate turns
        for session in corpus.sessions:
            for turn in session.turns:
                if turn.turn_id in ref_by_turn:
                    turn.metadata["cross_references"] = ref_by_turn[turn.turn_id]

        return corpus
