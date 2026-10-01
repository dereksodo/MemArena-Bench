"""D2: Cross-session anaphora — generate evaluation instances for reference resolution."""

from __future__ import annotations

import uuid
from typing import Dict, List

import numpy as np

from MASim.core.schema import AnaphoraGT, DialogueCorpus, Dimension, EvalInstance, Session


def _slug_to_name(slug: str) -> str:
    """Convert 'maya_chen' to 'Maya Chen'."""
    return " ".join(part.capitalize() for part in slug.split("_"))


def _extract_entity_sentences(text: str, entity: str, speaker: str) -> str:
    """Extract only the sentence(s) from *text* that mention *entity*.

    Falls back to the full text if no sentence-level split is possible.
    Prepends speaker attribution for judge clarity.
    """
    import re
    # Split on sentence-ending punctuation, keeping emoji-heavy chat in mind
    sentences = re.split(r'(?<=[.!?])\s+', text)
    entity_lower = entity.lower()
    relevant = [s.strip() for s in sentences if entity_lower in s.lower()]
    if relevant:
        excerpt = " ".join(relevant)
    else:
        # Fallback: full text (shouldn't happen given our filters)
        excerpt = text
    return f"{speaker} mentioned {entity}: {excerpt}"


_QUERY_TEMPLATES = [
    (
        "Someone brought up {entity} again recently — "
        "what was the original context when it first came up?"
    ),
    (
        "I remember {entity} being mentioned in a couple of different "
        "conversations. What was said about it the first time?"
    ),
    (
        "When {entity} came up the other day, it felt familiar — "
        "can you remind me what was discussed about {entity} earlier?"
    ),
    (
        "There was a reference to {entity} in a later chat. "
        "What were the details from the earlier conversation?"
    ),
    (
        "I keep hearing about {entity}. What's the backstory — "
        "when and how did it first come up?"
    ),
    (
        "{entity} was mentioned across different conversations. "
        "Can you piece together what was originally said versus "
        "what came up later?"
    ),
    (
        "Help me connect the dots on {entity} — it was discussed "
        "at least twice in separate conversations. What happened each time?"
    ),
    (
        "Going back to {entity}, I think it was first discussed a while ago "
        "and then referenced again. What do you recall?"
    ),
]


def generate_instances(
    corpus: DialogueCorpus,
    anaphora_gts: List[AnaphoraGT],
    max_instances: int = 200,
    seed: int = 42,
) -> List[EvalInstance]:
    """Generate D2 cross-session anaphora eval instances.

    For each tracked cross-reference, create a query asking the system
    to resolve what a reference refers to from a previous conversation.
    """
    rng = np.random.default_rng(seed)
    instances = []

    # Build lookups
    turn_lookup = {}
    session_lookup: Dict[str, Session] = {}
    session_order: Dict[str, int] = {}
    for i, session in enumerate(corpus.sessions):
        session_lookup[session.session_id] = session
        session_order[session.session_id] = i
        for turn in session.turns:
            turn_lookup[turn.turn_id] = turn

    for gt in anaphora_gts:
        if len(instances) >= max_instances:
            break

        ref_turn = turn_lookup.get(gt.reference_turn)
        ant_turn = turn_lookup.get(gt.antecedent_turn)
        if not ref_turn or not ant_turn:
            continue

        # Skip assistant turns — gold should be from a human speaker
        if ant_turn.speaker_id.startswith("assistant"):
            continue

        # Skip if the entity name doesn't actually appear in the antecedent text
        if gt.referent.lower() not in ant_turn.text.lower():
            continue

        template = _QUERY_TEMPLATES[rng.integers(len(_QUERY_TEMPLATES))]
        query = template.format(entity=gt.referent)

        difficulty = _assess_difficulty(gt, session_order, corpus)

        # Build gold answer: include speaker + the sentence(s) that mention the entity.
        # Using the full turn text makes token_f1 scoring unreliable when the turn
        # is long, so extract only sentences containing the entity name.
        speaker_name = _slug_to_name(ant_turn.speaker_id)
        gold_text = _extract_entity_sentences(ant_turn.text, gt.referent, speaker_name)

        instance = EvalInstance(
            instance_id=f"d2_{uuid.uuid4().hex[:12]}",
            dimension=Dimension.D2_ANAPHORA,
            query=query,
            ground_truth={
                "referent": gt.referent,
                "antecedent_session": gt.antecedent_session,
                "antecedent_turn": gt.antecedent_turn,
                "reference_session": gt.reference_session,
                "reference_turn": gt.reference_turn,
                "antecedent_text": gold_text,
            },
            difficulty=difficulty,
            metadata={
                "evidence_sessions": [gt.antecedent_session, gt.reference_session],
            },
        )
        instances.append(instance)

    return instances


def _assess_difficulty(
    gt: AnaphoraGT,
    session_order: Dict[str, int],
    corpus: DialogueCorpus,
) -> str:
    """Assess difficulty based on session gap and entity frequency.

    Larger session gap = harder (more hay to search through).
    More frequent entity = harder (more distractors).
    """
    ant_idx = session_order.get(gt.antecedent_session, 0)
    ref_idx = session_order.get(gt.reference_session, 0)
    gap = ref_idx - ant_idx

    # Count how many sessions mention this entity (more = harder)
    mention_count = 0
    for session in corpus.sessions:
        for turn in session.turns:
            if gt.referent in turn.entities_mentioned:
                mention_count += 1
                break

    if gap >= 5 or mention_count >= 4:
        return "hard"
    elif gap >= 2 or mention_count >= 2:
        return "medium"
    return "easy"
