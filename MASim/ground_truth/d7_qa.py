"""D7: Standard QA + temporal reasoning — generate factual questions from sessions.

LLM-generated questions are preferred.  The template fallback uses natural
phrasing with participant names and topical hints — never internal session IDs.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

import numpy as np

from MASim.core.schema import DialogueCorpus, Dimension, EvalInstance, is_assistant
from MASim.ground_truth.json_parser import parse_json_array
from MASim.prompts import QA_GENERATION_SYSTEM as QA_SYSTEM_PROMPT
from MASim.prompts import QA_GENERATION_USER as QA_USER_TEMPLATE
from MASim.utils.logging import get_logger

log = get_logger(__name__)

# ── Template fallback queries ────────────────────────────────────────────────
# Used when no LLM is available.  These reference participants and topic
# hints — never session IDs.

_RECALL_TEMPLATES = [
    "What did {speaker} say about {topic} when talking to {listener}?",
    "Do you remember what {speaker} mentioned about {topic}?",
    "What was {speaker}'s point about {topic} in their conversation with {listener}?",
    "Can you recall what {speaker} said regarding {topic}?",
    "What details did {speaker} share about {topic}?",
]

_TEMPORAL_TEMPLATES = [
    "Did {speaker} discuss {topic} before or after talking to {other_participant}?",
    "What was the most recent thing {speaker} mentioned about {topic}?",
    "In what order did {speaker} bring up {topic} and {other_topic}?",
]


def generate_instances(
    corpus: DialogueCorpus,
    max_instances: int = 200,
    llm_client: Any = None,
    seed: int = 42,
) -> List[EvalInstance]:
    """Generate D7 standard QA eval instances."""
    rng = np.random.default_rng(seed)
    instances = []

    eligible = [s for s in corpus.sessions if len(s.turns) >= 4]
    if not eligible:
        return instances

    n_sessions = min(max_instances // 3 + 1, len(eligible))
    selected = list(rng.choice(eligible, size=n_sessions, replace=False))

    if llm_client is not None:
        instances = _generate_via_llm(selected, llm_client, max_instances)
    else:
        instances = _generate_template_based(selected, rng, max_instances)

    return instances[:max_instances]


def _generate_via_llm(sessions, llm_client, max_instances: int) -> List[EvalInstance]:
    """Generate QA pairs using LLM."""
    tasks = []
    session_map = {}

    for session in sessions:
        names = {p: _first_name(p) for p in session.participants}
        transcript = "\n".join(
            f"{names.get(t.speaker_id, t.speaker_id)}: {t.text}"
            for t in session.turns[:15]
        )
        participants = " and ".join(names.values())
        prompt = QA_USER_TEMPLATE.format(
            participants=participants,
            transcript=transcript[:2000],
        )
        tasks.append({"system": QA_SYSTEM_PROMPT, "user": prompt, "tags": {"phase": "d7_qa_gen"}})
        session_map[len(tasks) - 1] = session

    log.info("Generating QA pairs for %d sessions via LLM...", len(tasks))
    responses = llm_client.generate_batch(tasks)

    instances = []
    for i, resp in enumerate(responses):
        session = session_map[i]
        try:
            qa_pairs = parse_json_array(resp)
        except (ValueError, TypeError):
            continue

        for qa in qa_pairs:
            if not isinstance(qa, dict):
                continue
            question = qa.get("question", "")
            if not question:
                continue
            instance = EvalInstance(
                instance_id=f"d7_{uuid.uuid4().hex[:12]}",
                dimension=Dimension.D7_QA,
                query=question,
                ground_truth={
                    "requires_temporal": qa.get("requires_temporal", False),
                    "source_session": session.session_id,
                    "answer": qa.get("answer", ""),
                },
                difficulty="hard" if qa.get("requires_temporal") else "medium",
                answerer_agent_id=session.participants[0] if session.participants else "",
                asker_agent_id=session.participants[1] if len(session.participants) > 1 else "",
                metadata={
                    "evidence_sessions": [session.session_id],
                },
            )
            instances.append(instance)

    return instances


def _generate_template_based(
    sessions, rng: np.random.Generator, max_instances: int,
) -> List[EvalInstance]:
    """Fallback: template-based QA generation without LLM.

    Uses natural phrasing with participant names and topic hints —
    never exposes session IDs.
    """
    instances = []

    for session in sessions:
        if len(instances) >= max_instances:
            break

        participants = [_first_name(p) for p in session.participants]
        session_entities = set()
        for turn in session.turns:
            session_entities.update(turn.entities_mentioned)

        for turn in session.turns:
            if len(turn.text.split()) < 8:
                continue

            speaker_name = _first_name(turn.speaker_id)
            listener_name = _first_name(
                [p for p in session.participants if p != turn.speaker_id][0]
                if len(session.participants) > 1 else turn.speaker_id
            )

            # Build a topic hint from entities or content words
            topic = _extract_topic(turn.text, turn.entities_mentioned)

            template = _RECALL_TEMPLATES[rng.integers(len(_RECALL_TEMPLATES))]
            question = template.format(
                speaker=speaker_name,
                listener=listener_name,
                topic=topic,
            )

            instance = EvalInstance(
                instance_id=f"d7_{uuid.uuid4().hex[:12]}",
                dimension=Dimension.D7_QA,
                query=question,
                ground_truth={
                    "requires_temporal": False,
                    "source_session": session.session_id,
                    "source_turn": turn.turn_id,
                    "answer": turn.text,
                },
                difficulty="easy",
                answerer_agent_id=session.participants[0] if session.participants else "",
                asker_agent_id=session.participants[1] if len(session.participants) > 1 else "",
                metadata={
                    "evidence_sessions": [session.session_id],
                },
            )
            instances.append(instance)

            if len(instances) >= max_instances:
                break

    return instances


def _first_name(agent_slug: str) -> str:
    """Convert 'eleanor_vance' → 'Eleanor'."""
    if is_assistant(agent_slug):
        return "the assistant"
    return agent_slug.split("_")[0].title()


def _extract_topic(text: str, entities: List[str]) -> str:
    """Build a short topical hint from entities or content words."""
    if entities:
        usable = [e for e in entities if len(e) > 2][:2]
        if usable:
            return " and ".join(usable)

    _stop = {"the", "a", "an", "is", "was", "are", "were", "that", "this",
             "with", "from", "have", "has", "been", "they", "them", "their",
             "about", "just", "also", "would", "could", "should", "really",
             "very", "much", "some", "into", "will", "your"}
    words = [w.strip(".,!?;:'\"") for w in text.split()
             if len(w) > 4 and w.lower().strip(".,!?;:'\"") not in _stop][:3]
    return " ".join(words) if words else "what was discussed"
