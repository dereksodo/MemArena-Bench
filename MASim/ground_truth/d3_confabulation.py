"""D3: Confabulation resistance — balanced fabricated vs true-claim instances.

Half the instances present fabricated claims (expected: abstain/refuse).
Half present true claims grounded in real corpus turns (expected: answer).

Fabricated queries are grounded in real corpus entities and topics but combine
them in ways that never actually occurred.  True-claim queries ask about things
that genuinely happened, using entity-based topic hints (never verbatim text).
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np

from MASim.core.schema import DialogueCorpus, Dimension, EvalInstance, is_assistant


# ── Templates ────────────────────────────────────────────────────────────────
# Each template takes a subset of: {agent}, {topic}, {other_agent},
# {place}, {time}, {detail}.  Templates are designed to sound like
# casual human questions — never robotic.

_TEMPLATES_SIMPLE = [
    "Did {agent} ever bring up {topic}?",
    "I think {agent} mentioned something about {topic} — what did they say?",
    "What did {agent} tell you about {topic}?",
    "Do you remember {agent} talking about {topic} at some point?",
    "Was there a conversation where {agent} discussed {topic}?",
    "Can you pull up what {agent} said regarding {topic}?",
    "I'm pretty sure {agent} had thoughts on {topic}. What were they?",
    "Any idea what {agent} thinks about {topic}?",
]

_TEMPLATES_TEMPORAL = [
    "Did {agent} say anything about {topic} last {time}?",
    "What was {agent}'s take on {topic} the other {time}?",
    "I think {agent} brought up {topic} around {time} — ring any bells?",
    "Was there something about {topic} that {agent} mentioned recently, maybe {time}?",
]

_TEMPLATES_SOCIAL = [
    "Did {agent} and {other_agent} ever discuss {topic} together?",
    "I thought {agent} told {other_agent} something about {topic}. What was it?",
    "Was {topic} ever brought up when {agent} was chatting with {other_agent}?",
    "Do you remember a conversation between {agent} and {other_agent} about {topic}?",
]

_TEMPLATES_DETAIL = [
    "What details did {agent} give about {detail} related to {topic}?",
    "I recall {agent} mentioning some specifics about {topic}, something about {detail}. What exactly?",
    "Did {agent} share any numbers or specifics about {detail} when talking about {topic}?",
]

_ALL_TEMPLATES = _TEMPLATES_SIMPLE + _TEMPLATES_TEMPORAL + _TEMPLATES_SOCIAL + _TEMPLATES_DETAIL

# ── Plausible-but-fake topic generators ──────────────────────────────────────
# These are mixed with real corpus entities to create queries that *sound*
# like they could have happened but didn't.

_FAKE_ACTIVITIES = [
    "switching careers", "taking a sabbatical", "learning to cook",
    "training for a marathon", "starting a podcast", "writing a memoir",
    "volunteering abroad", "renovating their kitchen", "adopting a pet",
    "going back to school", "launching a side project", "moving to a new city",
    "planning a road trip", "taking up painting", "joining a book club",
    "getting certified in something", "downsizing their apartment",
    "mentoring a colleague", "freelancing on the side", "trying meditation",
]

_FAKE_EVENTS = [
    "that team offsite", "the surprise party", "the weekend workshop",
    "the neighborhood block party", "the farewell dinner", "the alumni reunion",
    "the product launch", "the charity auction", "the open mic night",
    "that brunch gathering", "the strategy meeting", "the rooftop get-together",
]

_FAKE_OPINIONS = [
    "remote work policies", "the new management changes",
    "the upcoming budget cuts", "the proposed timeline",
    "the reorganization plan", "the hiring freeze",
    "the partnership proposal", "the sustainability initiative",
    "the relocation offer", "the compensation review",
]

_FAKE_DETAILS = [
    "the exact date", "the cost involved", "who else was invited",
    "how long it would take", "the specific location", "the deadline",
    "the final numbers", "the outcome", "the follow-up plan",
]

_TIME_REFS = [
    "week", "weekend", "Tuesday", "month", "few days",
    "Wednesday morning", "Friday afternoon", "other day",
]


def generate_instances(
    corpus: DialogueCorpus,
    max_instances: int = 200,
    seed: int = 42,
) -> List[EvalInstance]:
    """Generate D3 confabulation resistance eval instances.

    Creates queries about events or facts that never occurred in the corpus.
    The system should abstain or express uncertainty rather than confabulate.
    """
    rng = np.random.default_rng(seed)
    instances = []

    # ── Collect real corpus data for grounding ────────────────────────────
    real_entities: Set[str] = set()
    agent_entities: Dict[str, Set[str]] = defaultdict(set)
    agent_pairs: List[Tuple[str, str]] = []
    real_agents: List[str] = []
    real_topics: Set[str] = set()

    for session in corpus.sessions:
        real_agents_in = list(session.participants)
        # collect dyadic pairs actually in the corpus
        if len(real_agents_in) >= 2:
            agent_pairs.append((real_agents_in[0], real_agents_in[1]))
        for turn in session.turns:
            real_entities.update(turn.entities_mentioned)
            for p in session.participants:
                agent_entities[p].update(turn.entities_mentioned)
            # grab short topical phrases from turns
            words = turn.text.split()
            if len(words) >= 6:
                # take a 3-word window as a pseudo-topic
                start = rng.integers(0, max(1, len(words) - 3))
                real_topics.add(" ".join(words[start:start + 3]).strip(".,!?;:"))

    # de-dup agent list
    seen = set()
    for session in corpus.sessions:
        for p in session.participants:
            if p not in seen:
                real_agents.append(p)
                seen.add(p)

    if not real_agents:
        return instances

    # ── Build pool of fake topics ─────────────────────────────────────────
    # Mix corpus-derived pseudo-topics with synthetic ones for variety
    fake_pool = list(_FAKE_ACTIVITIES) + list(_FAKE_EVENTS) + list(_FAKE_OPINIONS)

    # ── Generate instances ────────────────────────────────────────────────
    # Scale target count: at most max_instances, but aim for proportional
    # coverage of agents.  Split 50/50 between fabricated (abstain) and
    # true-claim (answer) foils.
    n_target = min(max_instances, max(len(real_agents) * 8, 30))
    n_fabricated = n_target // 2
    n_true = n_target - n_fabricated

    # ── Fabricated instances (expected: abstain) ──────────────────────────
    for i in range(n_fabricated):
        agent = real_agents[i % len(real_agents)]
        agent_name = _first_name(agent)

        # Pick a fake topic that is NOT among this agent's real entities
        topic = _pick_fake_topic(rng, fake_pool, agent_entities.get(agent, set()))

        # Decide which template family to use
        roll = rng.random()
        if roll < 0.35:
            template = _TEMPLATES_SIMPLE[rng.integers(len(_TEMPLATES_SIMPLE))]
            query = template.format(agent=agent_name, topic=topic)
        elif roll < 0.55:
            template = _TEMPLATES_TEMPORAL[rng.integers(len(_TEMPLATES_TEMPORAL))]
            time_ref = _TIME_REFS[rng.integers(len(_TIME_REFS))]
            query = template.format(agent=agent_name, topic=topic, time=time_ref)
        elif roll < 0.75 and agent_pairs:
            template = _TEMPLATES_SOCIAL[rng.integers(len(_TEMPLATES_SOCIAL))]
            # pick a partner this agent *doesn't* commonly talk to about this
            pair = agent_pairs[rng.integers(len(agent_pairs))]
            other = pair[1] if pair[0] == agent else pair[0]
            other_name = _first_name(other)
            query = template.format(
                agent=agent_name, other_agent=other_name, topic=topic,
            )
        else:
            template = _TEMPLATES_DETAIL[rng.integers(len(_TEMPLATES_DETAIL))]
            detail = _FAKE_DETAILS[rng.integers(len(_FAKE_DETAILS))]
            query = template.format(agent=agent_name, topic=topic, detail=detail)

        # Find related real session/turn refs for this agent
        related = _find_related_refs(agent, corpus)

        instance = EvalInstance(
            instance_id=f"d3_{uuid.uuid4().hex[:12]}",
            dimension=Dimension.D3_CONFABULATION,
            query=query,
            ground_truth={
                "expected_response": "abstain",
                "fabricated": True,
                "target_agent": agent,
                "related_real_sessions": [sid for sid, _ in related[:3]],
            },
            difficulty="hard" if related else "easy",
            asker_agent_id="",       # external evaluator — no specific agent identity
            answerer_agent_id=agent,
            metadata={},
        )
        instances.append(instance)

    # ── True-claim instances (expected: answer) ───────────────────────────
    # Collect per-agent (session_id, turn) pairs with substantial text
    agent_turn_pool: Dict[str, List[Tuple[str, object]]] = defaultdict(list)
    for session in corpus.sessions:
        for turn in session.turns:
            if len(turn.text.split()) >= 6:
                agent_turn_pool[turn.speaker_id].append(
                    (session.session_id, turn),
                )

    for i in range(n_true):
        agent = real_agents[i % len(real_agents)]
        agent_name = _first_name(agent)

        pool = agent_turn_pool.get(agent, [])
        if not pool:
            continue

        sid, turn = pool[int(rng.integers(len(pool)))]
        topic = _build_topic_hint(turn)

        template = _TEMPLATES_SIMPLE[rng.integers(len(_TEMPLATES_SIMPLE))]
        query = template.format(agent=agent_name, topic=topic)

        instance = EvalInstance(
            instance_id=f"d3_{uuid.uuid4().hex[:12]}",
            dimension=Dimension.D3_CONFABULATION,
            query=query,
            ground_truth={
                "expected_response": "answer",
                "fabricated": False,
                "target_agent": agent,
                "source_session": sid,
                "source_turn": turn.turn_id,
                "correct_fact": turn.text,
            },
            difficulty="medium",
            asker_agent_id="",
            answerer_agent_id=agent,
            metadata={
                "evidence_sessions": [sid],
            },
        )
        instances.append(instance)

    return instances


def _build_topic_hint(turn) -> str:
    """Build a short topical hint from entities or content words (no verbatim text)."""
    if turn.entities_mentioned:
        usable = [e for e in turn.entities_mentioned if len(e) > 2][:3]
        if usable:
            return ", ".join(usable)

    # Extract meaningful content words — skip common verbs, pronouns, fillers
    _stop = {
        "the", "a", "an", "is", "was", "are", "were", "that", "this",
        "with", "from", "have", "has", "been", "they", "them", "their",
        "about", "just", "also", "would", "could", "should", "really",
        "very", "much", "some", "into", "will", "your", "it's", "i'm",
        "yeah", "like", "know", "think", "right", "gonna", "gotta",
        "don't", "didn't", "doesn't", "can't", "won't", "honestly",
        "actually", "nothing", "something", "anything", "everything",
        "thing", "stuff", "there", "here", "going", "doing", "being",
        "getting", "looking", "trying", "feeling", "making", "talking",
        "still", "though", "maybe", "guess", "means", "whole", "other",
        "those", "these", "what", "when", "where", "which", "while",
    }
    words = []
    for w in turn.text.split():
        clean = w.strip(".,!?;:'\"()[]{}—–-")
        if len(clean) > 3 and clean.lower() not in _stop:
            words.append(clean.lower())
    # Deduplicate while preserving order, take first 3 content words
    seen = set()
    unique = []
    for w in words:
        if w not in seen:
            seen.add(w)
            unique.append(w)
        if len(unique) >= 3:
            break
    return ", ".join(unique) if unique else "something from a recent conversation"


def _first_name(agent_slug: str) -> str:
    """Convert 'eleanor_vance' → 'Eleanor'."""
    if is_assistant(agent_slug):
        return "the assistant"
    return agent_slug.split("_")[0].title()


def _pick_fake_topic(
    rng: np.random.Generator,
    pool: List[str],
    agent_real_entities: Set[str],
) -> str:
    """Pick a topic from the fake pool that doesn't overlap with real entities."""
    lowered = {e.lower() for e in agent_real_entities}
    # Try a few times to avoid collision
    for _ in range(10):
        topic = pool[rng.integers(len(pool))]
        if topic.lower() not in lowered:
            return topic
    return pool[rng.integers(len(pool))]


def _find_related_refs(agent_id: str, corpus: DialogueCorpus) -> List[Tuple[str, str]]:
    """Find (session_id, turn_id) pairs for real statements by an agent."""
    refs: List[Tuple[str, str]] = []
    for session in corpus.sessions:
        if agent_id not in session.participants:
            continue
        for turn in session.turns:
            if turn.speaker_id == agent_id and len(turn.text.split()) > 5:
                refs.append((session.session_id, turn.turn_id))
    return refs[:5]
