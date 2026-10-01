"""Ego-centric projection: π(u_i, D) → filtered corpus.

The core operation that distinguishes MemArena from omniscient benchmarks.
Given a user agent and the full dialogue corpus, returns only what that
user has directly experienced or been told about.
"""

from __future__ import annotations

from typing import Any, Dict, List

from MASim.core.schema import DialogueCorpus, DialogueTurn, EvalInstance, Session
from MASim.utils.logging import get_logger

log = get_logger(__name__)


def ego_project(user_id: str, corpus: DialogueCorpus) -> DialogueCorpus:
    """Project the full corpus to an ego-centric view for a given user.

    Returns a new DialogueCorpus containing only:
    - Sessions where user_id is a participant
    - Events that user_id was visible to
    - The user's own agent state
    """
    # Filter sessions
    ego_sessions = [
        s for s in corpus.sessions if user_id in s.participants
    ]

    # Filter events
    ego_events = [
        e for e in corpus.events if user_id in e.visibility_mask
    ]

    # Build ego agent dict (include user + their conversation partners)
    partner_ids = set()
    for s in ego_sessions:
        partner_ids.update(s.participants)
    ego_agents = {
        aid: state for aid, state in corpus.agents.items()
        if aid in partner_ids
    }

    # Filter graph edges
    ego_edges = [
        e for e in corpus.social_graph_edges
        if e.get("source") == user_id or e.get("target") == user_id
    ]

    projected = DialogueCorpus(
        sessions=ego_sessions,
        agents=ego_agents,
        events=ego_events,
        social_graph_edges=ego_edges,
    )

    log.debug(
        "Ego projection for %s: %d/%d sessions, %d/%d events",
        user_id,
        len(ego_sessions), len(corpus.sessions),
        len(ego_events), len(corpus.events),
    )
    return projected


def build_ego_memory(user_id: str, corpus: DialogueCorpus) -> List[DialogueTurn]:
    """Build a flat, chronologically ordered list of all turns visible to a user.

    This represents the user's complete memory — everything they've seen or said.
    """
    turns: List[DialogueTurn] = []
    for session in corpus.sessions:
        if user_id in session.participants:
            turns.extend(session.turns)

    # Sort by timestamp for chronological ordering
    turns.sort(key=lambda t: t.timestamp)
    return turns


def build_ego_context(
    user_id: str,
    corpus: DialogueCorpus,
    max_turns: int = 200,
) -> List[Dict[str, Any]]:
    """Build ego context suitable for eval instance input.

    Returns a list of turn dicts in chronological order, limited to max_turns.
    """
    memory = build_ego_memory(user_id, corpus)
    if len(memory) > max_turns:
        memory = memory[-max_turns:]  # keep most recent
    return [t.to_dict() for t in memory]


def ego_filter_eval_instances(
    user_id: str,
    instances: List[EvalInstance],
    corpus: DialogueCorpus,
) -> List[EvalInstance]:
    """Filter eval instances to only those answerable from the user's ego view.

    For each instance, verify that the required evidence sessions
    are within the user's projected corpus.
    """
    ego_session_ids = {
        s.session_id for s in corpus.sessions if user_id in s.participants
    }

    filtered = []
    for inst in instances:
        evidence = inst.metadata.get("evidence_sessions", [])
        if not evidence:
            # No evidence requirement — include
            filtered.append(inst)
        elif all(sid in ego_session_ids for sid in evidence):
            filtered.append(inst)

    log.debug(
        "Ego filter for %s: %d/%d instances retained",
        user_id, len(filtered), len(instances),
    )
    return filtered
