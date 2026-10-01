"""D6: Metadata completeness — generate exhaustive retrieval queries.

Design goals:
1. Deduplicate: one triple per (entity, attribute) pair, picking a
   representative source.  This prevents O(n_turns * n_entities) blow-up.
2. Natural queries: varied templates that sound human.
3. Multi-source aggregation: "hard" instances require gathering info
   from multiple sessions about the same entity.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from MASim.core.schema import DialogueCorpus, Dimension, EvalInstance, is_assistant


# ── Per-attribute query templates ────────────────────────────────────────────

_SPEAKER_TEMPLATES = [
    "Who brought up {entity} in conversation?",
    "Do you remember who was talking about {entity}?",
    "Who first mentioned {entity}?",
    "Which person discussed {entity}?",
]

_CONTEXT_TEMPLATES = [
    "What was the context when {entity} came up?",
    "In what setting was {entity} discussed?",
    "What were people saying about {entity}?",
    "Can you give me the details around the {entity} discussion?",
]

_PARTICIPANT_TEMPLATES = [
    "Who was in the room when {entity} was discussed?",
    "Which people were part of the conversation about {entity}?",
    "Who was present when someone mentioned {entity}?",
    "Who else heard the discussion about {entity}?",
]

_SESSION_TEMPLATES = [
    "When was {entity} discussed?",
    "Around what time did {entity} come up in conversation?",
    "How recently was {entity} mentioned?",
    "Can you tell me when the conversation about {entity} happened?",
]

_TEMPLATES_BY_ATTR = {
    "speaker": _SPEAKER_TEMPLATES,
    "context": _CONTEXT_TEMPLATES,
    "participants": _PARTICIPANT_TEMPLATES,
    "session": _SESSION_TEMPLATES,
}

# ── Lifecycle metadata templates ───────────────────────────────────────────

_LOCATION_QUERY_TEMPLATES = {
    "home_location": [
        "Where does {entity} live?",
        "What is {entity}'s home?",
        "Do you know where {entity} lives?",
        "Where is {entity}'s place?",
    ],
    "work_location": [
        "Where does {entity} usually work?",
        "What is {entity}'s workplace?",
        "Do you know where {entity} goes to work?",
        "Where does {entity} work from?",
    ],
    "conversation_location": [
        "Where did the conversation about {entity} happen?",
        "Where were people when {entity} came up?",
        "At what location was {entity} discussed?",
        "Where was the discussion about {entity}?",
    ],
}

_GROUP_QUERY_TEMPLATES = {
    "group_membership": [
        "Which group is {entity} a member of?",
        "What group does {entity} belong to?",
        "Is {entity} part of any group?",
        "What team or group is {entity} in?",
    ],
    "group_role": [
        "What is {entity}'s role in {context}?",
        "What does {entity} do in {context}?",
        "What role does {entity} play in {context}?",
        "How would you describe {entity}'s position in {context}?",
    ],
    "meeting_location": [
        "Where does {entity} usually meet?",
        "Where does {entity} hold their meetings?",
        "What is the meeting spot for {entity}?",
        "Where does {entity} get together?",
    ],
    "members": [
        "Who is in {entity}?",
        "Who are the members of {entity}?",
        "Who belongs to {entity}?",
        "Can you list the people in {entity}?",
    ],
}


def _slug_to_display(slug: str) -> str:
    """Convert 'maya_chen' → 'Maya Chen', '__assistant__' → 'AI Assistant'."""
    if is_assistant(slug):
        return "AI Assistant"
    return " ".join(p.capitalize() for p in slug.split("_"))


def generate_instances(
    corpus: DialogueCorpus,
    max_instances: int = 200,
    seed: int = 42,
) -> List[EvalInstance]:
    """Generate D6 metadata completeness eval instances.

    Creates queries about entity attributes extracted from conversation
    content: who said what, when, about whom, etc.  Also generates
    lifecycle metadata queries (locations, groups) when data is available.

    Deduplicates by (entity, attribute) to prevent instance explosion
    at scale.  Entities mentioned in multiple sessions get "hard"
    difficulty (requires cross-session aggregation).
    """
    rng = np.random.default_rng(seed)
    instances = []

    triples = _extract_deduplicated_triples(corpus)

    # Add lifecycle triples (locations + groups)
    triples.extend(_extract_location_triples(corpus))
    triples.extend(_extract_group_triples(corpus))

    # Shuffle for variety, then cap
    rng.shuffle(triples)

    for entity, attribute, value, evidence, difficulty in triples[:max_instances]:
        # Convert slug-like entities (e.g. "tunde_bakare") to display names
        display_entity = _slug_to_display(entity) if "_" in entity else entity

        # Pick templates: lifecycle attributes have their own template dicts
        if attribute in _LOCATION_QUERY_TEMPLATES:
            templates = _LOCATION_QUERY_TEMPLATES[attribute]
        elif attribute in _GROUP_QUERY_TEMPLATES:
            templates = _GROUP_QUERY_TEMPLATES[attribute]
        else:
            templates = _TEMPLATES_BY_ATTR.get(attribute, [f"What do you know about {{entity}}'s {attribute}?"])

        # Some group templates use {context}
        context = evidence.get("context", "")
        query = templates[rng.integers(len(templates))].format(
            entity=display_entity, context=context,
        )

        instance = EvalInstance(
            instance_id=f"d6_{uuid.uuid4().hex[:12]}",
            dimension=Dimension.D6_METADATA,
            query=query,
            ground_truth={
                "entity": entity,
                "attribute": attribute,
                "value": value,
                "source_session": evidence.get("session_id", ""),
                "source_turn": evidence.get("turn_id", ""),
            },
            difficulty=difficulty,
            metadata={
                "evidence_sessions": evidence.get("all_sessions", [evidence.get("session_id", "")]),
            },
        )
        instances.append(instance)

    return instances


def _extract_deduplicated_triples(
    corpus: DialogueCorpus,
) -> List[Tuple[str, str, Any, Dict[str, Any], str]]:
    """Extract deduplicated (entity, attribute, value, evidence, difficulty) tuples.

    Groups by (entity, attribute) and keeps one representative value,
    but tracks all sessions where the entity appears for difficulty
    assessment and evidence linking.
    """
    # entity -> {attribute -> list of (value, session_id, turn_id)}
    grouped: Dict[str, Dict[str, List[Tuple[Any, str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    # entity -> set of session_ids (for difficulty)
    entity_sessions: Dict[str, Set[str]] = defaultdict(set)

    for session in corpus.sessions:
        # Pre-compute display-name participant list for this session
        display_parts = [_slug_to_display(p) for p in session.participants]
        display_participants = ", ".join(display_parts)
        session_meta = ""
        loc = getattr(session, "location_id", "") or ""
        mod = getattr(session, "modality", "") or ""
        if loc or mod:
            loc_str = loc.replace("loc_", "").replace("_", " ").title() if loc else ""
            mod_str = mod.replace("_", " ") if mod else ""
            parts = [p for p in [loc_str, mod_str] if p]
            session_meta = " | ".join(parts)

        for turn in session.turns:
            for entity in turn.entities_mentioned:
                sid = session.session_id
                tid = turn.turn_id
                entity_sessions[entity].add(sid)

                speaker_display = _slug_to_display(turn.speaker_id)
                grouped[entity]["speaker"].append(
                    (speaker_display, sid, tid)
                )
                # Extract a short excerpt of what was said about the entity
                text = turn.text or ""
                excerpt = text[:200].strip()
                if not excerpt:
                    excerpt = f"(mentioned in turn by {speaker_display})"
                grouped[entity]["context"].append(
                    (excerpt, sid, tid)
                )
                grouped[entity]["participants"].append(
                    (display_participants, sid, tid)
                )

        # Session-level triple (once per session, not per turn)
        session_entities = set()
        for turn in session.turns:
            session_entities.update(turn.entities_mentioned)
        for entity in session_entities:
            first_turn = session.turns[0] if session.turns else None
            if first_turn:
                # Human-readable session label
                from datetime import datetime, timezone, timedelta
                _base = datetime(2025, 1, 1, tzinfo=timezone.utc)
                session_dt = _base + timedelta(days=float(session.start_time))
                session_label = session_dt.strftime("%B %d, %Y")
                grouped[entity]["session"].append(
                    (session_label, session.session_id, first_turn.turn_id)
                )

    # Flatten: one triple per (entity, attribute), pick a representative
    triples = []
    for entity, attr_map in grouped.items():
        n_sessions = len(entity_sessions[entity])
        # Skip very short or generic entity names (likely noise)
        if len(entity) <= 2:
            continue
        _GENERIC_ENTITIES = {
            "model", "system", "data", "time", "day", "week", "month",
            "thing", "stuff", "way", "lot", "bit", "part", "kind",
        }
        if entity.lower() in _GENERIC_ENTITIES:
            continue

        for attribute, values in attr_map.items():
            if not values:
                continue
            # Pick the first occurrence as representative
            val, sid, tid = values[0]
            all_sids = list(entity_sessions[entity])
            evidence = {
                "session_id": sid,
                "turn_id": tid,
                "all_sessions": all_sids,
            }
            # Difficulty: easy if single session, medium if 2-3, hard if 4+
            if n_sessions >= 4:
                difficulty = "hard"
            elif n_sessions >= 2:
                difficulty = "medium"
            else:
                difficulty = "easy"

            triples.append((entity, attribute, val, evidence, difficulty))

    return triples


# ---------------------------------------------------------------------------
# Lifecycle metadata extraction
# ---------------------------------------------------------------------------

def _extract_location_triples(
    corpus: DialogueCorpus,
) -> List[Tuple[str, str, Any, Dict[str, Any], str]]:
    """Extract triples for location-related metadata from lifecycle data."""
    triples: List[Tuple[str, str, Any, Dict[str, Any], str]] = []

    if not corpus.locations:
        return triples

    location_map = {loc.location_id: loc for loc in corpus.locations}

    # home_location: from persona's home_location_id
    for aid, agent_state in corpus.agents.items():
        home_loc_id = getattr(agent_state.persona, "home_location_id", "")
        if home_loc_id and home_loc_id in location_map:
            loc = location_map[home_loc_id]
            triples.append((
                aid, "home_location", loc.name,
                {"session_id": "", "turn_id": "", "all_sessions": []},
                "easy",
            ))

    # work_location: most-frequent workplace from activity_logs
    if corpus.activity_logs:
        for aid, entries in corpus.activity_logs.items():
            loc_counts: Dict[str, int] = defaultdict(int)
            for entry in entries:
                if entry.activity_type in ("solo", "errand") and entry.location_id:
                    loc = location_map.get(entry.location_id)
                    if loc and loc.location_type == "workplace":
                        loc_counts[entry.location_id] += 1
                # Also count dialogue/group_meeting at workplace locations
                if entry.activity_type in ("dialogue", "group_meeting") and entry.location_id:
                    loc = location_map.get(entry.location_id)
                    if loc and loc.location_type == "workplace":
                        loc_counts[entry.location_id] += 1
            if loc_counts:
                best_loc_id = max(loc_counts, key=loc_counts.get)
                loc = location_map[best_loc_id]
                triples.append((
                    aid, "work_location", loc.name,
                    {"session_id": "", "turn_id": "", "all_sessions": []},
                    "medium",
                ))

    # conversation_location: where a session took place
    for session in corpus.sessions:
        if session.location_id and session.location_id in location_map:
            loc = location_map[session.location_id]
            # Use entity mentions from the session as the entity being queried
            session_entities = set()
            for turn in session.turns:
                session_entities.update(turn.entities_mentioned)
            for entity in list(session_entities)[:2]:  # limit per session
                if len(entity) > 2:
                    triples.append((
                        entity, "conversation_location", loc.name,
                        {
                            "session_id": session.session_id,
                            "turn_id": session.turns[0].turn_id if session.turns else "",
                            "all_sessions": [session.session_id],
                        },
                        "medium",
                    ))

    return triples


def _extract_group_triples(
    corpus: DialogueCorpus,
) -> List[Tuple[str, str, Any, Dict[str, Any], str]]:
    """Extract triples for group-related metadata from lifecycle data."""
    triples: List[Tuple[str, str, Any, Dict[str, Any], str]] = []

    if not corpus.groups:
        return triples

    location_map = {loc.location_id: loc for loc in corpus.locations} if corpus.locations else {}

    for group in corpus.groups:
        # group_membership: each member is in this group
        for member_id, role_name in group.members.items():
            triples.append((
                member_id, "group_membership", group.name,
                {"session_id": "", "turn_id": "", "all_sessions": [], "context": group.name},
                "easy",
            ))

            # group_role: what role does this member play
            if role_name:
                triples.append((
                    member_id, "group_role", role_name,
                    {"session_id": "", "turn_id": "", "all_sessions": [], "context": group.name},
                    "medium",
                ))

        # meeting_location: where does the group meet
        if group.home_location_id and group.home_location_id in location_map:
            loc = location_map[group.home_location_id]
            triples.append((
                group.name, "meeting_location", loc.name,
                {"session_id": "", "turn_id": "", "all_sessions": []},
                "easy",
            ))

        # members: who is in the group
        if group.members:
            member_names = ", ".join(
                _slug_to_display(mid) for mid in sorted(group.members.keys())
            )
            triples.append((
                group.name, "members", member_names,
                {"session_id": "", "turn_id": "", "all_sessions": []},
                "medium",
            ))

    return triples
