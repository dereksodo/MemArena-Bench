"""Query-time rule for eval instances (every dimension except D6).

``metadata.query_timestamp`` is the simulated time at which a question is
asked. Deployable backends only see history up to it: the memory-cache
builder (``scripts/reproduce/build_memory_cache.py``) ingests every session
whose day is <= ``int(query_timestamp)``, and the BM25 / dense retrievers drop
messages delivered on a later day. Oracle ignores it.

Rule (applied in ``MASim/pipeline/orchestrator.py`` and by
``scripts/fix_query_timestamps.py``):

1. ``after_last_evidence``: the item has evidence sessions
   (``metadata.evidence_sessions``). The question is asked when the evidence
   session that ends last has ended: the max over the evidence sessions of
   their start time, end time and turn timestamps. Every evidence session is
   therefore visible to the deployable backends.
2. ``after_last_ego_session``: no evidence sessions (e.g. the abstain items of
   d3_confabulation). The question is asked when the ego's last session has
   ended, so the backends see the ego's whole history, the same history the
   ``full_ego`` Oracle context covers.
3. ``after_last_session``: no evidence and no known ego sessions. The question
   is asked when the last session of the world has ended.

D6 (d4_permission) keeps its own rule, ``d4_permission.horizon_timestamp``:
last ingested day + 0.99, after every session of the world.

The previous rule (start of the FIRST evidence session, none without
evidence) hid later evidence sessions from the time-filtered backends.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

RULE_AFTER_LAST_EVIDENCE = "after_last_evidence"
RULE_AFTER_LAST_EGO_SESSION = "after_last_ego_session"
RULE_AFTER_LAST_SESSION = "after_last_session"


def _get(obj: Any, key: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


def session_end_time(sess: Any) -> float:
    """Latest timestamp of a session: start, end and every turn.

    Accepts a ``MASim.core.schema.Session`` or its ``corpus_sessions.jsonl``
    dict form.
    """
    times = []
    for key in ("start_time", "end_time"):
        v = _get(sess, key)
        if v is not None:
            times.append(float(v))
    for turn in _get(sess, "turns") or []:
        v = _get(turn, "timestamp")
        if v is not None:
            times.append(float(v))
    return max(times) if times else 0.0


def session_end_times(sessions: Iterable[Any]) -> dict:
    """``{session_id: session_end_time}`` for a corpus."""
    return {str(_get(s, "session_id")): session_end_time(s) for s in sessions}


def query_timestamp(
    evidence: Sequence[str],
    ego_sessions: Optional[Sequence[str]],
    session_ends: Mapping[str, float],
) -> Tuple[float, str]:
    """Return ``(query_timestamp, rule)`` for one instance (see module docstring)."""
    ev_ends = [session_ends[s] for s in (evidence or []) if s in session_ends]
    if ev_ends:
        return max(ev_ends), RULE_AFTER_LAST_EVIDENCE
    ego_ends = [session_ends[s] for s in (ego_sessions or []) if s in session_ends]
    if ego_ends:
        return max(ego_ends), RULE_AFTER_LAST_EGO_SESSION
    if not session_ends:
        raise ValueError("empty corpus: no query time")
    return max(session_ends.values()), RULE_AFTER_LAST_SESSION
