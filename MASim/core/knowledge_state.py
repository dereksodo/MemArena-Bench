"""KnowledgeState: per-agent incremental knowledge tracking.

Tracks what each agent knows (facts, events, social information)
to enable ground-truth derivation and ego-centric projection.

After each dialogue session, call update_from_session() to populate
the agent's fact store from the completed turns.  This enables:
  - Natural cross-session anaphora in future dialogues (agent can
    reference things they heard before)
  - Accurate ground-truth for D2 (coreference) and D1 (conflict)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from MASim.core.schema import ActivityEntry, Session

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def _is_assistant(pid: str) -> bool:
    return pid.startswith("assistant_") or pid == "__assistant__"


def _tokenize(s: str) -> List[str]:
    """Tokenize a string into lowercase alphanumeric words for BM25."""
    return [t.lower() for t in _WORD_RE.findall(s or "")]


def _slug_to_display_name(slug: str) -> str:
    """Convert an agent ID slug to a readable display name.

    e.g. "maya_chen" -> "Maya Chen", "world" -> "someone",
         "__assistant__" -> "AI Assistant"
    """
    if slug == "world":
        return "someone"
    if _is_assistant(slug):
        return "AI Assistant"
    return " ".join(w.capitalize() for w in slug.split("_"))


@dataclass
class KnownFact:
    """A single fact known to an agent."""
    fact_id: str
    content: str
    source_agent: str     # who told this agent
    session_id: str       # when it was learned
    timestamp: float
    confidence: float = 1.0
    superseded_by: Optional[str] = None  # fact_id of a contradicting fact


@dataclass
class KnownFeeling:
    """A single feeling/mood recorded for an agent after a session."""
    feeling_id: str
    content: str            # e.g. "felt surprised when Maya announced leaving"
    trigger_agent: str      # who caused it, or "self"
    session_id: str
    timestamp: float
    valence: float = 0.0   # -1.0 negative to +1.0 positive
    intensity: float = 0.5  # 0.0 mild to 1.0 strong


@dataclass
class KnowledgeState:
    """Incremental knowledge state for a single agent.

    Maintains a record of all facts known to the agent,
    indexed by entity and by source for efficient querying.

    Also tracks per-person impressions: behavioural observations about
    specific other agents accumulated across sessions (e.g. "tends to
    deflect with humour when the topic turns serious").
    """
    agent_id: str = ""
    facts: Dict[str, KnownFact] = field(default_factory=dict)              # fact_id -> KnownFact
    feelings: Dict[str, KnownFeeling] = field(default_factory=dict)        # feeling_id -> KnownFeeling
    entity_index: Dict[str, Set[str]] = field(default_factory=dict)        # entity -> set of fact_ids
    events_seen: Set[str] = field(default_factory=set)                     # event_ids observed
    sessions_participated: List[str] = field(default_factory=list)         # session_ids in order
    person_impressions: Dict[str, List[str]] = field(default_factory=dict) # agent_id -> observations
    # BM25 internals (not serialized, rebuilt lazily)
    _fact_corpus_keys: List[str] = field(init=False, repr=False, default_factory=list)
    _feeling_corpus_keys: List[str] = field(init=False, repr=False, default_factory=list)
    _fact_bm25: Any = field(init=False, repr=False, default=None)
    _feeling_bm25: Any = field(init=False, repr=False, default=None)
    _bm25_dirty: bool = field(init=False, repr=False, default=True)

    def add_fact(self, fact: KnownFact) -> None:
        """Record a new fact in the knowledge state."""
        self.facts[fact.fact_id] = fact
        self._bm25_dirty = True

    def add_entity_fact(self, entity: str, fact_id: str) -> None:
        """Index a fact by entity mention."""
        if entity not in self.entity_index:
            self.entity_index[entity] = set()
        self.entity_index[entity].add(fact_id)

    def observe_event(self, event_id: str) -> None:
        """Record that this agent observed an event."""
        self.events_seen.add(event_id)

    def participate_in_session(self, session_id: str) -> None:
        """Record participation in a dialogue session."""
        if session_id not in self.sessions_participated:
            self.sessions_participated.append(session_id)

    def knows_about(self, entity: str) -> List[KnownFact]:
        """Retrieve all facts related to an entity."""
        fact_ids = self.entity_index.get(entity, set())
        return [self.facts[fid] for fid in fact_ids if fid in self.facts]

    def has_seen_event(self, event_id: str) -> bool:
        return event_id in self.events_seen

    def get_latest_fact(self, entity: str) -> Optional[KnownFact]:
        """Get the most recent fact about an entity."""
        facts = self.knows_about(entity)
        if not facts:
            return None
        return max(facts, key=lambda f: f.timestamp)

    def get_conflicting_facts(self) -> List[tuple]:
        """Find pairs of facts where one supersedes another."""
        conflicts = []
        for fid, fact in self.facts.items():
            if fact.superseded_by and fact.superseded_by in self.facts:
                conflicts.append((fact, self.facts[fact.superseded_by]))
        return conflicts

    def update_from_session(self, session: "Session", agent_id: str) -> None:
        """Populate knowledge state from a completed dialogue session.

        Called after _batch_extract_session_knowledge() has annotated turns.
        For every turn the agent *heard*, we store its distilled facts as
        individual KnownFact entries (falling back to the raw turn text when
        no facts were extracted).  Every fact is indexed by the turn's entities
        so entity-targeted retrieval works correctly.
        """
        self.participate_in_session(session.session_id)

        for turn in session.turns:
            if turn.speaker_id == agent_id:
                continue  # own utterances — already known
            is_group = turn.metadata.get("group_conversation") or turn.listener_id == "group"
            is_participant = agent_id in session.participants
            is_listener = turn.listener_id == agent_id
            if not (is_listener or (is_group and is_participant)):
                continue

            # Use distilled facts when available; fall back to raw turn text
            contents: List[str] = turn.extracted_facts if turn.extracted_facts else [turn.text]

            for content in contents:
                fact_id = f"{agent_id}_fact_{self._next_fact_id()}"
                fact = KnownFact(
                    fact_id=fact_id,
                    content=content,
                    source_agent=turn.speaker_id,
                    session_id=session.session_id,
                    timestamp=turn.timestamp,
                )
                self.add_fact(fact)

                # Index every fact by all entities mentioned in its source turn
                for entity in turn.entities_mentioned:
                    self.add_entity_fact(entity, fact_id)

    def _next_fact_id(self) -> str:
        """Generate a monotonically increasing fact counter suffix."""
        n = len(self.facts)
        return f"{n:06d}"

    def get_recent_facts(self, n: int = 10) -> List[KnownFact]:
        """Return the n most recently learned facts, newest first."""
        sorted_facts = sorted(self.facts.values(), key=lambda f: f.timestamp, reverse=True)
        return sorted_facts[:n]

    # ------------------------------------------------------------------
    # Feelings store
    # ------------------------------------------------------------------

    def add_feeling(self, feeling: KnownFeeling) -> None:
        """Record a new feeling in the knowledge state."""
        self.feelings[feeling.feeling_id] = feeling
        self._bm25_dirty = True

    def get_recent_feelings(self, n: int = 5) -> List[KnownFeeling]:
        """Return the n most recent feelings, newest first."""
        return sorted(self.feelings.values(), key=lambda f: f.timestamp, reverse=True)[:n]

    # ------------------------------------------------------------------
    # BM25 retrieval
    # ------------------------------------------------------------------

    def _rebuild_bm25(self) -> None:
        """Rebuild BM25 indices from current facts/feelings. Called lazily."""
        from rank_bm25 import BM25Okapi

        # Facts index
        self._fact_corpus_keys = list(self.facts.keys())
        tokenized = [_tokenize(self.facts[k].content) for k in self._fact_corpus_keys]
        self._fact_bm25 = BM25Okapi(tokenized) if tokenized else None

        # Feelings index
        self._feeling_corpus_keys = list(self.feelings.keys())
        tokenized_f = [_tokenize(self.feelings[k].content) for k in self._feeling_corpus_keys]
        self._feeling_bm25 = BM25Okapi(tokenized_f) if tokenized_f else None

        self._bm25_dirty = False

    def search_facts(self, query: str, top_k: int = 5) -> List[KnownFact]:
        """Return top-k facts matching query via BM25. Falls back to recency."""
        if self._bm25_dirty:
            self._rebuild_bm25()
        if not self._fact_bm25 or not self._fact_corpus_keys:
            return self.get_recent_facts(top_k)
        scores = self._fact_bm25.get_scores(_tokenize(query))
        scored = sorted(
            zip(self._fact_corpus_keys, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [
            self.facts[fid] for fid, sc in scored[:top_k] if sc > 0
        ] or self.get_recent_facts(top_k)

    def search_feelings(self, query: str, top_k: int = 3) -> List[KnownFeeling]:
        """Return top-k feelings matching query via BM25. Falls back to recency."""
        if self._bm25_dirty:
            self._rebuild_bm25()
        if not self._feeling_bm25 or not self._feeling_corpus_keys:
            return self.get_recent_feelings(top_k)
        scores = self._feeling_bm25.get_scores(_tokenize(query))
        scored = sorted(
            zip(self._feeling_corpus_keys, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [
            self.feelings[fid] for fid, sc in scored[:top_k] if sc > 0
        ] or self.get_recent_feelings(top_k)

    def search_memory(self, query: str, top_k: int = 8) -> List:
        """Combined search: interleave facts and feelings by timestamp."""
        facts = self.search_facts(query, top_k)
        feelings = self.search_feelings(query, top_k // 2)
        combined: List[Tuple[float, Any]] = (
            [(f.timestamp, f) for f in facts]
            + [(f.timestamp, f) for f in feelings]
        )
        combined.sort(key=lambda x: x[0])
        return [item for _, item in combined][:top_k]

    def get_context_facts(
        self,
        entity_hints: Optional[List[str]] = None,
        max_facts: int = 8,
    ) -> List[KnownFact]:
        """Return KnownFact objects relevant to the given entity hints.

        Priority order: entity-targeted facts (sorted by recency) first,
        then padded with the most recent facts not already included.
        Falls back to recency-only when no hints are given.
        """
        candidate_ids: Set[str] = set()

        if entity_hints:
            for hint in entity_hints:
                hint_lower = hint.lower()
                for indexed_entity, fact_ids in self.entity_index.items():
                    if hint_lower in indexed_entity.lower() or indexed_entity.lower() in hint_lower:
                        candidate_ids.update(fact_ids)

        targeted = sorted(
            [self.facts[fid] for fid in candidate_ids if fid in self.facts],
            key=lambda f: f.timestamp,
            reverse=True,
        )[:max_facts]

        seen_ids = {f.fact_id for f in targeted}
        extras = [
            f for f in self.get_recent_facts(max_facts)
            if f.fact_id not in seen_ids
        ][: max(0, max_facts - len(targeted))]

        combined = targeted + extras
        combined.sort(key=lambda f: f.timestamp)  # chronological for readability
        return combined

    def summarise_for_context(
        self,
        entity_hints: Optional[List[str]] = None,
        max_facts: int = 8,
    ) -> str:
        """Format context-relevant facts as a bullet list for prompt injection."""
        facts = self.get_context_facts(entity_hints, max_facts)
        if not facts:
            return ""
        lines = []
        for f in facts:
            name = _slug_to_display_name(f.source_agent)
            lines.append(f"- {name}: \"{f.content}\"")
        return "\n".join(lines)

    def pick_memory_to_pin(
        self,
        entity_hints: Optional[List[str]] = None,
    ) -> Optional[KnownFact]:
        """Return the single most contextually relevant fact to explicitly reference.

        Used for probabilistic memory pinning: surfaces the top-ranked fact so
        the agent can be explicitly instructed to weave it into the current turn.
        Returns None if the agent has no memories yet.
        """
        facts = self.get_context_facts(entity_hints, max_facts=5)
        # Prefer facts with extracted content (distilled) over raw turn text
        distilled = [f for f in facts if len(f.content) < 200 and not f.content.endswith("...")]
        return (distilled or facts or [None])[0]

    def summarise_for_prompt(self, max_facts: int = 8) -> str:
        """Recency-only summary — convenience wrapper for contexts without entity hints."""
        return self.summarise_for_context(entity_hints=None, max_facts=max_facts)

    def update_person_impression(self, agent_id: str, observations: List[str]) -> None:
        """Accumulate behavioural observations about another agent.

        Observations are appended across sessions (capped at 5 per person to
        avoid bloating prompts).  Exact duplicates are silently skipped.
        """
        if not observations:
            return
        existing = self.person_impressions.setdefault(agent_id, [])
        for obs in observations:
            obs = obs.strip()
            if obs and obs not in existing:
                existing.append(obs)
        # Keep only the most recent 5 observations
        self.person_impressions[agent_id] = existing[-5:]

    def impression_for_prompt(self, agent_id: str) -> str:
        """Return a bullet-list of behavioural impressions about a specific agent.

        Returns an empty string when no impressions have been recorded yet.
        """
        observations = self.person_impressions.get(agent_id, [])
        if not observations:
            return ""
        return "\n".join(f"- {obs}" for obs in observations)

    def update_from_activity(self, entry: "ActivityEntry") -> None:
        """Store a non-dialogue activity as a self-authored KnownFact.

        Called after MemoryWriter has populated entry.memory_text.
        Dialogue activities are already captured via update_from_session();
        skip them here to avoid duplication.
        """
        if entry.activity_type == "dialogue":
            return
        if not entry.memory_text:
            return

        fact_id = f"{self.agent_id}_act_{self._next_fact_id()}"
        fact = KnownFact(
            fact_id=fact_id,
            content=entry.memory_text,
            source_agent=self.agent_id,   # self-authored memory
            session_id="",
            timestamp=entry.start_time,
            confidence=0.9,
        )
        self.add_fact(fact)

        # Index by location and group so the memory surfaces in context retrieval
        if entry.location_id:
            self.add_entity_fact(entry.location_id, fact_id)
        if entry.group_id:
            self.add_entity_fact(entry.group_id, fact_id)
        # Index by activity type as a broad category
        self.add_entity_fact(entry.activity_type, fact_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "n_facts": len(self.facts),
            "n_feelings": len(self.feelings),
            "n_entities": len(self.entity_index),
            "n_events_seen": len(self.events_seen),
            "n_sessions": len(self.sessions_participated),
            "n_person_impressions": sum(len(v) for v in self.person_impressions.values()),
            "person_impressions": {k: list(v) for k, v in self.person_impressions.items()},
        }
