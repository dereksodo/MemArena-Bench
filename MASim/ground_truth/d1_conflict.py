"""D1: Conflict preservation — generate evaluation instances from injected + organic conflicts."""

from __future__ import annotations

import re
import uuid
from typing import Any, Dict, List, Set

import numpy as np

from MASim.core.schema import ConflictGT, DialogueCorpus, Dimension, EvalInstance, is_assistant
from MASim.utils.logging import get_logger

log = get_logger(__name__)


# Multiple human-sounding query templates.  {topic} is a short topical hint
# derived from entities;  {agent_a} / {agent_b} are the source and conflicting
# agents respectively.  Templates are designed so that a simple keyword search
# cannot locate the answer — they avoid quoting facts verbatim and use indirect
# phrasing.
_QUERY_TEMPLATES = [
    (
        "I feel like I got two different stories about {topic}. "
        "Can you help me figure out which version is accurate?"
    ),
    (
        "Something about {topic} doesn't add up — I think I heard "
        "one thing from {agent_a} and something else later. "
        "What exactly was said?"
    ),
    (
        "There seems to be some mix-up around {topic}. "
        "Do you remember hearing inconsistent details about it?"
    ),
    (
        "I'm confused about {topic}. I recall getting different "
        "information at different times — can you sort out what happened?"
    ),
    (
        "Did anyone give me conflicting info about {topic}? "
        "I vaguely remember hearing two versions."
    ),
    (
        "I need to double-check something about {topic}. "
        "I think the details changed between conversations — "
        "what do you have on file?"
    ),
    (
        "Regarding {topic} — was there a point where the story "
        "shifted or someone told me something different from what "
        "I'd heard before?"
    ),
    (
        "Can you pull up everything related to {topic}? "
        "I want to see if the facts are consistent across "
        "what different people told me."
    ),
]


def generate_instances(
    corpus: DialogueCorpus,
    conflict_gts: List[ConflictGT],
    max_instances: int = 200,
    seed: int = 42,
) -> List[EvalInstance]:
    """Generate D1 conflict preservation eval instances.

    Generates one instance per (conflict_gt, query_template) pair so that
    each injected conflict contributes up to len(_QUERY_TEMPLATES) instances
    with varied natural-language phrasing.  The total is capped at
    max_instances.
    """
    rng = np.random.default_rng(seed)

    # Build entity lookup for natural topic references
    session_entities = {}
    for session in corpus.sessions:
        entities = set()
        for turn in session.turns:
            entities.update(turn.entities_mentioned)
        session_entities[session.session_id] = entities

    instances = []
    for gt in conflict_gts:
        if len(instances) >= max_instances:
            break

        topic = _build_topic_hint(gt, session_entities)
        agent_a = _first_name(gt.source_agent)
        agent_b = _first_name(gt.conflicting_agent)
        difficulty = _assess_difficulty(gt)

        # Shuffle templates so different conflicts use different orderings
        template_order = list(range(len(_QUERY_TEMPLATES)))
        rng.shuffle(template_order)

        for t_idx in template_order:
            if len(instances) >= max_instances:
                break
            query = _QUERY_TEMPLATES[t_idx].format(
                topic=topic, agent_a=agent_a, agent_b=agent_b
            )
            instance = EvalInstance(
                instance_id=f"d1_{uuid.uuid4().hex[:12]}",
                dimension=Dimension.D1_CONFLICT,
                query=query,
                ground_truth={
                    "source_agent": gt.source_agent,
                    "conflicting_agent": gt.conflicting_agent,
                    "original_session": gt.original_session,
                    "conflicting_session": gt.conflicting_session,
                    "original_timestamp": gt.original_timestamp,
                    "conflicting_timestamp": gt.conflicting_timestamp,
                    "original_fact": gt.fact,
                    "contradictory_fact": gt.contradictory_fact,
                    "original_detail": gt.original_detail,
                    "changed_detail": gt.changed_detail,
                },
                difficulty=difficulty,
                asker_agent_id=gt.conflicting_agent,
                answerer_agent_id=gt.source_agent,
                metadata={
                    "evidence_sessions": [gt.original_session, gt.conflicting_session],
                    "template_idx": t_idx,
                },
            )
            instances.append(instance)
    return instances


def _build_topic_hint(gt: ConflictGT, session_entities: dict) -> str:
    """Build a short topical hint from the actual conflict content.

    Prioritises:
      1. The extracted detail fields (original_detail / changed_detail)
      2. A diff of original vs contradictory fact (words unique to one version)
      3. Content words from the fact itself
    """
    # 1. If detail fields exist, derive hint from them
    if gt.original_detail and gt.changed_detail:
        combined = f"{gt.original_detail} {gt.changed_detail}"
        hint_words = _extract_hint_words(combined)
        if hint_words:
            return " ".join(hint_words)

    # 2. Diff the two versions — pick words that differ
    if gt.fact and gt.contradictory_fact:
        orig_words = set(gt.fact.lower().split())
        contra_words = set(gt.contradictory_fact.lower().split())
        # Words unique to either version = the actual change
        diff_words = (orig_words ^ contra_words)
        hint = _extract_hint_words(" ".join(diff_words))
        if hint:
            return " ".join(hint)
        # If the diff is too small, use shared content words
        hint = _extract_hint_words(gt.fact)
        if hint:
            return " ".join(hint)

    # 3. Entities from the session as last resort
    entities = session_entities.get(gt.original_session, set())
    if entities:
        usable = sorted(e for e in entities if len(e) > 2)[:3]
        if usable:
            return " and ".join(usable)

    return "that recent topic"


_HINT_STOP = frozenset({
    "the", "a", "an", "is", "was", "are", "were", "that", "this",
    "with", "from", "have", "has", "been", "they", "them", "their",
    "about", "just", "also", "would", "could", "should", "really",
    "very", "much", "some", "into", "will", "your", "it's", "you",
    "and", "but", "for", "not", "don't", "didn't", "doesn't",
    "like", "know", "think", "said", "told", "yeah", "okay",
    "it", "i'm", "i'll", "i've", "we're", "he's", "she's",
    "what", "how", "when", "where", "who", "why", "can", "do",
    "did", "does", "got", "get", "going", "went", "come", "came",
    "one", "two", "more", "than", "then", "too", "so", "if",
})


def _extract_hint_words(text: str, max_words: int = 4) -> List[str]:
    """Extract distinctive content words from text for use as a topic hint."""
    words = re.findall(r"[a-z][a-z'-]+", text.lower())
    useful = [w for w in words if len(w) > 3 and w not in _HINT_STOP]
    # Deduplicate while preserving order
    seen = set()
    result = []
    for w in useful:
        if w not in seen:
            seen.add(w)
            result.append(w)
        if len(result) >= max_words:
            break
    return result


def _first_name(agent_slug: str) -> str:
    """Convert 'eleanor_vance' → 'Eleanor'."""
    if is_assistant(agent_slug):
        return "the assistant"
    return agent_slug.split("_")[0].title()


def _assess_difficulty(gt: ConflictGT) -> str:
    """Heuristic difficulty assessment based on temporal gap."""
    gap = abs(gt.conflicting_timestamp - gt.original_timestamp)
    if gap < 10:
        return "easy"
    elif gap < 100:
        return "medium"
    return "hard"


# ---------------------------------------------------------------------------
# Organic conflict detection from memory probe sessions
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOP = frozenset({
    "the", "a", "an", "is", "was", "are", "were", "that", "this",
    "with", "from", "have", "has", "been", "they", "them", "their",
    "about", "just", "also", "would", "could", "should", "really",
    "very", "much", "some", "into", "will", "your", "you", "not",
    "don", "didn", "doesn", "and", "but", "for", "its", "it",
})


def _content_words(text: str) -> Set[str]:
    """Extract meaningful content words from text."""
    return {w for w in _WORD_RE.findall(text.lower()) if len(w) > 2 and w not in _STOP}


def detect_organic_conflicts(
    corpus: DialogueCorpus,
    agents: Dict[str, Any],
    max_overlap: float = 0.3,
) -> List[ConflictGT]:
    """Detect version mismatches between source facts and probe restatements.

    For each probe session, compares the person's restated facts
    (extracted_facts from their turns) against the original source facts
    (looked up via source_fact_ids in session metadata). Low lexical overlap
    indicates the person restated differently — an organic conflict.

    Args:
        corpus: full dialogue corpus (must include probe sessions)
        agents: Dict[agent_id, EntityAgent] with populated knowledge stores
        max_overlap: Jaccard threshold below which we flag a mismatch

    Returns:
        List of ConflictGT entries representing organic conflicts.
    """
    organic: List[ConflictGT] = []

    for session in corpus.sessions:
        if session.metadata.get("session_type") != "person_agent_probe":
            continue

        source_fact_ids = session.metadata.get("source_fact_ids", [])
        if not source_fact_ids:
            continue

        # Identify the human participant
        human_pids = [p for p in session.participants if not is_assistant(p)]
        if not human_pids:
            continue
        agent_id = human_pids[0]
        agent = agents.get(agent_id)
        if not agent:
            continue

        # Look up original source facts from agent's knowledge store
        source_facts = []
        for fid in source_fact_ids:
            fact = agent.knowledge.facts.get(fid)
            if fact:
                source_facts.append(fact)
        if not source_facts:
            continue

        # Collect all extracted facts from the person's turns in this probe
        restatements: List[str] = []
        for turn in session.turns:
            if turn.speaker_id == agent_id and turn.extracted_facts:
                restatements.extend(turn.extracted_facts)
        if not restatements:
            continue

        # Compare each source fact against restatements
        restatement_words = set()
        for r in restatements:
            restatement_words.update(_content_words(r))

        for fact in source_facts:
            fact_words = _content_words(fact.content)
            if not fact_words:
                continue

            # Jaccard overlap
            intersection = fact_words & restatement_words
            union = fact_words | restatement_words
            overlap = len(intersection) / len(union) if union else 0.0

            if overlap < max_overlap:
                # Find the most different restatement for the contradictory_fact
                best_restatement = ""
                best_diff = 1.0
                for r in restatements:
                    r_words = _content_words(r)
                    r_overlap = (
                        len(fact_words & r_words) / len(fact_words | r_words)
                        if (fact_words | r_words) else 0.0
                    )
                    if r_overlap < best_diff:
                        best_diff = r_overlap
                        best_restatement = r

                gt = ConflictGT(
                    fact=fact.content,
                    contradictory_fact=best_restatement,
                    source_agent=fact.source_agent,
                    conflicting_agent=agent_id,
                    original_session=fact.session_id,
                    conflicting_session=session.session_id,
                    original_timestamp=fact.timestamp,
                    conflicting_timestamp=session.end_time,
                )
                organic.append(gt)

    log.info("Detected %d organic conflicts from %d probe sessions",
             len(organic),
             sum(1 for s in corpus.sessions
                 if s.metadata.get("session_type") == "person_agent_probe"))
    return organic
