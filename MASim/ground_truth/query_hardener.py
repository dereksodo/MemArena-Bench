"""Query hardening techniques to defeat surface-level RAG.

1. Paraphrase queries (D1-D6) — rewrite to eliminate verbatim lexical overlap
2. Temporal reasoning (D8) — before/after, ordering, recency questions
3. Agent-perspective scoping (D7 append) — questions only answerable from one viewpoint
4. Counterfactual probes (D10) — present altered facts, expect correction
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from MASim.core.schema import DialogueCorpus, Dimension, EvalInstance, Session, is_assistant
from MASim.ground_truth.json_parser import parse_json_array as _parse_llm_json_array
from MASim.ground_truth.json_parser import parse_json_object as _parse_llm_json_object
from memarena.text_cleaning import clean_llm_text
from MASim.prompts import (
    HARDENER_COUNTERFACTUAL_SYSTEM as _COUNTERFACTUAL_SYSTEM,
    HARDENER_COUNTERFACTUAL_USER as _COUNTERFACTUAL_USER,
    HARDENER_PARAPHRASE_SYSTEM as _PARAPHRASE_SYSTEM,
    HARDENER_PERSPECTIVE_SYSTEM as _PERSPECTIVE_SYSTEM,
    HARDENER_PERSPECTIVE_USER as _PERSPECTIVE_USER,
    HARDENER_TEMPORAL_SYSTEM as _TEMPORAL_SYSTEM,
    HARDENER_TEMPORAL_USER as _TEMPORAL_USER,
)
from MASim.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Paraphrase samples per query: an empty or rejected first paraphrase gets one more try.
PARAPHRASE_ATTEMPTS = 2


@dataclass
class HardeningConfig:
    paraphrase_enabled: bool = True
    temporal_enabled: bool = True
    perspective_enabled: bool = True
    counterfactual_enabled: bool = True
    max_instances_per_dim: int = 30
    max_lexical_overlap: float = 0.3  # Jaccard threshold for paraphrase quality
    seed: int = 42

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HardeningConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _jaccard(a: str, b: str) -> float:
    """Compute Jaccard token overlap between two strings."""
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _parse_json_array(text: str) -> List[Dict[str, Any]]:
    """Best-effort parse a JSON array from LLM output."""
    try:
        return _parse_llm_json_array(text)
    except (ValueError, TypeError):
        return []
    return []


def _parse_json_object(text: str) -> Dict[str, Any]:
    """Best-effort parse a JSON object from LLM output."""
    try:
        return _parse_llm_json_object(text)
    except (ValueError, TypeError):
        return {}
    return {}


# ---------------------------------------------------------------------------
# Question feature labels
# ---------------------------------------------------------------------------

# Default feature per dimension; some dimensions refine further from ground_truth.
_DIM_DEFAULT_FEATURE = {
    Dimension.D1_CONFLICT: "conflict_detection",
    Dimension.D2_ANAPHORA: "anaphora_resolution",
    Dimension.D3_CONFABULATION: "confabulation_resistance",
    Dimension.D4_PERMISSION: "permission_compliance",
    Dimension.D5_CLOZE: "cloze_reconstruction",
    Dimension.D6_METADATA: "metadata_recall",
    Dimension.D7_QA: "factual_recall",
    Dimension.D8_TEMPORAL: "temporal_reasoning",
    Dimension.D10_COUNTERFACTUAL: "counterfactual_correction",
}

# Sub-features derived from ground_truth / metadata fields.
_D6_ATTR_FEATURE = {
    "speaker": "speaker_attribution",
    "session": "temporal_metadata",
    "context": "context_recall",
    "participants": "participant_recall",
    # Lifecycle metadata sub-features
    "home_location": "location_metadata",
    "work_location": "location_metadata",
    "conversation_location": "location_metadata",
    "group_membership": "group_metadata",
    "group_role": "group_metadata",
    "meeting_location": "group_metadata",
    "members": "group_metadata",
}


def _question_feature(inst: EvalInstance) -> str:
    """Derive a fine-grained question_feature tag for an instance."""
    dim = inst.dimension
    gt = inst.ground_truth
    meta = inst.metadata

    # D4 sub-features (the D4 builder sets them; the scorer groups by them)
    if dim == Dimension.D4_PERMISSION:
        if meta.get("question_feature"):
            return meta["question_feature"]
        if gt.get("sensitivity_category"):
            return "autonomous_privacy"
        if gt.get("known_requester"):
            return "known_requester"
        if gt.get("anonymous_querier"):
            return "anonymous_querier"
        return "permission_compliance"

    # D6 sub-features
    if dim == Dimension.D6_METADATA:
        return _D6_ATTR_FEATURE.get(gt.get("attribute", ""), "metadata_recall")

    # D7 sub-features
    if dim == Dimension.D7_QA:
        if meta.get("perspective_scoped"):
            return "perspective_recall"
        if gt.get("requires_temporal"):
            return "temporal_recall"
        return "factual_recall"

    # D8 sub-features
    if dim == Dimension.D8_TEMPORAL:
        ttype = gt.get("temporal_type", "")
        if ttype:
            return f"temporal_{ttype}"
        return "temporal_reasoning"

    return _DIM_DEFAULT_FEATURE.get(dim, "unknown")


def _infer_query_agent(
    inst: EvalInstance, session_map: Dict[str, Session],
) -> str:
    """Best-effort inference of which agent is being asked this question."""
    gt = inst.ground_truth
    meta = inst.metadata

    # Already set
    if meta.get("query_agent"):
        return meta["query_agent"]

    # D4 explicit field
    if gt.get("query_agent"):
        return gt["query_agent"]

    # D3 confabulation: prefer a participant from source_session (so the
    # query_agent's ego context actually contains the evidence)
    if gt.get("target_agent"):
        target = gt["target_agent"]
        source_sid = gt.get("source_session", "")
        if source_sid:
            source_sess = session_map.get(source_sid)
            if source_sess:
                for p in source_sess.participants:
                    if p != target:
                        return p
        # Fallback: any session containing the target
        for sid, sess in session_map.items():
            if target in sess.participants:
                for p in sess.participants:
                    if p != target:
                        return p
        return target  # fallback

    # D6 metadata: use the speaker from ground_truth
    if gt.get("value") and gt.get("attribute") == "speaker":
        # value is the speaker_id, but the *asker* is someone who was present
        source_sid = gt.get("source_session", "")
        sess = session_map.get(source_sid)
        if sess and sess.participants:
            return sess.participants[0]

    # Otherwise, use answerer_agent_id if set (the agent whose memory is tested).
    if inst.answerer_agent_id and not is_assistant(inst.answerer_agent_id):
        return inst.answerer_agent_id

    # Generic: pick first participant from evidence sessions
    evidence = meta.get("evidence_sessions", [])
    for sid in evidence:
        sess = session_map.get(sid)
        if sess and sess.participants:
            return sess.participants[0]

    # Last resort: pick first participant from first session
    if session_map:
        first = next(iter(session_map.values()))
        if first.participants:
            return first.participants[0]

    return ""


# ---------------------------------------------------------------------------
# QueryHardener
# ---------------------------------------------------------------------------

class QueryHardener:
    """Post-processing pass that hardens eval queries against surface-level RAG."""

    def __init__(self, llm_client: Any, cfg: Optional[HardeningConfig] = None):
        self.llm = llm_client
        self.cfg = cfg or HardeningConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

    # ----- public entry point -----

    def harden_all(
        self,
        instances: Dict[Dimension, List[EvalInstance]],
        corpus: DialogueCorpus,
    ) -> Dict[Dimension, List[EvalInstance]]:
        """Apply all enabled hardening techniques.

        Modifies *instances* in-place and may add new dimension keys.
        Returns the updated dict.
        """
        if self.cfg.paraphrase_enabled:
            log.info("Hardening: paraphrasing D1-D6 queries...")
            self._paraphrase_queries(instances)

        if self.cfg.temporal_enabled:
            log.info("Hardening: generating D8 temporal instances...")
            instances[Dimension.D8_TEMPORAL] = self._generate_temporal(corpus)
            # Append lifecycle boundary instances (sleep/wake/meeting)
            lifecycle_temporal = self._generate_temporal_lifecycle(corpus)
            if lifecycle_temporal:
                log.info("Hardening: appending %d D8 lifecycle boundary instances", len(lifecycle_temporal))
                instances[Dimension.D8_TEMPORAL].extend(lifecycle_temporal)

        if self.cfg.perspective_enabled:
            log.info("Hardening: generating perspective-scoped D7 instances...")
            extras = self._generate_perspective(corpus)
            instances.setdefault(Dimension.D7_QA, []).extend(extras)

        if self.cfg.counterfactual_enabled:
            log.info("Hardening: generating D10 counterfactual instances...")
            instances[Dimension.D10_COUNTERFACTUAL] = self._generate_counterfactual(corpus)

        # Always enrich: assign query_agent + question_feature to every instance
        log.info("Hardening: enriching metadata (query_agent, question_feature)...")
        self._enrich_metadata(instances, corpus)
        self._clean_queries(instances)

        return instances

    # ----- Technique 1: Paraphrase -----

    def _paraphrase_queries(
        self, instances: Dict[Dimension, List[EvalInstance]]
    ) -> None:
        """Rewrite queries for D1, D2, D4, D6 to eliminate verbatim overlap.

        D5 is intentionally excluded: the query IS the cloze passage with blanks
        and multiple-choice options — paraphrasing it destroys the test entirely.

        D4 (paper D6): topic-cued fact items keep their template wording and
        are not paraphrased; a paraphrase of any other D4 fact item must still
        pass the D6 query checks (``d4_permission.paraphrase_problems``), or
        the unhardened query is kept.
        """
        # Imported here: d4_permission imports this module.
        from MASim.ground_truth import d4_permission

        target_dims = {
            Dimension.D1_CONFLICT,
            Dimension.D2_ANAPHORA,
            Dimension.D4_PERMISSION,
            Dimension.D6_METADATA,
        }

        # Collect all instances that need paraphrasing
        to_paraphrase: List[EvalInstance] = []
        for dim in target_dims:
            to_paraphrase.extend(
                inst for inst in instances.get(dim, []) if d4_permission.paraphrase_allowed(inst)
            )

        if not to_paraphrase:
            return

        prompts = [
            {"system": _PARAPHRASE_SYSTEM, "user": inst.query, "tags": {"phase": "hardener_paraphrase"}}
            for inst in to_paraphrase
        ]

        log.info("Paraphrasing %d queries via LLM batch...", len(prompts))
        responses = list(self.llm.generate_batch(prompts))

        # An empty or rejected paraphrase gets one more sample (PARAPHRASE_ATTEMPTS in total).
        n_replaced = 0
        pending = list(range(len(to_paraphrase)))
        for attempt in range(PARAPHRASE_ATTEMPTS):
            if attempt:
                if not pending:
                    break
                log.info("Re-sampling %d empty or rejected paraphrases", len(pending))
                for i, resp in zip(pending, self.llm.generate_batch([prompts[i] for i in pending])):
                    responses[i] = resp
            still_pending = []
            for i in pending:
                inst = to_paraphrase[i]
                rephrased = clean_llm_text(responses[i])
                if not rephrased:
                    inst.metadata["paraphrase_rejected"] = True
                    still_pending.append(i)
                    continue

                # Compute overlap with the original source text in ground_truth
                source_text = self._extract_source_text(inst)
                overlap = _jaccard(rephrased, source_text) if source_text else 0.0

                d6_problems = d4_permission.paraphrase_problems(inst, rephrased)
                if overlap <= self.cfg.max_lexical_overlap and not d6_problems:
                    for key in ("paraphrase_rejected", "paraphrase_overlap", "paraphrase_rejected_checks"):
                        inst.metadata.pop(key, None)
                    inst.metadata["original_query"] = inst.query
                    inst.metadata["lexical_overlap_score"] = round(overlap, 4)
                    inst.metadata["hardened_query"] = rephrased
                    inst.query = rephrased
                    n_replaced += 1
                else:
                    # Keep original but record the failed attempt
                    inst.metadata["paraphrase_rejected"] = True
                    inst.metadata["paraphrase_overlap"] = round(overlap, 4)
                    if d6_problems:
                        inst.metadata["paraphrase_rejected_checks"] = d6_problems
                    still_pending.append(i)
            pending = still_pending

        log.info("Paraphrased %d / %d queries (%.0f%%)",
                 n_replaced, len(to_paraphrase),
                 100 * n_replaced / max(len(to_paraphrase), 1))

    @staticmethod
    def _extract_source_text(inst: EvalInstance) -> str:
        """Pull the main corpus text from ground_truth for overlap checking."""
        gt = inst.ground_truth
        # Try common ground truth fields that contain source text
        for key in ("deleted_span", "fact", "value"):
            if key in gt and gt[key]:
                return str(gt[key])
        return inst.query

    @staticmethod
    def _clean_queries(instances: Dict[Dimension, List[EvalInstance]]) -> None:
        """Remove reasoning wrappers from all generated query text."""
        for inst_list in instances.values():
            for inst in inst_list:
                cleaned = clean_llm_text(inst.query)
                if cleaned and cleaned != inst.query:
                    inst.metadata.setdefault("unclean_query", inst.query)
                    inst.query = cleaned
                hardened = inst.metadata.get("hardened_query")
                if isinstance(hardened, str):
                    inst.metadata["hardened_query"] = clean_llm_text(hardened)

    # ----- Technique 2: Temporal -----

    def _generate_temporal(self, corpus: DialogueCorpus) -> List[EvalInstance]:
        """Generate D8 temporal reasoning instances.

        Selects 3 full sessions per agent (early, middle, late) and asks the
        LLM to produce temporal questions from the complete transcripts.
        """
        _NUM_SESSIONS = 3  # number of full sessions to include per agent

        # Build per-agent chronological session list
        agent_sessions: Dict[str, List[Session]] = defaultdict(list)
        for session in corpus.sessions:
            if not session.turns:
                continue
            for pid in session.participants:
                agent_sessions[pid].append(session)

        # Sort chronologically and select agents with >= _NUM_SESSIONS sessions
        eligible_agents = []
        for aid, sess_list in agent_sessions.items():
            if is_assistant(aid):
                continue  # skip the PA assistant — not a real persona
            sess_list.sort(key=lambda s: s.start_time)
            if len(sess_list) >= _NUM_SESSIONS:
                eligible_agents.append(aid)

        if not eligible_agents:
            log.warning("No agents with >=%d sessions for temporal generation", _NUM_SESSIONS)
            return []

        max_agents = min(len(eligible_agents), self.cfg.max_instances_per_dim // 3 + 1)
        selected = list(self.rng.choice(eligible_agents, size=max_agents, replace=False))

        prompts = []
        # Track which 3 sessions were picked per agent for ground truth
        selected_sessions: Dict[str, List[Session]] = {}
        for aid in selected:
            agent_name = self._first_name(aid)
            sess_list = agent_sessions[aid]
            # Pick 3 spread-out sessions: first, middle, last
            n = len(sess_list)
            indices = [0, n // 2, n - 1]
            # Deduplicate in case n < 4
            indices = sorted(set(indices))
            picked = [sess_list[i] for i in indices]
            # Pad to _NUM_SESSIONS if dedup reduced count (only when n==2, shouldn't happen)
            while len(picked) < _NUM_SESSIONS and len(picked) < n:
                for i in range(n):
                    if sess_list[i] not in picked:
                        picked.append(sess_list[i])
                        break
            selected_sessions[aid] = picked

            # Format full transcripts with natural names
            transcript_blocks = []
            for i, sess in enumerate(picked):
                lines = []
                for t in sess.turns:
                    speaker = self._first_name(t.speaker_id)
                    lines.append(f"  {speaker}: {t.text}")
                participants = ", ".join(
                    self._first_name(p) for p in sess.participants
                )
                transcript_blocks.append(
                    f"--- Conversation {i+1} (between {participants}) ---\n"
                    + "\n".join(lines)
                )
            transcripts_text = "\n\n".join(transcript_blocks)

            prompts.append({
                "system": _TEMPORAL_SYSTEM,
                "user": _TEMPORAL_USER.format(
                    agent_name=agent_name, transcripts=transcripts_text,
                ),
                "tags": {"phase": "d8_temporal_gen"},
            })

        log.info("Generating temporal questions for %d agents...", len(prompts))
        responses = self.llm.generate_batch(prompts)

        instances = []
        for aid, resp in zip(selected, responses):
            qa_items = _parse_json_array(resp)
            picked_ids = [s.session_id for s in selected_sessions[aid]]
            for qa in qa_items:
                if not isinstance(qa, dict) or "question" not in qa:
                    continue
                inst = EvalInstance(
                    instance_id=f"d8_{uuid.uuid4().hex[:12]}",
                    dimension=Dimension.D8_TEMPORAL,
                    query=qa["question"],
                    ground_truth={
                        "temporal_type": qa.get("temporal_type", ""),
                        "agent": aid,
                        "source_sessions": picked_ids,
                        "answer": qa.get("answer", ""),
                    },
                    difficulty="hard",
                    metadata={
                        "query_agent": aid,
                        "evidence_sessions": picked_ids,
                    },
                )
                instances.append(inst)

        return instances[:self.cfg.max_instances_per_dim]

    # ----- Technique 2b: Temporal lifecycle (sleep/wake boundaries) -----

    _SLEEP_BOUNDARY_TEMPLATES = [
        "What was the last thing {agent_name} discussed before going to sleep?",
        "What was {agent_name} talking about right before bed?",
        "Who was {agent_name} chatting with before they went to sleep?",
        "What was on {agent_name}'s mind just before sleeping?",
    ]

    _WAKE_BOUNDARY_TEMPLATES = [
        "Who did {agent_name} talk to first thing after waking up?",
        "What did {agent_name} do first after waking up?",
        "Who was the first person {agent_name} spoke with in the morning?",
        "What was {agent_name}'s first conversation after waking?",
    ]

    _ACTIVITY_BOUNDARY_TEMPLATES = [
        "What was {agent_name} doing right before the {group} meeting?",
        "What did {agent_name} do just before the {group} session?",
        "What was {agent_name} up to before heading to the {group} meeting?",
    ]

    def _generate_temporal_lifecycle(self, corpus: DialogueCorpus) -> List[EvalInstance]:
        """Generate D8 instances from activity log boundary transitions.

        Template-based (no LLM). Finds sleep/wake boundaries and
        pre-meeting activities.
        """
        if not corpus.activity_logs:
            return []

        group_map = {g.group_id: g for g in corpus.groups} if corpus.groups else {}
        instances: List[EvalInstance] = []

        for agent_id, entries in corpus.activity_logs.items():
            if not entries:
                continue
            # Sort by start_time
            sorted_entries = sorted(entries, key=lambda e: e.start_time)
            agent_name = self._first_name(agent_id)

            for i, entry in enumerate(sorted_entries):
                # Pre-sleep: last dialogue/group_meeting before a sleep entry
                if entry.activity_type == "sleep" and i > 0:
                    prev = sorted_entries[i - 1]
                    if prev.activity_type in ("dialogue", "group_meeting"):
                        answer = prev.description or prev.memory_text or "unknown"
                        if prev.participants:
                            participant_names = ", ".join(
                                self._first_name(p) for p in prev.participants
                            )
                            answer = f"Talking with {participant_names}: {answer}"

                        template = self._SLEEP_BOUNDARY_TEMPLATES[
                            self.rng.integers(len(self._SLEEP_BOUNDARY_TEMPLATES))
                        ]
                        query = template.format(agent_name=agent_name)
                        instances.append(EvalInstance(
                            instance_id=f"d8_slp_{uuid.uuid4().hex[:12]}",
                            dimension=Dimension.D8_TEMPORAL,
                            query=query,
                            ground_truth={
                                "temporal_type": "sleep_boundary",
                                "agent": agent_id,
                                "answer": answer,
                                "boundary": "pre_sleep",
                                "linked_session": prev.linked_session_id,
                            },
                            difficulty="hard",
                            metadata={
                                "query_agent": agent_id,
                                "evidence_sessions": [prev.linked_session_id] if prev.linked_session_id else [],
                            },
                        ))

                # Post-wake: first dialogue/group_meeting after a sleep entry
                if entry.activity_type == "sleep" and i + 1 < len(sorted_entries):
                    nxt = sorted_entries[i + 1]
                    if nxt.activity_type in ("dialogue", "group_meeting"):
                        if nxt.participants:
                            answer = ", ".join(self._first_name(p) for p in nxt.participants)
                        else:
                            answer = nxt.description or nxt.memory_text or "unknown"

                        template = self._WAKE_BOUNDARY_TEMPLATES[
                            self.rng.integers(len(self._WAKE_BOUNDARY_TEMPLATES))
                        ]
                        query = template.format(agent_name=agent_name)
                        instances.append(EvalInstance(
                            instance_id=f"d8_wak_{uuid.uuid4().hex[:12]}",
                            dimension=Dimension.D8_TEMPORAL,
                            query=query,
                            ground_truth={
                                "temporal_type": "wake_boundary",
                                "agent": agent_id,
                                "answer": answer,
                                "boundary": "post_wake",
                                "linked_session": nxt.linked_session_id,
                            },
                            difficulty="hard",
                            metadata={
                                "query_agent": agent_id,
                                "evidence_sessions": [nxt.linked_session_id] if nxt.linked_session_id else [],
                            },
                        ))

                # Pre-meeting: entry immediately before a group_meeting
                if entry.activity_type == "group_meeting" and i > 0:
                    prev = sorted_entries[i - 1]
                    if prev.activity_type != "sleep":  # skip trivial "was sleeping"
                        group_name = group_map.get(entry.group_id, None)
                        group_label = group_name.name if group_name else entry.group_id
                        answer = prev.description or prev.memory_text or prev.activity_type

                        template = self._ACTIVITY_BOUNDARY_TEMPLATES[
                            self.rng.integers(len(self._ACTIVITY_BOUNDARY_TEMPLATES))
                        ]
                        query = template.format(agent_name=agent_name, group=group_label)
                        instances.append(EvalInstance(
                            instance_id=f"d8_mtg_{uuid.uuid4().hex[:12]}",
                            dimension=Dimension.D8_TEMPORAL,
                            query=query,
                            ground_truth={
                                "temporal_type": "activity_boundary",
                                "agent": agent_id,
                                "answer": answer,
                                "boundary": "pre_meeting",
                                "group": group_label,
                            },
                            difficulty="hard",
                            metadata={
                                "query_agent": agent_id,
                                "evidence_sessions": [],
                            },
                        ))

        return instances

    @staticmethod
    def _first_name(agent_slug: str) -> str:
        """Convert 'eleanor_vance' → 'Eleanor'."""
        if is_assistant(agent_slug):
            return "the assistant"
        return agent_slug.split("_")[0].title()

    # ----- Technique 4: Perspective -----

    def _generate_perspective(self, corpus: DialogueCorpus) -> List[EvalInstance]:
        """Generate perspective-scoped D7 instances appended with metadata flag."""
        agent_sessions: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        for session in corpus.sessions:
            # Use natural names in transcripts shown to LLM
            transcript = "\n".join(
                f"  {self._first_name(t.speaker_id)}: {t.text[:100]}"
                for t in session.turns[:8]
            )
            participant_names = ", ".join(
                self._first_name(p) for p in session.participants
            )
            summary = {
                "session_id": session.session_id,
                "participants": participant_names,
                "transcript": transcript,
            }
            for pid in session.participants:
                agent_sessions[pid].append(summary)

        eligible = [aid for aid, sl in agent_sessions.items() if len(sl) >= 2 and not is_assistant(aid)]
        if not eligible:
            return []

        max_agents = min(len(eligible), self.cfg.max_instances_per_dim // 2 + 1)
        selected = list(self.rng.choice(eligible, size=max_agents, replace=False))

        prompts = []
        for aid in selected:
            agent_name = self._first_name(aid)
            sessions_text = "\n".join(
                f"Conversation {i+1} ({s['participants']}):\n{s['transcript']}"
                for i, s in enumerate(agent_sessions[aid][:5])
            )
            prompts.append({
                "system": _PERSPECTIVE_SYSTEM,
                "user": _PERSPECTIVE_USER.format(
                    agent_name=agent_name, agent_id=aid, sessions=sessions_text,
                ),
                "tags": {"phase": "d7_perspective_gen"},
            })

        log.info("Generating perspective questions for %d agents...", len(prompts))
        responses = self.llm.generate_batch(prompts)

        instances = []
        for aid, resp in zip(selected, responses):
            qa_items = _parse_json_array(resp)
            for qa in qa_items:
                if not isinstance(qa, dict) or "question" not in qa:
                    continue
                inst = EvalInstance(
                    instance_id=f"d7p_{uuid.uuid4().hex[:12]}",
                    dimension=Dimension.D7_QA,
                    query=qa["question"],
                    ground_truth={
                        "required_agent": aid,
                        "answer": qa.get("answer", ""),
                    },
                    difficulty="hard",
                    metadata={
                        "perspective_scoped": True,
                        "query_agent": aid,
                        "evidence_sessions": [
                            s["session_id"] for s in agent_sessions[aid]
                        ],
                    },
                )
                instances.append(inst)

        return instances[:self.cfg.max_instances_per_dim]

    # ----- Metadata enrichment -----

    @staticmethod
    def _enrich_metadata(
        instances: Dict[Dimension, List[EvalInstance]],
        corpus: DialogueCorpus,
    ) -> None:
        """Ensure every instance has query_agent and question_feature."""
        # Build lookup: session_id -> Session for resolving agents
        session_map: Dict[str, Session] = {
            s.session_id: s for s in corpus.sessions
        }

        for dim, inst_list in instances.items():
            for inst in inst_list:
                # --- query_agent ---
                # D4 sets it where an asker exists; inferring one would give
                # anonymous-querier probes a known asker.
                if dim != Dimension.D4_PERMISSION and not inst.metadata.get("query_agent"):
                    inst.metadata["query_agent"] = _infer_query_agent(
                        inst, session_map,
                    )

                # --- question_feature ---
                inst.metadata["question_feature"] = _question_feature(inst)

    # ----- Technique 5: Counterfactual -----

    def _generate_counterfactual(self, corpus: DialogueCorpus) -> List[EvalInstance]:
        """Generate D10 counterfactual instances via LLM."""
        # Collect substantial turns (>=10 words, not injected conflicts)
        candidates = []
        for session in corpus.sessions:
            for turn in session.turns:
                if (len(turn.text.split()) >= 10
                        and not turn.metadata.get("injected_conflict")):
                    candidates.append((session, turn))

        if not candidates:
            return []

        max_picks = min(len(candidates), self.cfg.max_instances_per_dim)
        indices = self.rng.choice(len(candidates), size=max_picks, replace=False)
        selected = [candidates[i] for i in indices]

        prompts = []
        for session, turn in selected:
            prompts.append({
                "system": _COUNTERFACTUAL_SYSTEM,
                "user": _COUNTERFACTUAL_USER.format(
                    speaker=self._first_name(turn.speaker_id),
                    text=turn.text[:300],
                ),
                "tags": {"phase": "d10_counterfactual_gen"},
            })

        log.info("Generating counterfactual probes for %d turns...", len(prompts))
        responses = self.llm.generate_batch(prompts)

        instances = []
        for (session, turn), resp in zip(selected, responses):
            obj = _parse_json_object(resp)
            if not obj or "question" not in obj:
                continue

            inst = EvalInstance(
                instance_id=f"d10_{uuid.uuid4().hex[:12]}",
                dimension=Dimension.D10_COUNTERFACTUAL,
                query=obj["question"],
                ground_truth={
                    "expected_response": "correct_premise",
                    "altered_detail": obj.get("altered_detail", ""),
                    "source_session": session.session_id,
                    "source_speaker": turn.speaker_id,
                    "source_turn": turn.turn_id,
                    "real_fact": obj.get("real_fact", ""),
                },
                difficulty="hard",
                metadata={
                    "query_agent": session.participants[0],
                    "evidence_sessions": [session.session_id],
                },
            )
            instances.append(inst)

        return instances[:self.cfg.max_instances_per_dim]
