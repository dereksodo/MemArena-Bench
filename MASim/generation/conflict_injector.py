"""Inject contradictions into dialogue corpus with ground truth tracking.

Creates controlled information conflicts where different agents tell the
ego-user contradictory facts, enabling D1 (conflict preservation) evaluation.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from MASim.core.schema import ConflictGT, DialogueCorpus, DialogueTurn, Session
from MASim.generation._inject_helpers import batch_insert_and_react
from MASim.ground_truth.json_parser import clean_llm_json_text, parse_json_object
from MASim.prompts import CONFLICT_SYSTEM as CONFLICT_SYSTEM_PROMPT
from MASim.prompts import CONFLICT_USER as CONFLICT_USER_TEMPLATE
from MASim.utils.logging import get_logger

log = get_logger(__name__)


def _parse_conflict_response(response: str) -> dict:
    """Parse the LLM conflict response (expected JSON with statement + details).

    Falls back gracefully: if JSON parsing fails, treats the entire response
    as the statement with empty detail fields.
    """
    try:
        data = parse_json_object(response)
        return {
            "statement": str(data.get("statement", response)),
            "original_detail": str(data.get("original_detail", "")),
            "changed_detail": str(data.get("changed_detail", "")),
        }
    except (ValueError, TypeError):
        pass

    # Fallback: raw text is the statement, no extracted details
    raw = clean_llm_json_text(response)
    return {"statement": raw, "original_detail": "", "changed_detail": ""}


@dataclass
class ConflictInjectorConfig:
    """Configuration for conflict injection."""
    inject_prob: float = 0.1     # probability of injecting per eligible session
    max_injections: Optional[int] = None  # None = no cap
    min_session_gap: int = 1     # minimum sessions between original and conflict
    seed: int = 42


class ConflictInjector:
    """Inject contradictory facts into dialogue corpus."""

    def __init__(self, llm_client: Any, cfg: ConflictInjectorConfig):
        self.llm_client = llm_client
        self.cfg = cfg
        self._rng = np.random.default_rng(cfg.seed)

    def inject(
        self,
        corpus: DialogueCorpus,
        dry_run: bool = False,
    ) -> Tuple[DialogueCorpus, List[ConflictGT]]:
        """Inject conflicts into a dialogue corpus.

        Returns the modified corpus and a list of conflict ground truths.
        """
        ground_truths: List[ConflictGT] = []
        sessions = list(corpus.sessions)

        if len(sessions) < self.cfg.min_session_gap + 1:
            log.warning("Not enough sessions for conflict injection")
            return corpus, ground_truths

        # Find candidate turns (substantial statements, not greetings)
        candidates = self._find_candidates(sessions)
        self._rng.shuffle(candidates)

        n_inject = min(int(len(candidates) * self.cfg.inject_prob), len(candidates))
        if self.cfg.max_injections is not None:
            n_inject = min(n_inject, self.cfg.max_injections)

        log.info("Injecting up to %d conflicts from %d candidates", n_inject, len(candidates))

        # Generate contradictory versions — iterate all candidates until
        # we collect enough with valid targets
        tasks = []
        for orig_turn, orig_session_idx in candidates:
            if len(tasks) >= n_inject:
                break

            # Pick a later session with a different partner
            target_sessions = [
                (i, s) for i, s in enumerate(sessions)
                if i > orig_session_idx + self.cfg.min_session_gap
                and orig_turn.speaker_id not in s.participants
            ]
            if not target_sessions:
                continue

            target_idx, target_session = self._rng.choice(
                [(i, s) for i, s in target_sessions]
            )
            new_speaker = self._rng.choice(target_session.participants)

            prompt = CONFLICT_USER_TEMPLATE.format(
                speaker=orig_turn.speaker_id,
                original=orig_turn.text,
                new_speaker=new_speaker,
            )
            tasks.append({
                "system": CONFLICT_SYSTEM_PROMPT,
                "user": prompt,
                "meta": {
                    "orig_turn": orig_turn,
                    "orig_session_idx": orig_session_idx,
                    "target_session_idx": target_idx,
                    "new_speaker": new_speaker,
                },
            })

        if not tasks:
            return corpus, ground_truths

        if dry_run:
            responses = [
                json.dumps({
                    "statement": f"[CONFLICT] contradicts: {t['meta']['orig_turn'].text[:50]}",
                    "original_detail": "original detail",
                    "changed_detail": "changed detail",
                })
                for t in tasks
            ]
        else:
            prompts = [{"system": t["system"], "user": t["user"], "tags": {"phase": "conflict_gen"}} for t in tasks]
            responses = self.llm_client.generate_batch(prompts)

        # Build all conflict turns, then batch-generate reactions
        react_items = []
        conflict_meta = []
        for task, response in zip(tasks, responses):
            meta = task["meta"]
            target_idx = meta["target_session_idx"]
            target_session = sessions[target_idx]

            # Parse JSON response to extract statement and detail fields
            parsed = _parse_conflict_response(response)
            statement_text = parsed["statement"]
            original_detail = parsed["original_detail"]
            changed_detail = parsed["changed_detail"]

            listener_id = [p for p in target_session.participants if p != meta["new_speaker"]][0]
            conflict_turn = DialogueTurn(
                turn_id=f"{target_session.session_id}_conflict_{uuid.uuid4().hex[:8]}",
                session_id=target_session.session_id,
                speaker_id=meta["new_speaker"],
                listener_id=listener_id,
                text=statement_text,
                timestamp=target_session.end_time - 0.0001,
                metadata={"injected_conflict": True, "original_turn_id": meta["orig_turn"].turn_id},
            )
            react_items.append((target_session, conflict_turn))
            conflict_meta.append((meta, target_session, conflict_turn, statement_text,
                                  original_detail, changed_detail))

        batch_insert_and_react(react_items, self.llm_client, self._rng, dry_run=dry_run)

        # Record ground truth
        for meta, target_session, conflict_turn, stmt, orig_det, chg_det in conflict_meta:
            gt = ConflictGT(
                fact=meta["orig_turn"].text,
                contradictory_fact=stmt,
                source_agent=meta["orig_turn"].speaker_id,
                conflicting_agent=meta["new_speaker"],
                original_session=sessions[meta["orig_session_idx"]].session_id,
                conflicting_session=target_session.session_id,
                original_timestamp=meta["orig_turn"].timestamp,
                conflicting_timestamp=conflict_turn.timestamp,
                original_detail=orig_det,
                changed_detail=chg_det,
            )
            ground_truths.append(gt)

        log.info("Injected %d conflicts", len(ground_truths))
        return corpus, ground_truths

    def _find_candidates(
        self, sessions: List[Session]
    ) -> List[Tuple[DialogueTurn, int]]:
        """Find turns suitable for contradiction (substantive statements)."""
        candidates = []
        for sess_idx, session in enumerate(sessions):
            # Skip person-agent sessions (activity/reflection/probe) —
            # injecting conflicts into probe sessions creates gibberish turns
            stype = session.metadata.get("session_type", "")
            if stype.startswith("person_agent"):
                continue
            for turn in session.turns:
                # Skip very short turns (greetings, etc.)
                if len(turn.text.split()) < 8:
                    continue
                # Skip already-injected turns
                if turn.metadata.get("injected_conflict"):
                    continue
                candidates.append((turn, sess_idx))
        return candidates
