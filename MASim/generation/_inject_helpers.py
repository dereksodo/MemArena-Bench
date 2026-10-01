"""Shared helpers for permission and privacy injectors."""

from __future__ import annotations

import uuid
from typing import Any, List, Optional, Tuple

import numpy as np

from MASim.core.schema import DialogueTurn, Session
from MASim.prompts import INJECT_REACT_SYSTEM as _REACT_SYSTEM
from MASim.prompts import INJECT_REACT_USER as _REACT_USER
from MASim.utils.logging import get_logger

log = get_logger(__name__)


def insert_and_react(
    session: Session,
    injected_turn: DialogueTurn,
    llm_client: Any,
    rng: np.random.Generator,
    dry_run: bool = False,
) -> None:
    """Insert an injected turn at a random mid-conversation position and
    generate a 1-turn reaction from the listener.

    Modifies *session.turns* in-place.
    """
    pos = _pick_insert_pos(session, rng)
    session.turns.insert(pos, injected_turn)

    # Generate a reaction from the listener
    speaker = injected_turn.speaker_id
    listener = injected_turn.listener_id

    if dry_run:
        react_text = f"[DRY RUN] {listener} reacts to injected turn"
    else:
        system = _REACT_SYSTEM.format(listener=listener)
        user = _REACT_USER.format(
            listener=listener,
            speaker=speaker,
            injected_text=injected_turn.text[:500],
        )
        react_text = llm_client.generate(system, user, tags={"phase": "inject_react"})

    react_turn = DialogueTurn(
        turn_id=f"{session.session_id}_react_{uuid.uuid4().hex[:8]}",
        session_id=session.session_id,
        speaker_id=listener,
        listener_id=speaker,
        text=react_text,
        timestamp=injected_turn.timestamp + 0.00001,
        metadata={"injected_reaction": True},
    )
    session.turns.insert(pos + 1, react_turn)


def batch_insert_and_react(
    items: List[Tuple[Session, DialogueTurn]],
    llm_client: Any,
    rng: np.random.Generator,
    dry_run: bool = False,
) -> None:
    """Batch version: insert all injected turns and generate reactions in one LLM batch.

    Each item is (session, injected_turn). Modifies sessions in-place.
    """
    if not items:
        return

    # Phase 1: insert all injected turns and collect reaction prompts
    insert_positions: List[int] = []
    prompts: List[dict] = []
    for session, injected_turn in items:
        pos = _pick_insert_pos(session, rng)
        session.turns.insert(pos, injected_turn)
        insert_positions.append(pos)

        speaker = injected_turn.speaker_id
        listener = injected_turn.listener_id
        prompts.append({
            "system": _REACT_SYSTEM.format(listener=listener),
            "user": _REACT_USER.format(
                listener=listener,
                speaker=speaker,
                injected_text=injected_turn.text[:500],
            ),
            "tags": {"phase": "inject_react"},
        })

    # Phase 2: batch-generate all reactions
    if dry_run:
        responses = [
            f"[DRY RUN] {turn.listener_id} reacts to injected turn"
            for _, turn in items
        ]
    else:
        log.info("Batch-generating %d inject reactions via LLM...", len(prompts))
        responses = llm_client.generate_batch(prompts)

    # Phase 3: insert reaction turns
    for (session, injected_turn), pos, react_text in zip(items, insert_positions, responses):
        react_turn = DialogueTurn(
            turn_id=f"{session.session_id}_react_{uuid.uuid4().hex[:8]}",
            session_id=session.session_id,
            speaker_id=injected_turn.listener_id,
            listener_id=injected_turn.speaker_id,
            text=react_text,
            timestamp=injected_turn.timestamp + 0.00001,
            metadata={"injected_reaction": True},
        )
        session.turns.insert(pos + 1, react_turn)


def _pick_insert_pos(session: Session, rng: np.random.Generator) -> int:
    """Pick a random mid-conversation position for injection."""
    n = len(session.turns)
    lo = min(2, n)
    hi = max(lo + 1, n - 1)
    return int(rng.integers(lo, hi))
