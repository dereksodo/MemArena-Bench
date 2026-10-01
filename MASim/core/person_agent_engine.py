"""PersonAgentEngine: person-to-AI-assistant dialogue generation.

Two sub-types:
  1. Activity narration — person tells the assistant about daily life.
  2. Memory reflection — person reviews a past conversation with the assistant.
"""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

from MASim.core.agent import EntityAgent
from MASim.core.dialogue_engine import (
    _build_modality_prefix,
    _is_conversation_ending,
    _parse_turn_output,
    batch_extract_session_knowledge,
)
from MASim.core.schema import DialogueTurn, Session, assistant_id_for, is_assistant
from MASim.prompts import ASSISTANT_SYSTEM as _ASSISTANT_SYSTEM_PROMPT
from MASim.utils.logging import get_logger

log = get_logger(__name__)

_assistant_id = assistant_id_for  # local alias

# Legacy constant — kept for backward compatibility in other modules.
ASSISTANT_ID = "__assistant__"


class PersonAgentEngine:
    """Generate person-agent dialogues (activity narration, memory reflection, memory probe)."""

    def __init__(
        self,
        llm_client: Any,
        max_turns: int = 30,
        min_turns: int = 15,
        probe_max_turns: int = 10,
        probe_min_turns: int = 4,
        no_token_limit: bool = False,
    ) -> None:
        self.llm_client = llm_client
        self.max_turns = max_turns
        self.min_turns = min_turns
        self.probe_max_turns = probe_max_turns
        self.probe_min_turns = probe_min_turns
        self.no_token_limit = no_token_limit

    # ------------------------------------------------------------------
    # Activity narration
    # ------------------------------------------------------------------

    def simulate_activity_narration(
        self,
        agent: EntityAgent,
        topic: Dict[str, Any],
        timestamp: float,
        dry_run: bool = False,
        session_seed: Optional[int] = None,
    ) -> Session:
        """Person tells the AI assistant about a daily-life topic.

        The assistant asks follow-up questions; the person elaborates.
        """
        if session_seed is not None:
            import numpy as _np
            agent._rng = _np.random.default_rng(session_seed ^ 0xB1B1B1B1)

        _aid = _assistant_id(agent.agent_id)
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        turns: List[DialogueTurn] = []
        current_time = timestamp
        topic_prompt = topic.get("prompt", "Tell me about your day.")

        # First turn: assistant opens with the topic prompt
        opener_turn = DialogueTurn(
            turn_id=f"{session_id}_t000",
            session_id=session_id,
            speaker_id=_aid,
            listener_id=agent.agent_id,
            text=topic_prompt if not dry_run else f"[DRY RUN] Assistant: {topic_prompt}",
            timestamp=current_time,
            metadata={"turn_num": 0, "person_agent": True},
        )
        turns.append(opener_turn)
        agent.receive_turn(opener_turn)
        current_time += 0.001

        for turn_num in range(1, self.max_turns):
            is_person_turn = (turn_num % 2 == 1)  # odd = person, even = assistant

            if is_person_turn:
                # Person speaks using full EntityAgent persona
                context_prefix = (
                    "This is a text conversation with an AI assistant.\n"
                    f"Topic: {topic_prompt}\n"
                )
                system_prompt, user_prompt = agent.build_generation_context(
                    partner_name="AI Assistant",
                    partner_agent_id=_aid,
                    event=None,
                    recent_turns=turns,
                    context_prefix=context_prefix,
                    modality="text_message",
                    no_token_limit=self.no_token_limit,
                )

                if dry_run:
                    thought, content = "", f"[DRY RUN] {agent.persona.name} turn {turn_num}"
                else:
                    raw = self.llm_client.generate(
                        system_prompt, user_prompt,
                        tags={"phase": "pa_person_turn", "step": turn_num},
                    )
                    thought, content = _parse_turn_output(raw)

                turn = DialogueTurn(
                    turn_id=f"{session_id}_t{turn_num:03d}",
                    session_id=session_id,
                    speaker_id=agent.agent_id,
                    listener_id=_aid,
                    text=content,
                    timestamp=current_time,
                    metadata={"turn_num": turn_num, "thought": thought, "person_agent": True},
                )
            else:
                # Assistant responds with follow-up questions
                transcript = "\n".join(
                    f"{'AI Assistant' if is_assistant(t.speaker_id) else agent.persona.name}: {t.text}"
                    for t in turns[-8:]
                )
                assistant_prompt = (
                    f"Continue this conversation. The user is talking about: {topic_prompt}\n\n"
                    f"{transcript}\n\nAI Assistant:"
                )

                if dry_run:
                    content = f"[DRY RUN] Assistant turn {turn_num}"
                else:
                    raw = self.llm_client.generate(
                        _ASSISTANT_SYSTEM_PROMPT, assistant_prompt,
                        tags={"phase": "pa_assistant_turn", "step": turn_num},
                    )
                    _, content = _parse_turn_output(raw)

                turn = DialogueTurn(
                    turn_id=f"{session_id}_t{turn_num:03d}",
                    session_id=session_id,
                    speaker_id=_aid,
                    listener_id=agent.agent_id,
                    text=content,
                    timestamp=current_time,
                    metadata={"turn_num": turn_num, "person_agent": True},
                )

            turns.append(turn)
            agent.receive_turn(turn)
            current_time += 0.001

            if turn_num >= self.min_turns - 1 and _is_conversation_ending(content):
                break

        # NER extraction (knowledge for the human agent only)
        if not dry_run:
            batch_extract_session_knowledge(self.llm_client, turns)

        session = Session(
            session_id=session_id,
            participants=[agent.agent_id, _aid],
            turns=turns,
            start_time=timestamp,
            end_time=current_time,
            modality="text_message",
            metadata={
                "session_type": "person_agent_activity",
                "topic_id": topic.get("id", ""),
                "topic_prompt": topic_prompt,
            },
        )

        # Knowledge update only for the human agent (after NER)
        agent.knowledge.update_from_session(session, agent.agent_id)

        # Mood update for the human
        if not dry_run:
            self._update_agent_mood(agent, session)

        log.debug(
            "Person-agent activity session %s: %s, %d turns, topic=%s",
            session_id, agent.persona.name, len(turns), topic.get("id", ""),
        )
        return session

    # ------------------------------------------------------------------
    # Memory reflection
    # ------------------------------------------------------------------

    def simulate_memory_reflection(
        self,
        agent: EntityAgent,
        target_session: Session,
        timestamp: float,
        dry_run: bool = False,
        session_seed: Optional[int] = None,
    ) -> Session:
        """Person reviews a past conversation with the AI assistant.

        The assistant summarises highlights from target_session and the
        person reflects, corrects, or elaborates.
        """
        if session_seed is not None:
            import numpy as _np
            agent._rng = _np.random.default_rng(session_seed ^ 0xC2C2C2C2)

        _aid = _assistant_id(agent.agent_id)
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        turns: List[DialogueTurn] = []
        current_time = timestamp

        # Build a brief summary of the target session for the assistant
        target_summary = self._summarise_session_for_reflection(target_session, agent.agent_id)

        # First turn: assistant brings up the past conversation
        opener_text = (
            f"I'd like to revisit a conversation you had earlier. "
            f"Here's what I remember: {target_summary} "
            f"What stands out to you about that?"
        )
        if dry_run:
            opener_text = f"[DRY RUN] Assistant: Let's revisit session {target_session.session_id}"

        opener_turn = DialogueTurn(
            turn_id=f"{session_id}_t000",
            session_id=session_id,
            speaker_id=_aid,
            listener_id=agent.agent_id,
            text=opener_text,
            timestamp=current_time,
            metadata={"turn_num": 0, "person_agent": True},
        )
        turns.append(opener_turn)
        agent.receive_turn(opener_turn)
        current_time += 0.001

        for turn_num in range(1, self.max_turns):
            is_person_turn = (turn_num % 2 == 1)

            if is_person_turn:
                context_prefix = (
                    "This is a text conversation with an AI assistant.\n"
                    "You're reflecting on a past conversation you had.\n"
                    f"Context: {target_summary}\n"
                )
                system_prompt, user_prompt = agent.build_generation_context(
                    partner_name="AI Assistant",
                    partner_agent_id=_aid,
                    event=None,
                    recent_turns=turns,
                    context_prefix=context_prefix,
                    modality="text_message",
                    no_token_limit=self.no_token_limit,
                )

                if dry_run:
                    thought, content = "", f"[DRY RUN] {agent.persona.name} reflection turn {turn_num}"
                else:
                    raw = self.llm_client.generate(
                        system_prompt, user_prompt,
                        tags={"phase": "pa_person_turn", "step": turn_num},
                    )
                    thought, content = _parse_turn_output(raw)

                turn = DialogueTurn(
                    turn_id=f"{session_id}_t{turn_num:03d}",
                    session_id=session_id,
                    speaker_id=agent.agent_id,
                    listener_id=_aid,
                    text=content,
                    timestamp=current_time,
                    metadata={"turn_num": turn_num, "thought": thought, "person_agent": True},
                )
            else:
                transcript = "\n".join(
                    f"{'AI Assistant' if is_assistant(t.speaker_id) else agent.persona.name}: {t.text}"
                    for t in turns[-8:]
                )
                assistant_prompt = (
                    f"Continue this reflection conversation. You're helping the user "
                    f"think back on a past conversation.\n"
                    f"Summary of the past conversation: {target_summary}\n\n"
                    f"{transcript}\n\nAI Assistant:"
                )

                if dry_run:
                    content = f"[DRY RUN] Assistant reflection turn {turn_num}"
                else:
                    raw = self.llm_client.generate(
                        _ASSISTANT_SYSTEM_PROMPT, assistant_prompt,
                        tags={"phase": "pa_assistant_turn", "step": turn_num},
                    )
                    _, content = _parse_turn_output(raw)

                turn = DialogueTurn(
                    turn_id=f"{session_id}_t{turn_num:03d}",
                    session_id=session_id,
                    speaker_id=_aid,
                    listener_id=agent.agent_id,
                    text=content,
                    timestamp=current_time,
                    metadata={"turn_num": turn_num, "person_agent": True},
                )

            turns.append(turn)
            agent.receive_turn(turn)
            current_time += 0.001

            if turn_num >= self.min_turns - 1 and _is_conversation_ending(content):
                break

        if not dry_run:
            batch_extract_session_knowledge(self.llm_client, turns)

        session = Session(
            session_id=session_id,
            participants=[agent.agent_id, _aid],
            turns=turns,
            start_time=timestamp,
            end_time=current_time,
            modality="text_message",
            metadata={
                "session_type": "person_agent_reflection",
                "target_session_id": target_session.session_id,
            },
        )

        agent.knowledge.update_from_session(session, agent.agent_id)

        if not dry_run:
            self._update_agent_mood(agent, session)

        log.debug(
            "Person-agent reflection session %s: %s, %d turns, target=%s",
            session_id, agent.persona.name, len(turns), target_session.session_id,
        )
        return session

    # ------------------------------------------------------------------
    # Step-wise batched generation
    # ------------------------------------------------------------------

    def simulate_pa_sessions_stepped(
        self,
        configs: List[Dict[str, Any]],
        dry_run: bool = False,
        call_tags: Optional[Dict[str, Any]] = None,
    ) -> List[Session]:
        """Run all PA sessions with step-level batching.

        Instead of running each session independently in a thread, this
        processes all sessions lock-step: at each turn number, it gathers
        prompts from every active session, fires them all via
        generate_batch(), then parses the responses.  This maximises GPU
        utilisation by sending N requests per batch instead of 1-at-a-time.

        Args:
            configs: list of dicts, each with keys:
                agent, session_type, topic, target_session, probe_spec,
                timestamp, session_seed

        Returns:
            List of Session objects (call batch_finalize_pa_sessions() after).
        """
        if not configs:
            return []

        import time as _time

        total = len(configs)
        _wall_start = _time.monotonic()

        # ---- Phase 0: Initialise per-session state ----
        class _Slot:
            """Mutable state for one in-flight PA session."""
            __slots__ = (
                "idx", "agent", "stype", "aid", "session_id", "topic_prompt",
                "target_summary", "probe_spec", "effective_max", "effective_min",
                "turns", "current_time", "finished",
            )

        slots: List[_Slot] = []
        for i, cfg in enumerate(configs):
            s = _Slot()
            s.idx = i
            s.agent = cfg["agent"]
            s.stype = cfg["session_type"]
            s.aid = _assistant_id(s.agent.agent_id)
            s.session_id = f"sess_{uuid.uuid4().hex[:12]}"
            s.finished = False

            if s.stype == "activity":
                s.topic_prompt = cfg["topic"].get("prompt", "Tell me about your day.")
                s.target_summary = None
                s.probe_spec = None
            elif s.stype == "reflection":
                s.topic_prompt = None
                s.target_summary = self._summarise_session_for_reflection(
                    cfg["target_session"], s.agent.agent_id,
                )
                s.probe_spec = None
            else:  # probe
                s.topic_prompt = None
                s.target_summary = None
                s.probe_spec = cfg["probe_spec"]

            s.effective_max = self.probe_max_turns if s.stype == "probe" else self.max_turns
            s.effective_min = self.probe_min_turns if s.stype == "probe" else self.min_turns

            # Opener turn (no LLM call)
            if s.stype == "activity":
                opener_text = s.topic_prompt if not dry_run else f"[DRY RUN] Assistant: {s.topic_prompt}"
            elif s.stype == "reflection":
                if dry_run:
                    opener_text = f"[DRY RUN] Assistant: Let's revisit session {cfg['target_session'].session_id}"
                else:
                    opener_text = (
                        f"I'd like to revisit a conversation you had earlier. "
                        f"Here's what I remember: {s.target_summary} "
                        f"What stands out to you about that?"
                    )
            else:  # probe
                if dry_run:
                    opener_text = f"[DRY RUN] Assistant: Memory probe — {s.probe_spec.probe_type}"
                else:
                    opener_text = s.probe_spec.question_seed

            opener_turn = DialogueTurn(
                turn_id=f"{s.session_id}_t000",
                session_id=s.session_id,
                speaker_id=s.aid,
                listener_id=s.agent.agent_id,
                text=opener_text,
                timestamp=cfg["timestamp"],
                metadata={"turn_num": 0, "person_agent": True},
            )
            s.turns = [opener_turn]
            s.current_time = cfg["timestamp"] + 0.001
            slots.append(s)

        global_max_turns = max(s.effective_max for s in slots)
        total_turns_generated = 0

        log.info(
            "PA batch pipeline: %d sessions, max_steps=%d, concurrency=%d",
            total, global_max_turns, self.llm_client.cfg.concurrency,
        )

        # ---- Phase 1: Step-level batching ----
        for step in range(1, global_max_turns):
            active = [s for s in slots if not s.finished and step < s.effective_max]
            if not active:
                break

            is_person_turn = (step % 2 == 1)

            if dry_run:
                for s in active:
                    if is_person_turn:
                        content = f"[DRY RUN] {s.agent.persona.name} turn {step}"
                        spk, lis = s.agent.agent_id, s.aid
                    else:
                        content = f"[DRY RUN] Assistant turn {step}"
                        spk, lis = s.aid, s.agent.agent_id
                    t = DialogueTurn(
                        turn_id=f"{s.session_id}_t{step:03d}",
                        session_id=s.session_id,
                        speaker_id=spk, listener_id=lis,
                        text=content, timestamp=s.current_time,
                        metadata={"turn_num": step, "thought": "", "person_agent": True},
                    )
                    s.turns.append(t)
                    s.current_time += 0.001
                    total_turns_generated += 1
                    if step >= s.effective_min - 1 and _is_conversation_ending(content):
                        s.finished = True
                continue

            # Build batch prompts
            prompts: List[Dict[str, Any]] = []
            batch_slots: List[_Slot] = []

            for s in active:
                if is_person_turn:
                    if s.stype == "activity":
                        ctx = (
                            "This is a text conversation with an AI assistant.\n"
                            f"Topic: {s.topic_prompt}\n"
                        )
                    elif s.stype == "reflection":
                        ctx = (
                            "This is a text conversation with an AI assistant.\n"
                            "You're reflecting on a past conversation you had.\n"
                            f"Context: {s.target_summary}\n"
                        )
                    else:  # probe
                        fact_snippets = " | ".join(
                            f.content for f in s.probe_spec.source_facts[:3]
                        )
                        ctx = (
                            "This is a text conversation with an AI assistant.\n"
                            "The assistant is asking you about things you remember.\n"
                            "You are recalling from memory. You may not remember every detail "
                            "perfectly. Answer naturally — if you're unsure, say so. "
                            "Don't make things up, but it's okay to be approximate.\n"
                            f"Some things you vaguely recall: {fact_snippets}\n"
                        )
                    sys_p, usr_p = s.agent.build_generation_context(
                        partner_name="AI Assistant",
                        partner_agent_id=s.aid,
                        event=None,
                        recent_turns=s.turns,
                        context_prefix=ctx,
                        modality="text_message",
                        no_token_limit=self.no_token_limit,
                    )
                    _tags: Dict[str, Any] = {"phase": "pa_person_turn", "step": step, "session_type": s.stype}
                    if call_tags:
                        _tags.update(call_tags)
                    prompts.append({"system": sys_p, "user": usr_p, "tags": _tags})
                else:
                    transcript = "\n".join(
                        f"{'AI Assistant' if is_assistant(t.speaker_id) else s.agent.persona.name}: {t.text}"
                        for t in s.turns[-8:]
                    )
                    if s.stype == "activity":
                        usr_p = (
                            f"Continue this conversation. The user is talking about: {s.topic_prompt}\n\n"
                            f"{transcript}\n\nAI Assistant:"
                        )
                    elif s.stype == "reflection":
                        usr_p = (
                            f"Continue this reflection conversation. You're helping the user "
                            f"think back on a past conversation.\n"
                            f"Summary of the past conversation: {s.target_summary}\n\n"
                            f"{transcript}\n\nAI Assistant:"
                        )
                    else:  # probe
                        usr_p = self._build_probe_assistant_prompt(
                            s.probe_spec, s.agent.persona.name, transcript, step,
                        )
                    _tags = {"phase": "pa_assistant_turn", "step": step, "session_type": s.stype}
                    if call_tags:
                        _tags.update(call_tags)
                    prompts.append({"system": _ASSISTANT_SYSTEM_PROMPT, "user": usr_p, "tags": _tags})
                batch_slots.append(s)

            # Fire batch
            responses = self.llm_client.generate_batch(prompts)

            # Parse responses and append turns
            for s, raw in zip(batch_slots, responses):
                if is_person_turn:
                    thought, content = _parse_turn_output(raw)
                    spk, lis = s.agent.agent_id, s.aid
                else:
                    _, content = _parse_turn_output(raw)
                    thought = ""
                    spk, lis = s.aid, s.agent.agent_id

                t = DialogueTurn(
                    turn_id=f"{s.session_id}_t{step:03d}",
                    session_id=s.session_id,
                    speaker_id=spk, listener_id=lis,
                    text=content, timestamp=s.current_time,
                    metadata={"turn_num": step, "thought": thought, "person_agent": True},
                )
                s.turns.append(t)
                s.current_time += 0.001
                total_turns_generated += 1
                if step >= s.effective_min - 1 and _is_conversation_ending(content):
                    s.finished = True

            n_done = sum(1 for s in slots if s.finished)
            elapsed = _time.monotonic() - _wall_start
            tps = total_turns_generated / elapsed if elapsed > 0 else 0
            log.info(
                "PA step %d: batch=%d, done=%d/%d (%.0f%%) | "
                "%.0f turns | %.1f turns/s | elapsed: %.0fs",
                step, len(batch_slots), n_done, total, 100 * n_done / total,
                total_turns_generated, tps, elapsed,
            )

        # ---- Phase 2: Build Session objects ----
        sessions: List[Session] = []
        for i, s in enumerate(slots):
            cfg = configs[i]
            if s.stype == "activity":
                meta = {
                    "session_type": "person_agent_activity",
                    "topic_id": cfg.get("topic", {}).get("id", ""),
                    "topic_prompt": s.topic_prompt,
                }
            elif s.stype == "reflection":
                meta = {
                    "session_type": "person_agent_reflection",
                    "target_session_id": cfg["target_session"].session_id if cfg.get("target_session") else "",
                }
            else:  # probe
                meta = {
                    "session_type": "person_agent_probe",
                    "probe_type": s.probe_spec.probe_type,
                    "dimension_hint": s.probe_spec.dimension_hint,
                    "source_fact_ids": [f.fact_id for f in s.probe_spec.source_facts],
                }
            sessions.append(Session(
                session_id=s.session_id,
                participants=[s.agent.agent_id, s.aid],
                turns=s.turns,
                start_time=cfg["timestamp"],
                end_time=s.current_time,
                modality="text_message",
                metadata=meta,
            ))

        elapsed = _time.monotonic() - _wall_start
        log.info(
            "PA batch pipeline done: %d sessions, %d turns in %.1fs (%.2fs/session)",
            len(sessions), total_turns_generated, elapsed,
            elapsed / max(1, len(sessions)),
        )
        return sessions

    # ------------------------------------------------------------------
    # Wavefront (async per-session advancement) PA generation
    # ------------------------------------------------------------------

    def simulate_pa_sessions_wavefront(
        self,
        configs: List[Dict[str, Any]],
        dry_run: bool = False,
        call_tags: Optional[Dict[str, Any]] = None,
        stop_event: Optional[threading.Event] = None,
        on_complete_callback: Optional[Callable] = None,
    ) -> List[Session]:
        """Run all PA sessions with async wavefront scheduling.

        Each session advances independently as soon as its current turn's
        LLM response arrives — no cross-session synchronisation barriers.

        Returns:
            List of Session objects (call batch_finalize_pa_sessions() after).
        """
        if not configs:
            return []

        import time as _time

        total = len(configs)
        _wall_start = _time.monotonic()

        # --- shared state ---
        _stop_event = stop_event if stop_event is not None else threading.Event()
        _done_lock = threading.Lock()
        _active_count = total
        _all_done = threading.Event()
        _total_turns = [0]  # mutable counter under _done_lock

        # --- per-session state (reuse _Slot pattern) ---
        class _Slot:
            __slots__ = (
                "idx", "agent", "stype", "aid", "session_id", "topic_prompt",
                "target_summary", "probe_spec", "effective_max", "effective_min",
                "turns", "current_time", "step",
            )

        slots: List[_Slot] = []
        for i, cfg in enumerate(configs):
            s = _Slot()
            s.idx = i
            s.agent = cfg["agent"]
            s.stype = cfg["session_type"]
            s.aid = _assistant_id(s.agent.agent_id)
            s.session_id = f"sess_{uuid.uuid4().hex[:12]}"

            if s.stype == "activity":
                s.topic_prompt = cfg["topic"].get("prompt", "Tell me about your day.")
                s.target_summary = None
                s.probe_spec = None
            elif s.stype == "reflection":
                s.topic_prompt = None
                s.target_summary = self._summarise_session_for_reflection(
                    cfg["target_session"], s.agent.agent_id,
                )
                s.probe_spec = None
            else:  # probe
                s.topic_prompt = None
                s.target_summary = None
                s.probe_spec = cfg["probe_spec"]

            s.effective_max = self.probe_max_turns if s.stype == "probe" else self.max_turns
            s.effective_min = self.probe_min_turns if s.stype == "probe" else self.min_turns

            # Opener turn (step 0, no LLM call)
            if s.stype == "activity":
                opener_text = s.topic_prompt if not dry_run else f"[DRY RUN] Assistant: {s.topic_prompt}"
            elif s.stype == "reflection":
                if dry_run:
                    opener_text = f"[DRY RUN] Assistant: Let's revisit session {cfg['target_session'].session_id}"
                else:
                    opener_text = (
                        f"I'd like to revisit a conversation you had earlier. "
                        f"Here's what I remember: {s.target_summary} "
                        f"What stands out to you about that?"
                    )
            else:  # probe
                if dry_run:
                    opener_text = f"[DRY RUN] Assistant: Memory probe — {s.probe_spec.probe_type}"
                else:
                    opener_text = s.probe_spec.question_seed

            opener_turn = DialogueTurn(
                turn_id=f"{s.session_id}_t000",
                session_id=s.session_id,
                speaker_id=s.aid,
                listener_id=s.agent.agent_id,
                text=opener_text,
                timestamp=cfg["timestamp"],
                metadata={"turn_num": 0, "person_agent": True},
            )
            s.turns = [opener_turn]
            s.current_time = cfg["timestamp"] + 0.001
            s.step = 1  # opener is step 0, next LLM call is step 1
            slots.append(s)

        concurrency = getattr(self.llm_client, "cfg", None)
        max_workers = concurrency.concurrency if concurrency else 32

        log.info(
            "PA wavefront: %d sessions, max_workers=%d",
            total, max_workers,
        )

        # Collect finished sessions (thread-safe)
        finished_sessions: List[Session] = []

        def _finish_slot(s: _Slot, cfg: Dict, was_stopped: bool = False) -> None:
            """Build Session from completed slot and record it."""
            if s.stype == "activity":
                meta = {
                    "session_type": "person_agent_activity",
                    "topic_id": cfg.get("topic", {}).get("id", ""),
                    "topic_prompt": s.topic_prompt,
                }
            elif s.stype == "reflection":
                meta = {
                    "session_type": "person_agent_reflection",
                    "target_session_id": cfg["target_session"].session_id if cfg.get("target_session") else "",
                }
            else:  # probe
                meta = {
                    "session_type": "person_agent_probe",
                    "probe_type": s.probe_spec.probe_type,
                    "dimension_hint": s.probe_spec.dimension_hint,
                    "source_fact_ids": [f.fact_id for f in s.probe_spec.source_facts],
                }
            if was_stopped:
                meta["truncated"] = True
            session = Session(
                session_id=s.session_id,
                participants=[s.agent.agent_id, s.aid],
                turns=s.turns,
                start_time=cfg["timestamp"],
                end_time=s.current_time,
                modality="text_message",
                metadata=meta,
            )
            nonlocal _active_count
            with _done_lock:
                if not was_stopped:
                    finished_sessions.append(session)
                    _total_turns[0] += len(s.turns) - 1  # minus opener
                    if on_complete_callback:
                        on_complete_callback()
                _active_count -= 1
                if _active_count <= 0:
                    _all_done.set()

        def _advance_pa_slot(s: _Slot, cfg: Dict) -> Optional[_Slot]:
            """Generate one turn for slot s. Returns s if more work, else None."""
            step = s.step
            if step >= s.effective_max or _stop_event.is_set():
                _finish_slot(s, cfg, was_stopped=_stop_event.is_set())
                return None

            is_person_turn = (step % 2 == 1)

            if dry_run:
                if is_person_turn:
                    content = f"[DRY RUN] {s.agent.persona.name} turn {step}"
                    spk, lis = s.agent.agent_id, s.aid
                else:
                    content = f"[DRY RUN] Assistant turn {step}"
                    spk, lis = s.aid, s.agent.agent_id
                t = DialogueTurn(
                    turn_id=f"{s.session_id}_t{step:03d}",
                    session_id=s.session_id,
                    speaker_id=spk, listener_id=lis,
                    text=content, timestamp=s.current_time,
                    metadata={"turn_num": step, "thought": "", "person_agent": True},
                )
                s.turns.append(t)
                s.current_time += 0.001
                s.step += 1
                if step >= s.effective_min - 1 and _is_conversation_ending(content):
                    _finish_slot(s, cfg, was_stopped=False)
                    return None
                if _stop_event.is_set():
                    _finish_slot(s, cfg, was_stopped=True)
                    return None
                return s

            # --- LLM call ---
            if is_person_turn:
                if s.stype == "activity":
                    ctx = (
                        "This is a text conversation with an AI assistant.\n"
                        f"Topic: {s.topic_prompt}\n"
                    )
                elif s.stype == "reflection":
                    ctx = (
                        "This is a text conversation with an AI assistant.\n"
                        "You're reflecting on a past conversation you had.\n"
                        f"Context: {s.target_summary}\n"
                    )
                else:  # probe
                    fact_snippets = " | ".join(
                        f.content for f in s.probe_spec.source_facts[:3]
                    )
                    ctx = (
                        "This is a text conversation with an AI assistant.\n"
                        "The assistant is asking you about things you remember.\n"
                        "You are recalling from memory. You may not remember every detail "
                        "perfectly. Answer naturally — if you're unsure, say so. "
                        "Don't make things up, but it's okay to be approximate.\n"
                        f"Some things you vaguely recall: {fact_snippets}\n"
                    )
                sys_p, usr_p = s.agent.build_generation_context(
                    partner_name="AI Assistant",
                    partner_agent_id=s.aid,
                    event=None,
                    recent_turns=s.turns,
                    context_prefix=ctx,
                    modality="text_message",
                    no_token_limit=self.no_token_limit,
                )
                _tags: Dict[str, Any] = {
                    "phase": "pa_person_turn_wavefront",
                    "step": step, "session_type": s.stype,
                }
                if call_tags:
                    _tags.update(call_tags)
                stop_seqs = ["\nAI Assistant:", "\n\nAI Assistant:"]
                raw = self.llm_client.generate(sys_p, usr_p, tags=_tags,
                                               stop=stop_seqs)
                thought, content = _parse_turn_output(raw)
                spk, lis = s.agent.agent_id, s.aid
            else:
                # Assistant turn
                transcript = "\n".join(
                    f"{'AI Assistant' if is_assistant(t.speaker_id) else s.agent.persona.name}: {t.text}"
                    for t in s.turns[-8:]
                )
                if s.stype == "activity":
                    usr_p = (
                        f"Continue this conversation. The user is talking about: {s.topic_prompt}\n\n"
                        f"{transcript}\n\nAI Assistant:"
                    )
                elif s.stype == "reflection":
                    usr_p = (
                        f"Continue this reflection conversation. You're helping the user "
                        f"think back on a past conversation.\n"
                        f"Summary of the past conversation: {s.target_summary}\n\n"
                        f"{transcript}\n\nAI Assistant:"
                    )
                else:  # probe
                    usr_p = self._build_probe_assistant_prompt(
                        s.probe_spec, s.agent.persona.name, transcript, step,
                    )
                _tags = {
                    "phase": "pa_assistant_turn_wavefront",
                    "step": step, "session_type": s.stype,
                }
                if call_tags:
                    _tags.update(call_tags)
                person_name = s.agent.persona.name
                stop_seqs = [f"\n{person_name}:", f"\n\n{person_name}:"]
                raw = self.llm_client.generate(
                    _ASSISTANT_SYSTEM_PROMPT, usr_p, tags=_tags,
                    stop=stop_seqs,
                )
                _, content = _parse_turn_output(raw)
                thought = ""
                spk, lis = s.aid, s.agent.agent_id

            t = DialogueTurn(
                turn_id=f"{s.session_id}_t{step:03d}",
                session_id=s.session_id,
                speaker_id=spk, listener_id=lis,
                text=content, timestamp=s.current_time,
                metadata={"turn_num": step, "thought": thought, "person_agent": True},
            )
            s.turns.append(t)
            s.current_time += 0.001
            s.step += 1

            if step >= s.effective_min - 1 and _is_conversation_ending(content):
                _finish_slot(s, cfg, was_stopped=False)
                return None

            # Check stop event AFTER appending the turn
            if _stop_event.is_set():
                _finish_slot(s, cfg, was_stopped=True)
                return None

            return s  # more work needed

        def _on_step_done(future, s, cfg, executor):
            """Callback: chain next step or finish."""
            try:
                result_s = future.result()
            except Exception:
                log.exception("PA wavefront session %s failed", s.session_id)
                nonlocal _active_count
                with _done_lock:
                    _active_count -= 1
                    if _active_count <= 0:
                        _all_done.set()
                return

            if result_s is not None:
                f = executor.submit(_advance_pa_slot, result_s, cfg)
                f.add_done_callback(
                    lambda fut: _on_step_done(fut, result_s, cfg, executor)
                )

        # --- launch all sessions ---
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for s, cfg in zip(slots, configs):
                f = executor.submit(_advance_pa_slot, s, cfg)
                f.add_done_callback(
                    lambda fut, _s=s, _c=cfg: _on_step_done(fut, _s, _c, executor)
                )

            _all_done.wait()

        elapsed = _time.monotonic() - _wall_start
        log.info(
            "PA wavefront done: %d sessions, %d turns in %.1fs (%.2fs/session)",
            len(finished_sessions), _total_turns[0], elapsed,
            elapsed / max(1, len(finished_sessions)),
        )
        return finished_sessions

    def batch_finalize_pa_sessions(
        self,
        sessions: List[Session],
        agent_map: Dict[str, EntityAgent],
        call_tags: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Batch NER and knowledge update for PA sessions.

        Call after simulate_pa_sessions_stepped() to perform post-session
        processing in batches instead of per-session.
        """
        if not sessions:
            return

        import json as _json
        from MASim.prompts import KNOWLEDGE_EXTRACTION_SYSTEM as KNOWLEDGE_SYSTEM_PROMPT

        # --- Phase 1: Batch NER ---
        ner_prompts: List[Optional[Dict]] = []
        for sess in sessions:
            if not sess.turns:
                ner_prompts.append(None)
                continue
            transcript = "\n".join(
                f"Turn {i}: {t.text}" for i, t in enumerate(sess.turns)
            )
            example = _json.dumps({
                "0": {
                    "entities": ["Maya", "Jordan", "Elm Street Café"],
                    "facts": [
                        "Maya visited Elm Street Café with Jordan",
                        "Maya found the pastries at the café incredible",
                    ],
                },
                "1": {"entities": ["Jordan"], "facts": ["Jordan had recommended the café to Maya"]},
                "2": {"entities": [], "facts": []},
            })
            prompt = (
                f"Extract entities and memorable facts from each turn of this dialogue.\n\n"
                f"{transcript}\n\n"
                f"For facts: write concise third-person statements capturing specific "
                f"experiences, plans, opinions, or interpersonal details the speaker revealed. "
                f"Turns with no memorable content get an empty facts list.\n\n"
                f"Output JSON mapping turn numbers to objects:\n{example}"
            )
            max_tok = max(8192, len(sess.turns) * 180)
            _ner_tags: Dict[str, Any] = {"phase": "pa_ner"}
            if call_tags:
                _ner_tags.update(call_tags)
            ner_prompts.append({
                "system": KNOWLEDGE_SYSTEM_PROMPT,
                "user": prompt,
                "max_tokens": max_tok,
                "tags": _ner_tags,
            })

        valid_idx = [i for i, p in enumerate(ner_prompts) if p is not None]
        if valid_idx:
            batch = [ner_prompts[i] for i in valid_idx]
            from MASim.core.dialogue_engine import DialogueEngine
            ner_responses = self.llm_client.generate_batch(batch)
            for idx, raw in zip(valid_idx, ner_responses):
                DialogueEngine._parse_ner_response(raw, sessions[idx].turns)

        # --- Phase 2: Knowledge update (in-memory, fast) ---
        for sess in sessions:
            for pid in sess.participants:
                if pid in agent_map:
                    agent_map[pid].knowledge.update_from_session(sess, pid)

        # --- Phase 3: Batch mood updates ---
        mood_prompts: List[Dict] = []
        mood_targets: List[tuple] = []  # (pid, session) pairs
        for sess in sessions:
            relevant = [t for t in sess.turns[-6:] if t.text]
            if not relevant:
                continue
            # Only update mood for the human participant
            human_pids = [p for p in sess.participants if not is_assistant(p) and p in agent_map]
            if not human_pids:
                continue
            transcript = "\n".join(
                f"{'AI Assistant' if is_assistant(t.speaker_id) else agent_map.get(t.speaker_id, t).persona.name if hasattr(agent_map.get(t.speaker_id), 'persona') else t.speaker_id}: {t.text}"
                for t in relevant
            )
            for pid in human_pids:
                agent = agent_map[pid]
                _mood_tags: Dict[str, Any] = {"phase": "pa_mood"}
                if call_tags:
                    _mood_tags.update(call_tags)
                mood_prompts.append({
                    "system": (
                        "Assess how a person feels after a conversation. "
                        "Output one short phrase only — no explanation, no punctuation."
                    ),
                    "user": (
                        f"After this conversation, how is {agent.persona.name} feeling?"
                        f"\n\n{transcript}\n\nOne phrase:"
                    ),
                    "max_tokens": 8192,
                    "tags": _mood_tags,
                })
                mood_targets.append((pid, sess))

        if mood_prompts:
            from MASim.core.knowledge_state import KnownFeeling

            mood_responses = self.llm_client.generate_batch(mood_prompts)
            for (pid, sess), raw in zip(mood_targets, mood_responses):
                if pid in agent_map and raw:
                    mood_text = raw.strip().strip('"').strip("'").lower()
                    agent_map[pid].current_mood = mood_text
                    # Persist as KnownFeeling
                    feeling = KnownFeeling(
                        feeling_id=f"{pid}_feel_{len(agent_map[pid].knowledge.feelings):06d}",
                        content=mood_text,
                        trigger_agent="self",
                        session_id=sess.session_id,
                        timestamp=sess.end_time,
                    )
                    agent_map[pid].knowledge.add_feeling(feeling)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _summarise_session_for_reflection(
        session: Session, agent_id: str,
    ) -> str:
        """Build a brief natural-language summary of a session for reflection context."""
        partner_ids = [p for p in session.participants if p != agent_id]
        partner_label = ", ".join(partner_ids) if partner_ids else "someone"

        # Collect extracted facts from the session (max 5)
        facts = []
        for turn in session.turns:
            for f in turn.extracted_facts[:2]:
                facts.append(f)
                if len(facts) >= 5:
                    break
            if len(facts) >= 5:
                break

        if facts:
            fact_text = " ".join(facts)
            return f"You talked with {partner_label}. Key points: {fact_text}"
        # Fallback to first few turn texts
        snippets = [t.text[:80] for t in session.turns[:3] if t.text]
        snippet_text = " ... ".join(snippets)
        return f"You talked with {partner_label}. It went like: {snippet_text}"

    @staticmethod
    def _build_probe_assistant_prompt(
        probe_spec: Any,
        person_name: str,
        transcript: str,
        step: int,
    ) -> str:
        """Build assistant user-prompt for a memory probe turn.

        The assistant's role varies by probe type:
          - fact_recall: ask follow-up questions about recalled details
          - conflict_probe: highlight conflicting info, press for resolution
          - temporal_probe: ask about timeline ordering
        """
        pt = probe_spec.probe_type

        if pt == "conflict_probe":
            if step <= 4:
                return (
                    f"Continue probing this conflict. The user may have heard different "
                    f"versions of events from different people. Gently press for which "
                    f"version they believe, or ask if they're sure.\n\n"
                    f"{transcript}\n\nAI Assistant:"
                )
            else:
                return (
                    f"Wrap up the conversation about these conflicting memories. "
                    f"Summarise what the user said and thank them for thinking it through.\n\n"
                    f"{transcript}\n\nAI Assistant:"
                )

        elif pt == "temporal_probe":
            if step <= 4:
                return (
                    f"Continue asking about the timeline of events. Ask about "
                    f"what happened before or after specific moments, or ask "
                    f"about approximate timing.\n\n"
                    f"{transcript}\n\nAI Assistant:"
                )
            else:
                return (
                    f"Wrap up the timeline discussion. Summarise the order "
                    f"of events as the user described them.\n\n"
                    f"{transcript}\n\nAI Assistant:"
                )

        else:  # fact_recall
            if step <= 4:
                return (
                    f"Continue asking follow-up questions about the details "
                    f"the user mentioned. Ask about related people, places, "
                    f"or specifics they might remember.\n\n"
                    f"{transcript}\n\nAI Assistant:"
                )
            else:
                return (
                    f"Wrap up the conversation. Thank the user for sharing "
                    f"what they remember.\n\n"
                    f"{transcript}\n\nAI Assistant:"
                )

    def _update_agent_mood(self, agent: EntityAgent, session: Session) -> None:
        """Update emotional state after a person-agent session."""
        relevant_turns = [t for t in session.turns[-6:] if t.text and t.speaker_id == agent.agent_id]
        if not relevant_turns:
            return
        transcript = "\n".join(f"{agent.persona.name}: {t.text}" for t in relevant_turns)
        try:
            mood = self.llm_client.generate(
                "Assess how a person feels after a conversation. Output one short phrase only.",
                f"After this conversation, how is {agent.persona.name} feeling?\n\n{transcript}\n\nOne phrase:",
                max_tokens=32768,
                tags={"phase": "pa_mood"},
            )
            agent.current_mood = mood.strip().strip('"').strip("'").lower()
        except Exception as e:
            log.debug("Mood update failed for %s: %s", agent.agent_id, e)
