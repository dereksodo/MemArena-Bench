"""DialogueEngine: turn-by-turn LLM-based dialogue generation.

Simulates conversations between pairs of agents using an LLM backend.
Each turn is recorded with full metadata for ground truth derivation.
"""

from __future__ import annotations

import json
import re
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

from MASim.core.agent import EntityAgent
from MASim.core.schema import DialogueTurn, Session, WorldEvent
from MASim.ground_truth.json_parser import clean_llm_json_text, parse_llm_json
from MASim.prompts import (
    IMPRESSION_EXTRACTION_SYSTEM as IMPRESSION_SYSTEM_PROMPT,
    KNOWLEDGE_EXTRACTION_SYSTEM as KNOWLEDGE_SYSTEM_PROMPT,
    NER_EXTRACTION_SYSTEM as NER_SYSTEM_PROMPT,
)
from MASim.utils.logging import get_logger

log = get_logger(__name__)


def _parse_turn_output(text: str) -> tuple:
    """Parse THINK/SAY structured output from a dialogue generation call.

    Returns (thought, content). If the format is not present, thought is empty
    and content is the full raw text (graceful fallback).
    """
    think_match = re.search(r'THINK:\s*(.+?)(?=\nSAY:|\Z)', text, re.DOTALL | re.IGNORECASE)
    say_match = re.search(r'SAY:\s*(.+?)(?=\nTHINK:|\Z)', text, re.DOTALL | re.IGNORECASE)
    thought = think_match.group(1).strip() if think_match else ""
    content = say_match.group(1).strip() if say_match else text.strip()
    return thought, content


def _recover_partial_json(text: str) -> dict:
    """Recover turn entries from a truncated JSON response.

    Uses json.JSONDecoder.raw_decode to parse each "N": {...} object
    independently so that turns before the truncation point are still saved.
    """
    result: dict = {}
    text = clean_llm_json_text(text)
    decoder = json.JSONDecoder()
    for m in re.finditer(r'"(\d+)"\s*:', text):
        key = m.group(1)
        start = m.end()
        # Skip whitespace before the object
        while start < len(text) and text[start] in " \t\n\r":
            start += 1
        try:
            obj, _ = decoder.raw_decode(text, start)
            if isinstance(obj, dict):
                result[key] = obj
        except json.JSONDecodeError:
            pass  # this entry was truncated; stop trying further
    return result


def _apply_ner_result(result: Any, turns: List[DialogueTurn]) -> set[int]:
    """Apply a turn-indexed NER/fact response and return recovered turn ids."""
    if not isinstance(result, dict):
        raise ValueError(f"expected NER JSON object, got {type(result).__name__}")

    recovered: set[int] = set()
    skipped = 0
    for key, data in result.items():
        key_str = str(key)
        if not key_str.isdigit():
            skipped += 1
            continue
        idx = int(key_str)
        if 0 <= idx < len(turns) and isinstance(data, dict):
            entities = data.get("entities", [])
            facts = data.get("facts", [])
            if isinstance(entities, list):
                turns[idx].entities_mentioned = [str(e) for e in entities]
            if isinstance(facts, list):
                turns[idx].extracted_facts = [
                    str(f).strip() for f in facts if f and str(f).strip()
                ]
            recovered.add(idx)

    if skipped:
        log.debug("Skipped %d non-turn NER entries while parsing response", skipped)
    return recovered


def _build_modality_prefix(modality: str, speaker_persona: Any) -> str:
    """Build a context prefix describing the communication modality.

    Args:
        modality: "face_to_face", "voice_message", or "text_message"
        speaker_persona: PersonaCard of the current speaker

    Returns:
        A string to prepend to the user prompt (empty for face_to_face).
    """
    if modality == "voice_message":
        return (
            "This is a phone/video call. You and your conversation partner "
            "are in different locations, talking remotely. Adapt your speech "
            "naturally for this medium (e.g. greetings, sign-offs appropriate "
            "for a call).\n"
        )
    if modality == "text_message":
        tech = getattr(speaker_persona, "tech_affinity", 0.5)
        age = getattr(speaker_persona, "age", 30)

        # Texting style adapted to tech affinity
        if tech > 0.7:
            style = (
                "Use emojis, abbreviations (lol, omg, ngl, tbh), short fragmented "
                "messages. You're very fluent in digital communication."
            )
        elif tech >= 0.4:
            style = (
                "Occasional emoji, mostly complete sentences. You text comfortably "
                "but aren't overly casual about it."
            )
        else:
            style = (
                "Full sentences with proper punctuation. Minimal or no emoji. "
                "You text carefully, as if writing a short letter."
            )

        # Age-specific quirks
        age_note = ""
        if age < 25:
            age_note = " You sometimes double-text, use lowercase, and skip punctuation."
        elif age > 60:
            age_note = " You tend to over-capitalize, use ellipses, and add sign-offs like 'Best, [Name]'."

        return (
            f"This is a text message conversation. You are typing messages on your phone or computer.{age_note}\n"
            f"Texting style: {style}\n"
            "Insert `[IMAGE: brief description]` when you want to share a photo "
            "(e.g. `[IMAGE: sunset from my balcony]`). Only do this when it feels natural.\n"
        )
    # face_to_face — no prefix
    return ""


def batch_extract_session_knowledge(llm_client: Any, turns: List[DialogueTurn]) -> None:
    """Extract entities AND memorable facts from all turns in one LLM call.

    Each turn gets:
      - entities_mentioned : named people, places, organisations, specific things
      - extracted_facts    : 1-3 concise third-person factual statements

    Falls back to regex NER (no facts) if the LLM call fails.

    This is a module-level function so it can be reused by PersonAgentEngine.
    """
    transcript = "\n".join(f"Turn {i}: {t.text}" for i, t in enumerate(turns))
    example = json.dumps({
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

    # Hint: ~180 tokens per turn; ignored when client has unlimited_tokens=True
    max_tokens_hint = max(8192, len(turns) * 180)

    try:
        response = llm_client.generate(
            KNOWLEDGE_SYSTEM_PROMPT, prompt, max_tokens=max_tokens_hint,
            tags={"phase": "ner_knowledge"},
        )
        # Full parse first; partial recovery on truncation
        try:
            result = parse_llm_json(response)
        except ValueError:
            result = _recover_partial_json(response)
            if result:
                log.debug(
                    "Recovered %d/%d turn entries from truncated JSON",
                    len(result), len(turns),
                )

        recovered = _apply_ner_result(result, turns)

        # Regex NER only for turns the LLM didn't cover
        for idx, turn in enumerate(turns):
            if idx not in recovered:
                turn.entities_mentioned = _extract_entities_regex(turn.text)

    except (ValueError, KeyError) as e:
        log.warning("Session knowledge extraction failed, falling back to regex NER: %s", e)
        for turn in turns:
            turn.entities_mentioned = _extract_entities_regex(turn.text)


class DialogueEngine:
    """Generate multi-turn dialogues between agent pairs."""

    def __init__(
        self,
        llm_client: Any,  # inference.llm_client.LLMClient
        max_turns_per_session: int = 20,
        min_turns_per_session: int = 4,
        no_token_limit: bool = False,
    ):
        self.llm_client = llm_client
        self.max_turns = max_turns_per_session
        self.min_turns = min_turns_per_session
        self.no_token_limit = no_token_limit

    def simulate_conversation(
        self,
        agent_a: EntityAgent,
        agent_b: EntityAgent,
        triggering_event: Optional[WorldEvent],
        timestamp: float,
        dry_run: bool = False,
        modality: str = "face_to_face",
        session_seed: Optional[int] = None,
    ) -> Session:
        """Simulate a full conversation session between two agents.

        Agents alternate turns. Each turn is generated via LLM.
        """
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        turns: List[DialogueTurn] = []
        current_time = timestamp

        # Re-seed each agent's RNG uniquely per session so every dialogue
        # has independent random behaviour. XOR with a per-agent offset so
        # the two agents in the same session still differ from each other.
        if session_seed is not None:
            import numpy as _np
            agent_a._rng = _np.random.default_rng(session_seed ^ 0xA5A5A5A5)
            agent_b._rng = _np.random.default_rng(session_seed ^ 0x5A5A5A5A)

        agents = [agent_a, agent_b]
        speaker_idx = 0  # agent_a speaks first

        for turn_num in range(self.max_turns):
            speaker = agents[speaker_idx]
            listener = agents[1 - speaker_idx]

            # Build modality prefix:
            # - text_message: inject every turn (texting style must persist)
            # - voice_message: inject only on turn 0
            # - face_to_face: never
            if modality == "text_message":
                context_prefix = _build_modality_prefix(modality, speaker.persona)
            elif modality == "voice_message" and turn_num == 0:
                context_prefix = _build_modality_prefix(modality, speaker.persona)
            else:
                context_prefix = ""

            system_prompt, user_prompt = speaker.build_generation_context(
                partner_name=listener.persona.name,
                partner_agent_id=listener.agent_id,
                event=triggering_event if turn_num == 0 else None,
                recent_turns=turns,
                context_prefix=context_prefix,
                modality=modality,
                no_token_limit=self.no_token_limit,
            )

            if dry_run:
                thought, content = "", f"[DRY RUN] {speaker.persona.name} turn {turn_num}"
            else:
                raw = self.llm_client.generate(
                    system_prompt, user_prompt,
                    tags={"phase": "pp_turn", "step": turn_num},
                )
                thought, content = _parse_turn_output(raw)

            turn = DialogueTurn(
                turn_id=f"{session_id}_t{turn_num:03d}",
                session_id=session_id,
                speaker_id=speaker.agent_id,
                listener_id=listener.agent_id,
                text=content,
                timestamp=current_time,
                triggering_event=triggering_event.event_id if triggering_event and turn_num == 0 else None,
                metadata={"turn_num": turn_num, "thought": thought},
            )

            turns.append(turn)

            # Both agents receive the turn
            speaker.receive_turn(turn)
            listener.receive_turn(turn)

            current_time += 0.001  # sub-second increments within a session
            speaker_idx = 1 - speaker_idx  # alternate

            # Check for natural conversation end (after minimum turns)
            if turn_num >= self.min_turns - 1 and _is_conversation_ending(content):
                break

        # Batch NER: extract entities from all turns at once
        if not dry_run:
            self._batch_extract_session_knowledge(turns)

        session_meta: Dict[str, Any] = {}
        if modality == "voice_message":
            session_meta["voice_session"] = True
        elif modality == "text_message":
            session_meta["text_session"] = True

        session = Session(
            session_id=session_id,
            participants=[agent_a.agent_id, agent_b.agent_id],
            turns=turns,
            start_time=timestamp,
            end_time=current_time,
            triggering_events=[triggering_event.event_id] if triggering_event else [],
            modality=modality,
            metadata=session_meta,
        )

        # Update each agent's knowledge state from what they heard.
        # Done here (after NER) so entity_index is populated correctly.
        agent_a.knowledge.update_from_session(session, agent_a.agent_id)
        agent_b.knowledge.update_from_session(session, agent_b.agent_id)

        # Update emotional state for next session
        if not dry_run:
            self._update_agent_mood(agent_a, session)
            self._update_agent_mood(agent_b, session)

        # Extract behavioural impressions and cross-distribute:
        # what A observed about B's behaviour → stored in B's impression of A, and vice versa.
        if not dry_run:
            impressions = self._batch_extract_impressions(
                turns, [agent_a.agent_id, agent_b.agent_id]
            )
            if agent_a.agent_id in impressions:
                agent_b.knowledge.update_person_impression(
                    agent_a.agent_id, impressions[agent_a.agent_id]
                )
            if agent_b.agent_id in impressions:
                agent_a.knowledge.update_person_impression(
                    agent_b.agent_id, impressions[agent_b.agent_id]
                )

        log.debug(
            "Session %s: %s <-> %s, %d turns",
            session_id, agent_a.persona.name, agent_b.persona.name, len(turns),
        )
        return session

    def simulate_group_conversation(
        self,
        agents: List[EntityAgent],
        triggering_event: Optional[WorldEvent],
        timestamp: float,
        dry_run: bool = False,
        modality: str = "face_to_face",
        session_seed: Optional[int] = None,
        context_prefixes: Optional[Dict[str, str]] = None,
        call_tags: Optional[Dict[str, Any]] = None,
        defer_finalize: bool = False,
    ) -> Session:
        """Simulate a group conversation among 3+ agents.

        Agents take turns round-robin. Each speaker addresses the whole group.
        """
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        turns: List[DialogueTurn] = []
        current_time = timestamp

        # Re-seed each agent uniquely per session
        if session_seed is not None:
            import numpy as _np
            if defer_finalize:
                # Thread-safe: use session-local RNGs instead of mutating agent state
                _local_rngs = {
                    agents[i].agent_id: _np.random.default_rng(session_seed ^ (i * 0x11111111))
                    for i in range(len(agents))
                }
            else:
                for i, agent in enumerate(agents):
                    agent._rng = _np.random.default_rng(session_seed ^ (i * 0x11111111))
        n_agents = len(agents)

        for turn_num in range(self.max_turns):
            speaker = agents[turn_num % n_agents]
            others = [a for a in agents if a.agent_id != speaker.agent_id]

            # Role context prefix (location/group/role) per speaker
            role_ctx = (context_prefixes or {}).get(speaker.agent_id, "")

            # Build modality prefix per turn
            if modality == "text_message":
                ctx_prefix = role_ctx + _build_modality_prefix(modality, speaker.persona)
            elif modality == "voice_message" and turn_num == 0:
                ctx_prefix = role_ctx + _build_modality_prefix(modality, speaker.persona)
            else:
                ctx_prefix = role_ctx

            system_prompt, user_prompt = self._build_group_context(
                speaker, others, triggering_event if turn_num == 0 else None, turns,
                context_prefix=ctx_prefix,
                modality=modality,
                no_token_limit=self.no_token_limit,
            )

            if dry_run:
                thought, content = "", f"[DRY RUN] {speaker.persona.name} turn {turn_num}"
            else:
                _tags: Dict[str, Any] = {"phase": "pp_group_turn", "step": turn_num, "n_agents": n_agents}
                if call_tags:
                    _tags.update(call_tags)
                # Stop when the model starts generating another participant's turn
                stop_seqs = [f"\n{o.persona.name}:" for o in others]
                raw = self.llm_client.generate(system_prompt, user_prompt, tags=_tags,
                                               stop=stop_seqs)
                thought, content = _parse_turn_output(raw)

            turn = DialogueTurn(
                turn_id=f"{session_id}_t{turn_num:03d}",
                session_id=session_id,
                speaker_id=speaker.agent_id,
                listener_id="group",
                text=content,
                timestamp=current_time,
                triggering_event=triggering_event.event_id if triggering_event and turn_num == 0 else None,
                metadata={"turn_num": turn_num, "group_conversation": True, "thought": thought},
            )
            turns.append(turn)

            if not defer_finalize:
                for a in agents:
                    a.receive_turn(turn)

            current_time += 0.001
            if turn_num >= self.min_turns - 1 and _is_conversation_ending(content):
                break

        if not dry_run:
            self._batch_extract_session_knowledge(turns)

        group_meta: Dict[str, Any] = {"group_conversation": True}
        if modality == "voice_message":
            group_meta["voice_session"] = True
        elif modality == "text_message":
            group_meta["text_session"] = True

        session = Session(
            session_id=session_id,
            participants=[a.agent_id for a in agents],
            turns=turns,
            start_time=timestamp,
            end_time=current_time,
            triggering_events=[triggering_event.event_id] if triggering_event else [],
            modality=modality,
            metadata=group_meta,
        )

        # Update every participant's knowledge state from what they heard
        # (skipped when defer_finalize=True — batch_finalize_sessions handles it)
        if not defer_finalize:
            for agent in agents:
                agent.knowledge.update_from_session(session, agent.agent_id)

            # Update emotional state for each participant
            if not dry_run:
                for agent in agents:
                    self._update_agent_mood(agent, session)

            # Extract behavioural impressions for all participants and distribute:
            # observations about agent X go into every OTHER participant's impression of X.
            if not dry_run:
                participant_ids = [a.agent_id for a in agents]
                impressions = self._batch_extract_impressions(turns, participant_ids)
                agent_map = {a.agent_id: a for a in agents}
                for observed_id, observations in impressions.items():
                    for observer in agents:
                        if observer.agent_id != observed_id:
                            observer.knowledge.update_person_impression(
                                observed_id, observations
                            )

        log.debug(
            "Group session %s: %d agents, %d turns",
            session_id, n_agents, len(turns),
        )
        return session

    # ------------------------------------------------------------------
    # Step-wise batched conversation generation
    # ------------------------------------------------------------------

    def simulate_conversations_stepped(
        self,
        configs: List[Dict],
        dry_run: bool = False,
        call_tags: Optional[Dict[str, Any]] = None,
    ) -> List[Session]:
        """Simulate multiple pairwise conversations with step-wise batched turns.

        All sessions advance one turn at a time.  At each step every pending
        prompt is sent through ``generate_batch()`` so the backend sees N
        concurrent requests instead of 1-2.

        Args:
            configs: list of dicts, each with keys
                agent_a, agent_b  : EntityAgent instances
                event             : WorldEvent or None
                timestamp         : float
                modality          : str  (face_to_face / voice_message / text_message)
                session_seed      : int or None
            dry_run: produce placeholder text instead of LLM calls.

        Returns:
            Sessions with turns populated but **not** finalized — call
            ``batch_finalize_sessions()`` afterwards for NER / mood / impressions.
        """
        if not configs:
            return []

        # --- initialise per-session state ---
        states: List[Dict] = []
        for cfg in configs:
            session_id = f"sess_{uuid.uuid4().hex[:12]}"
            agent_a = cfg["agent_a"]
            agent_b = cfg["agent_b"]

            seed = cfg.get("session_seed")
            if seed is not None:
                import numpy as _np
                agent_a._rng = _np.random.default_rng(seed ^ 0xA5A5A5A5)
                agent_b._rng = _np.random.default_rng(seed ^ 0x5A5A5A5A)

            states.append({
                "session_id": session_id,
                "agent_a": agent_a,
                "agent_b": agent_b,
                "event": cfg.get("event"),
                "timestamp": cfg["timestamp"],
                "modality": cfg.get("modality", "face_to_face"),
                "context_prefix_a": cfg.get("context_prefix_a", ""),
                "context_prefix_b": cfg.get("context_prefix_b", ""),
                "turns": [],
                "speaker_idx": 0,
                "current_time": cfg["timestamp"],
                "done": False,
            })

        # --- step-wise loop ---
        for step in range(self.max_turns):
            active = [s for s in states if not s["done"]]
            if not active:
                break

            if dry_run:
                for s in active:
                    agents = [s["agent_a"], s["agent_b"]]
                    speaker = agents[s["speaker_idx"]]
                    listener = agents[1 - s["speaker_idx"]]

                    turn = DialogueTurn(
                        turn_id=f"{s['session_id']}_t{step:03d}",
                        session_id=s["session_id"],
                        speaker_id=speaker.agent_id,
                        listener_id=listener.agent_id,
                        text=f"[DRY RUN] {speaker.persona.name} turn {step}",
                        timestamp=s["current_time"],
                        triggering_event=(
                            s["event"].event_id if s["event"] and step == 0 else None
                        ),
                        metadata={"turn_num": step, "thought": ""},
                    )
                    s["turns"].append(turn)
                    speaker.receive_turn(turn)
                    listener.receive_turn(turn)
                    s["current_time"] += 0.001
                    s["speaker_idx"] = 1 - s["speaker_idx"]
                    if step >= self.min_turns - 1 and _is_conversation_ending(turn.text):
                        s["done"] = True
            else:
                # ---- BUILD prompts (READ-ONLY agent access) ----
                prompts: List[Dict] = []
                for s in active:
                    agents = [s["agent_a"], s["agent_b"]]
                    speaker = agents[s["speaker_idx"]]
                    listener = agents[1 - s["speaker_idx"]]

                    # Role context prefix (location/group/role)
                    role_ctx = s["context_prefix_a"] if s["speaker_idx"] == 0 else s["context_prefix_b"]

                    if s["modality"] == "text_message":
                        ctx = role_ctx + _build_modality_prefix(s["modality"], speaker.persona)
                    elif s["modality"] == "voice_message" and step == 0:
                        ctx = role_ctx + _build_modality_prefix(s["modality"], speaker.persona)
                    else:
                        ctx = role_ctx

                    sys_p, usr_p = speaker.build_generation_context(
                        partner_name=listener.persona.name,
                        partner_agent_id=listener.agent_id,
                        event=s["event"] if step == 0 else None,
                        recent_turns=s["turns"],
                        context_prefix=ctx,
                        modality=s["modality"],
                        no_token_limit=self.no_token_limit,
                        interest_domain=s.get("interest_domain", ""),
                    )
                    prompt_tags: Dict[str, Any] = {"phase": "pp_turn", "step": step, "n_active": len(active)}
                    if call_tags:
                        prompt_tags.update(call_tags)
                    prompts.append({"system": sys_p, "user": usr_p, "tags": prompt_tags})

                # ---- BATCH GENERATE (concurrent HTTP I/O) ----
                responses = self.llm_client.generate_batch(prompts)

                # ---- PROCESS results sequentially (WRITE) ----
                for s, raw in zip(active, responses):
                    agents = [s["agent_a"], s["agent_b"]]
                    speaker = agents[s["speaker_idx"]]
                    listener = agents[1 - s["speaker_idx"]]

                    thought, content = _parse_turn_output(raw)
                    turn = DialogueTurn(
                        turn_id=f"{s['session_id']}_t{step:03d}",
                        session_id=s["session_id"],
                        speaker_id=speaker.agent_id,
                        listener_id=listener.agent_id,
                        text=content,
                        timestamp=s["current_time"],
                        triggering_event=(
                            s["event"].event_id if s["event"] and step == 0 else None
                        ),
                        metadata={"turn_num": step, "thought": thought},
                    )
                    s["turns"].append(turn)
                    speaker.receive_turn(turn)
                    listener.receive_turn(turn)
                    s["current_time"] += 0.001
                    s["speaker_idx"] = 1 - s["speaker_idx"]
                    if step >= self.min_turns - 1 and _is_conversation_ending(content):
                        s["done"] = True

        # --- build Session objects ---
        sessions: List[Session] = []
        for s in states:
            meta: Dict[str, Any] = {}
            if s["modality"] == "voice_message":
                meta["voice_session"] = True
            elif s["modality"] == "text_message":
                meta["text_session"] = True

            session = Session(
                session_id=s["session_id"],
                participants=[s["agent_a"].agent_id, s["agent_b"].agent_id],
                turns=s["turns"],
                start_time=s["timestamp"],
                end_time=s["current_time"],
                triggering_events=[s["event"].event_id] if s["event"] else [],
                modality=s["modality"],
                metadata=meta,
            )
            sessions.append(session)
            log.debug(
                "Session %s: %s <-> %s, %d turns",
                s["session_id"],
                s["agent_a"].persona.name,
                s["agent_b"].persona.name,
                len(s["turns"]),
            )
        return sessions

    # ------------------------------------------------------------------
    # Racing pairwise conversation generation (--faster-batch)
    # ------------------------------------------------------------------

    def simulate_conversations_racing(
        self,
        configs: List[Dict],
        n_target: int,
        n_truncated_keep: int,
        dry_run: bool = False,
        call_tags: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Session], List[Session]]:
        """Race multiple pairwise conversations concurrently.

        Over-provisions configs, runs them all in parallel, and stops once
        ``n_target`` sessions complete.  Incomplete sessions are ranked by
        (turns DESC, tokens DESC) and the top ``n_truncated_keep`` are kept
        as truncated corpus entries.

        Returns:
            (completed_sessions, truncated_sessions)
        """
        if not configs:
            return [], []

        _stop_event = threading.Event()
        _done_lock = threading.Lock()
        _n_completed = 0

        def _run_one_session(cfg: Dict) -> Tuple[Session, bool]:
            nonlocal _n_completed

            session_id = f"sess_{uuid.uuid4().hex[:12]}"
            agent_a = cfg["agent_a"]
            agent_b = cfg["agent_b"]

            seed = cfg.get("session_seed")
            if seed is not None:
                import numpy as _np
                # Thread-local RNG copies to avoid shared state
                rng_a = _np.random.default_rng(seed ^ 0xA5A5A5A5)
                rng_b = _np.random.default_rng(seed ^ 0x5A5A5A5A)
            else:
                import numpy as _np
                rng_a = _np.random.default_rng()
                rng_b = _np.random.default_rng()

            agents_pair = [agent_a, agent_b]
            turns: List[DialogueTurn] = []
            current_time = cfg["timestamp"]
            speaker_idx = 0
            modality = cfg.get("modality", "face_to_face")
            event = cfg.get("event")

            for step in range(self.max_turns):
                # Check stop event at top of each iteration
                if _stop_event.is_set():
                    break

                speaker = agents_pair[speaker_idx]
                listener = agents_pair[1 - speaker_idx]

                # Role context prefix
                role_ctx = cfg.get("context_prefix_a", "") if speaker_idx == 0 else cfg.get("context_prefix_b", "")

                if modality == "text_message":
                    ctx = role_ctx + _build_modality_prefix(modality, speaker.persona)
                elif modality == "voice_message" and step == 0:
                    ctx = role_ctx + _build_modality_prefix(modality, speaker.persona)
                else:
                    ctx = role_ctx

                sys_p, usr_p = speaker.build_generation_context(
                    partner_name=listener.persona.name,
                    partner_agent_id=listener.agent_id,
                    event=event if step == 0 else None,
                    recent_turns=turns,
                    context_prefix=ctx,
                    modality=modality,
                    no_token_limit=self.no_token_limit,
                )

                if dry_run:
                    thought, content = "", f"[DRY RUN] {speaker.persona.name} turn {step}"
                else:
                    prompt_tags: Dict[str, Any] = {"phase": "pp_turn_racing", "step": step}
                    if call_tags:
                        prompt_tags.update(call_tags)
                    stop_seqs = [f"\n{listener.persona.name}:",
                                 f"\n\n{listener.persona.name}:"]
                    raw = self.llm_client.generate(sys_p, usr_p, tags=prompt_tags,
                                                   stop=stop_seqs)
                    thought, content = _parse_turn_output(raw)

                turn = DialogueTurn(
                    turn_id=f"{session_id}_t{step:03d}",
                    session_id=session_id,
                    speaker_id=speaker.agent_id,
                    listener_id=listener.agent_id,
                    text=content,
                    timestamp=current_time,
                    triggering_event=event.event_id if event and step == 0 else None,
                    metadata={"turn_num": step, "thought": thought},
                )
                turns.append(turn)

                # Do NOT call agent.receive_turn() — thread safety
                current_time += 0.001
                speaker_idx = 1 - speaker_idx

                if step >= self.min_turns - 1 and _is_conversation_ending(content):
                    break

            was_completed = not _stop_event.is_set()

            # Build session
            meta: Dict[str, Any] = {}
            if modality == "voice_message":
                meta["voice_session"] = True
            elif modality == "text_message":
                meta["text_session"] = True
            if not was_completed:
                meta["truncated"] = True

            session = Session(
                session_id=session_id,
                participants=[agent_a.agent_id, agent_b.agent_id],
                turns=turns,
                start_time=cfg["timestamp"],
                end_time=current_time,
                triggering_events=[event.event_id] if event else [],
                modality=modality,
                metadata=meta,
            )

            # Counting logic
            if was_completed:
                with _done_lock:
                    _n_completed += 1
                    if _n_completed >= n_target:
                        _stop_event.set()

            return session, was_completed

        # --- Launch all sessions concurrently ---
        concurrency = getattr(self.llm_client, "cfg", None)
        max_workers = concurrency.concurrency if concurrency else 32
        log.info(
            "Racing %d configs (%d target + %d extra), max_workers=%d",
            len(configs), n_target, len(configs) - n_target, max_workers,
        )

        completed: List[Session] = []
        truncated: List[Session] = []

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_run_one_session, cfg): cfg for cfg in configs}
            for future in as_completed(futures):
                try:
                    session, was_completed = future.result()
                    if was_completed:
                        completed.append(session)
                    else:
                        truncated.append(session)
                except Exception:
                    log.exception("Racing session failed")

        # Sort truncated by (turns DESC, tokens DESC) and keep top n_truncated_keep
        truncated.sort(
            key=lambda s: (
                len(s.turns),
                sum(len(t.text.split()) for t in s.turns),
            ),
            reverse=True,
        )
        kept_truncated = truncated[:n_truncated_keep]

        log.info(
            "Racing done: %d completed, %d truncated (%d kept)",
            len(completed), len(truncated), len(kept_truncated),
        )
        return completed[:n_target], kept_truncated

    # ------------------------------------------------------------------
    # Wavefront (async per-session advancement) conversation generation
    # ------------------------------------------------------------------

    def simulate_conversations_wavefront(
        self,
        configs: List[Dict],
        dry_run: bool = False,
        call_tags: Optional[Dict[str, Any]] = None,
        n_target: Optional[int] = None,
        n_truncated_keep: int = 0,
        stop_event: Optional[threading.Event] = None,
        on_complete_callback: Optional[Callable] = None,
    ) -> Tuple[List[Session], List[Session]]:
        """Run pairwise conversations with async wavefront scheduling.

        Each session advances independently as soon as its current turn's LLM
        response arrives — no cross-session synchronisation barriers.

        When ``n_target`` is set (racing mode), generation stops once that many
        sessions complete naturally.  Incomplete sessions are ranked by quality
        and the top ``n_truncated_keep`` are returned as truncated entries.

        When ``n_target`` is None, all sessions run to completion.

        Returns:
            (completed_sessions, truncated_sessions)
        """
        if not configs:
            return [], []

        import time as _time

        _wall_start = _time.monotonic()

        # --- shared state ---
        _stop_event = stop_event if stop_event is not None else threading.Event()
        _external_stop = stop_event is not None
        _done_lock = threading.Lock()
        _n_completed = 0
        _active_count = len(configs)
        _all_done = threading.Event()

        completed: List[Tuple[int, Session]] = []  # (index, session)
        truncated: List[Session] = []

        racing = n_target is not None

        # --- per-session state ---
        class _WaveState:
            __slots__ = (
                "idx", "session_id", "agent_a", "agent_b", "event",
                "timestamp", "modality", "context_prefix_a", "context_prefix_b",
                "turns", "speaker_idx", "current_time", "step",
            )

        wave_states: List[_WaveState] = []
        for i, cfg in enumerate(configs):
            ws = _WaveState()
            ws.idx = i
            ws.session_id = f"sess_{uuid.uuid4().hex[:12]}"
            ws.agent_a = cfg["agent_a"]
            ws.agent_b = cfg["agent_b"]
            ws.event = cfg.get("event")
            ws.timestamp = cfg["timestamp"]
            ws.modality = cfg.get("modality", "face_to_face")
            ws.context_prefix_a = cfg.get("context_prefix_a", "")
            ws.context_prefix_b = cfg.get("context_prefix_b", "")
            ws.turns = []
            ws.speaker_idx = 0
            ws.current_time = cfg["timestamp"]
            ws.step = 0

            seed = cfg.get("session_seed")
            if seed is not None:
                import numpy as _np
                ws.agent_a._rng = _np.random.default_rng(seed ^ 0xA5A5A5A5)
                ws.agent_b._rng = _np.random.default_rng(seed ^ 0x5A5A5A5A)

            wave_states.append(ws)

        concurrency = getattr(self.llm_client, "cfg", None)
        max_workers = concurrency.concurrency if concurrency else 32
        total = len(configs)

        log.info(
            "PP wavefront: %d sessions, max_turns=%d, workers=%d%s",
            total, self.max_turns, max_workers,
            f", racing target={n_target}" if racing else "",
        )

        def _finish_session(ws: _WaveState, was_stopped: bool) -> None:
            nonlocal _n_completed, _active_count

            meta: Dict[str, Any] = {}
            if ws.modality == "voice_message":
                meta["voice_session"] = True
            elif ws.modality == "text_message":
                meta["text_session"] = True
            if was_stopped:
                meta["truncated"] = True

            session = Session(
                session_id=ws.session_id,
                participants=[ws.agent_a.agent_id, ws.agent_b.agent_id],
                turns=ws.turns,
                start_time=ws.timestamp,
                end_time=ws.current_time,
                triggering_events=[ws.event.event_id] if ws.event else [],
                modality=ws.modality,
                metadata=meta,
            )

            with _done_lock:
                if not was_stopped:
                    _n_completed += 1
                    completed.append((ws.idx, session))
                    if on_complete_callback:
                        on_complete_callback()
                    if not _external_stop and racing and _n_completed >= n_target:
                        _stop_event.set()
                else:
                    truncated.append(session)
                _active_count -= 1
                if _active_count <= 0:
                    _all_done.set()

        def _advance_session(ws: _WaveState) -> Optional[_WaveState]:
            """Generate one turn for ws. Returns ws if more work needed, else None."""
            step = ws.step
            if step >= self.max_turns or _stop_event.is_set():
                _finish_session(ws, was_stopped=_stop_event.is_set())
                return None

            agents = [ws.agent_a, ws.agent_b]
            speaker = agents[ws.speaker_idx]
            listener = agents[1 - ws.speaker_idx]

            role_ctx = ws.context_prefix_a if ws.speaker_idx == 0 else ws.context_prefix_b

            if ws.modality == "text_message":
                ctx = role_ctx + _build_modality_prefix(ws.modality, speaker.persona)
            elif ws.modality == "voice_message" and step == 0:
                ctx = role_ctx + _build_modality_prefix(ws.modality, speaker.persona)
            else:
                ctx = role_ctx

            sys_p, usr_p = speaker.build_generation_context(
                partner_name=listener.persona.name,
                partner_agent_id=listener.agent_id,
                event=ws.event if step == 0 else None,
                recent_turns=ws.turns,
                context_prefix=ctx,
                modality=ws.modality,
                no_token_limit=self.no_token_limit,
            )

            if dry_run:
                thought, content = "", f"[DRY RUN] {speaker.persona.name} turn {step}"
            else:
                prompt_tags: Dict[str, Any] = {"phase": "pp_turn_wavefront", "step": step}
                if call_tags:
                    prompt_tags.update(call_tags)
                # Stop when the model starts generating the other person's turn
                stop_seqs = [f"\n{listener.persona.name}:",
                             f"\n\n{listener.persona.name}:"]
                raw = self.llm_client.generate(sys_p, usr_p, tags=prompt_tags,
                                               stop=stop_seqs)
                thought, content = _parse_turn_output(raw)

            turn = DialogueTurn(
                turn_id=f"{ws.session_id}_t{step:03d}",
                session_id=ws.session_id,
                speaker_id=speaker.agent_id,
                listener_id=listener.agent_id,
                text=content,
                timestamp=ws.current_time,
                triggering_event=(
                    ws.event.event_id if ws.event and step == 0 else None
                ),
                metadata={"turn_num": step, "thought": thought},
            )
            ws.turns.append(turn)
            # Do NOT call agent.receive_turn() — thread safety (same as racing)
            ws.current_time += 0.001
            ws.speaker_idx = 1 - ws.speaker_idx
            ws.step += 1

            if step >= self.min_turns - 1 and _is_conversation_ending(content):
                _finish_session(ws, was_stopped=False)
                return None

            # Check stop event AFTER appending the turn
            if _stop_event.is_set():
                _finish_session(ws, was_stopped=True)
                return None

            return ws  # more work needed

        def _on_step_done(future, ws, executor):
            """Callback: chain next step or finish."""
            try:
                result_ws = future.result()
            except Exception:
                log.exception("Wavefront session %s failed", ws.session_id)
                with _done_lock:
                    nonlocal _active_count
                    _active_count -= 1
                    if _active_count <= 0:
                        _all_done.set()
                return

            if result_ws is not None:
                # Session needs another step — submit and chain callback
                f = executor.submit(_advance_session, result_ws)
                f.add_done_callback(lambda fut: _on_step_done(fut, result_ws, executor))

        # --- launch all sessions ---
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for ws in wave_states:
                f = executor.submit(_advance_session, ws)
                f.add_done_callback(lambda fut, _ws=ws: _on_step_done(fut, _ws, executor))

            # Wait for all sessions to complete (or stop event)
            _all_done.wait()

        # --- collect results ---
        elapsed = _time.monotonic() - _wall_start
        total_turns = sum(len(ws.turns) for ws in wave_states)

        # Sort completed by original index to preserve ordering
        completed.sort(key=lambda x: x[0])
        completed_sessions = [sess for _, sess in completed]

        # Sort truncated by quality (turns DESC, tokens DESC)
        truncated.sort(
            key=lambda s: (
                len(s.turns),
                sum(len(t.text.split()) for t in s.turns),
            ),
            reverse=True,
        )
        kept_truncated = truncated[:n_truncated_keep] if racing else []

        log.info(
            "PP wavefront done: %d completed, %d truncated (%d kept), "
            "%d total turns in %.1fs (%.1f turns/s)",
            len(completed_sessions), len(truncated), len(kept_truncated),
            total_turns, elapsed, total_turns / max(0.1, elapsed),
        )

        if racing:
            return completed_sessions[:n_target], kept_truncated
        return completed_sessions, []

    # ------------------------------------------------------------------
    # Generate next turns for truncated sessions (D5 next-turn-prediction)
    # ------------------------------------------------------------------

    def generate_next_turns_for_truncated(
        self,
        truncated_sessions: List[Session],
        agent_map: Dict[str, "EntityAgent"],
        dry_run: bool = False,
        call_tags: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Generate one more turn per truncated session for D5 next-turn cloze.

        The generated turn is stored in ``sess.metadata["cloze_next_turn"]``
        and is NOT appended to ``sess.turns`` — it's metadata-only, used
        exclusively by D5 ground truth extraction.
        """
        if not truncated_sessions:
            return

        tasks: List[Dict] = []
        task_meta: List[Tuple[Session, str]] = []  # (session, speaker_id)

        for sess in truncated_sessions:
            if not sess.turns:
                continue
            # Next speaker is the listener of the last turn
            last_turn = sess.turns[-1]
            next_speaker_id = last_turn.listener_id
            if next_speaker_id not in agent_map:
                continue

            speaker = agent_map[next_speaker_id]
            other_id = last_turn.speaker_id

            sys_p, usr_p = speaker.build_generation_context(
                partner_name=agent_map[other_id].persona.name if other_id in agent_map else other_id,
                partner_agent_id=other_id,
                event=None,
                recent_turns=sess.turns,
                context_prefix="",
                modality=sess.modality,
                no_token_limit=self.no_token_limit,
            )

            prompt_tags: Dict[str, Any] = {"phase": "racing_next_turn"}
            if call_tags:
                prompt_tags.update(call_tags)

            if dry_run:
                sess.metadata["cloze_next_turn"] = {
                    "speaker_id": next_speaker_id,
                    "text": f"[DRY RUN] {speaker.persona.name} next turn",
                    "thought": "",
                    "timestamp": sess.end_time + 0.001,
                }
                continue

            tasks.append({"system": sys_p, "user": usr_p, "tags": prompt_tags})
            task_meta.append((sess, next_speaker_id))

        if not tasks:
            return

        log.info("Generating %d next-turn predictions for truncated sessions...", len(tasks))
        responses = self.llm_client.generate_batch(tasks)

        for (sess, speaker_id), raw in zip(task_meta, responses):
            thought, content = _parse_turn_output(raw)
            sess.metadata["cloze_next_turn"] = {
                "speaker_id": speaker_id,
                "text": content,
                "thought": thought,
                "timestamp": sess.end_time + 0.001,
            }

    def batch_finalize_sessions(
        self,
        sessions: List[Session],
        agent_map: Dict[str, "EntityAgent"],
        call_tags: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Batch NER, knowledge update, mood, and impressions across sessions.

        Call this after ``simulate_conversations_stepped()`` to perform all
        post-session processing in large batches instead of per-session.
        """
        if not sessions:
            return

        # --- Phase 1: Batch NER across all sessions ---
        ner_prompts: List[Optional[Dict]] = []
        for sess in sessions:
            if not sess.turns:
                ner_prompts.append(None)
                continue
            transcript = "\n".join(
                f"Turn {i}: {t.text}" for i, t in enumerate(sess.turns)
            )
            example = json.dumps({
                "0": {
                    "entities": ["Maya", "Jordan", "Elm Street Café"],
                    "facts": [
                        "Maya visited Elm Street Café with Jordan",
                        "Maya found the pastries at the café incredible",
                    ],
                },
                "1": {
                    "entities": ["Jordan"],
                    "facts": ["Jordan had recommended the café to Maya"],
                },
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
            _ner_tags: Dict[str, Any] = {"phase": "pp_ner"}
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
            ner_responses = self.llm_client.generate_batch(batch)
            for idx, raw in zip(valid_idx, ner_responses):
                self._parse_ner_response(raw, sessions[idx].turns)

        # --- Phase 2: Sequential knowledge update (fast, in-memory) ---
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
            transcript = "\n".join(f"{t.speaker_id}: {t.text}" for t in relevant)
            for pid in sess.participants:
                if pid in agent_map:
                    agent = agent_map[pid]
                    _mood_tags: Dict[str, Any] = {"phase": "pp_mood"}
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
                    log.debug("Mood update for %s: %s", pid, mood_text)
                    # Persist as KnownFeeling
                    feeling = KnownFeeling(
                        feeling_id=f"{pid}_feel_{len(agent_map[pid].knowledge.feelings):06d}",
                        content=mood_text,
                        trigger_agent="self",
                        session_id=sess.session_id,
                        timestamp=sess.end_time,
                    )
                    agent_map[pid].knowledge.add_feeling(feeling)

        # --- Phase 4: Batch impression extraction ---
        imp_prompts: List[Dict] = []
        imp_sess_idx: List[int] = []
        for i, sess in enumerate(sessions):
            if not sess.turns:
                continue
            transcript = "\n".join(f"{t.speaker_id}: {t.text}" for t in sess.turns)
            example = json.dumps(
                {pid: ["observation 1", "observation 2"] for pid in sess.participants},
                indent=2,
            )
            _imp_tags: Dict[str, Any] = {"phase": "pp_impression"}
            if call_tags:
                _imp_tags.update(call_tags)
            imp_prompts.append({
                "system": IMPRESSION_SYSTEM_PROMPT,
                "user": (
                    f"Read this conversation and note 1-2 specific behavioural observations "
                    f"about each speaker — how they communicate, what they reveal about their "
                    f"personality, their verbal habits, emotional cues, etc.\n\n"
                    f"{transcript}\n\n"
                    f"Output a JSON object with exactly these keys and string-array values:\n"
                    f"{example}"
                ),
                "max_tokens": 8192,
                "tags": _imp_tags,
            })
            imp_sess_idx.append(i)

        if imp_prompts:
            imp_responses = self.llm_client.generate_batch(imp_prompts)
            for si, raw in zip(imp_sess_idx, imp_responses):
                sess = sessions[si]
                try:
                    result = parse_llm_json(raw)
                    impressions = {
                        k: [str(obs) for obs in v if obs]
                        for k, v in result.items()
                        if k in sess.participants and isinstance(v, list)
                    }
                except (ValueError, TypeError, KeyError, AttributeError):
                    impressions = {}

                # Cross-distribute: observations about A → B's memory of A
                if len(sess.participants) == 2:
                    a_id, b_id = sess.participants
                    if a_id in impressions and b_id in agent_map:
                        agent_map[b_id].knowledge.update_person_impression(
                            a_id, impressions[a_id]
                        )
                    if b_id in impressions and a_id in agent_map:
                        agent_map[a_id].knowledge.update_person_impression(
                            b_id, impressions[b_id]
                        )

    @staticmethod
    def _parse_ner_response(raw: str, turns: List[DialogueTurn]) -> None:
        """Parse a NER/facts JSON response and annotate *turns* in-place."""
        try:
            try:
                result = parse_llm_json(raw)
            except ValueError:
                result = _recover_partial_json(raw)
                if result:
                    log.debug(
                        "Recovered %d/%d turn entries from truncated JSON",
                        len(result), len(turns),
                    )

            recovered = _apply_ner_result(result, turns)

            for idx, turn in enumerate(turns):
                if idx not in recovered:
                    turn.entities_mentioned = _extract_entities_regex(turn.text)

        except (ValueError, KeyError) as e:
            log.warning("Session NER extraction failed, falling back to regex: %s", e)
            for turn in turns:
                turn.entities_mentioned = _extract_entities_regex(turn.text)

    def unified_batch_finalize(
        self,
        pp_sessions: list,
        pa_sessions: list,
        agent_map: dict,
        call_tags: dict | None = None,
    ) -> None:
        """Finalize PP + PA sessions in a single LLM batch.

        Combines NER, mood, and impression prompts for both PP and PA
        sessions into one ``generate_batch`` call, then applies knowledge
        updates in-memory.
        """
        from MASim.core.knowledge_state import KnownFeeling
        from MASim.core.schema import is_assistant

        pp_list = list(pp_sessions)
        pa_list = list(pa_sessions)
        all_sessions = pp_list + pa_list
        n_pp = len(pp_list)
        n_pa = len(pa_list)
        # Index-based: sessions 0..n_pp-1 are PP, n_pp.. are PA
        if not all_sessions:
            return
        n_turns = sum(len(s.turns) for s in all_sessions)
        log.info(
            "Day %s: unified finalize — %d PP + %d PA sessions (%d turns)",
            call_tags.get("day_idx", "?") if call_tags else "?",
            n_pp, n_pa, n_turns,
        )

        # ================================================================
        # Build all prompts with markers so we can route results back
        # ================================================================
        batch: list = []
        # Each entry in dispatch tells us how to handle the response:
        #   ("ner", session_index)
        #   ("mood", agent_id, session_index)
        #   ("impression", session_index)
        dispatch: list = []

        ner_example = json.dumps({
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

        for si, sess in enumerate(all_sessions):
            is_pp = si < n_pp

            # --- NER prompt ---
            if sess.turns:
                transcript = "\n".join(
                    f"Turn {i}: {t.text}" for i, t in enumerate(sess.turns)
                )
                ner_phase = "pp_ner" if is_pp else "pa_ner"
                _tags: dict = {"phase": ner_phase}
                if call_tags:
                    _tags.update(call_tags)
                batch.append({
                    "system": KNOWLEDGE_SYSTEM_PROMPT,
                    "user": (
                        f"Extract entities and memorable facts from each turn of this dialogue.\n\n"
                        f"{transcript}\n\n"
                        f"For facts: write concise third-person statements capturing specific "
                        f"experiences, plans, opinions, or interpersonal details the speaker revealed. "
                        f"Turns with no memorable content get an empty facts list.\n\n"
                        f"Output JSON mapping turn numbers to objects:\n{ner_example}"
                    ),
                    "max_tokens": max(8192, len(sess.turns) * 180),
                    "tags": _tags,
                })
                dispatch.append(("ner", si))

            # --- Mood prompts (one per human participant) ---
            relevant = [t for t in sess.turns[-6:] if t.text]
            if relevant:
                if is_pp:
                    mood_pids = [p for p in sess.participants if p in agent_map]
                else:
                    mood_pids = [p for p in sess.participants
                                 if not is_assistant(p) and p in agent_map]

                # Build transcript with readable names
                mood_lines = []
                for t in relevant:
                    if is_assistant(t.speaker_id):
                        name = "AI Assistant"
                    elif t.speaker_id in agent_map and hasattr(agent_map[t.speaker_id], 'persona'):
                        name = agent_map[t.speaker_id].persona.name
                    else:
                        name = t.speaker_id
                    mood_lines.append(f"{name}: {t.text}")
                mood_transcript = "\n".join(mood_lines)

                for pid in mood_pids:
                    mood_phase = "pp_mood" if is_pp else "pa_mood"
                    _tags = {"phase": mood_phase}
                    if call_tags:
                        _tags.update(call_tags)
                    batch.append({
                        "system": (
                            "Assess how a person feels after a conversation. "
                            "Output one short phrase only — no explanation, no punctuation."
                        ),
                        "user": (
                            f"After this conversation, how is {agent_map[pid].persona.name} feeling?"
                            f"\n\n{mood_transcript}\n\nOne phrase:"
                        ),
                        "max_tokens": 8192,
                        "tags": _tags,
                    })
                    dispatch.append(("mood", pid, si))

            # --- Impression prompt (PP only) ---
            if is_pp and sess.turns:
                transcript = "\n".join(
                    f"{t.speaker_id}: {t.text}" for t in sess.turns
                )
                example = json.dumps(
                    {pid: ["observation 1", "observation 2"] for pid in sess.participants},
                    indent=2,
                )
                _tags = {"phase": "pp_impression"}
                if call_tags:
                    _tags.update(call_tags)
                batch.append({
                    "system": IMPRESSION_SYSTEM_PROMPT,
                    "user": (
                        f"Read this conversation and note 1-2 specific behavioural observations "
                        f"about each speaker — how they communicate, what they reveal about their "
                        f"personality, their verbal habits, emotional cues, etc.\n\n"
                        f"{transcript}\n\n"
                        f"Output a JSON object with exactly these keys and string-array values:\n"
                        f"{example}"
                    ),
                    "max_tokens": 8192,
                    "tags": _tags,
                })
                dispatch.append(("impression", si))

        # ================================================================
        # Single batch LLM call
        # ================================================================
        if not batch:
            # Still do knowledge update even with no LLM prompts
            for sess in all_sessions:
                for pid in sess.participants:
                    if pid in agent_map:
                        agent_map[pid].knowledge.update_from_session(sess, pid)
            return

        log.info(
            "Unified finalize: sending %d prompts (NER + mood + impressions) in one batch",
            len(batch),
        )
        responses = self.llm_client.generate_batch(batch)

        # ================================================================
        # Route responses back
        # ================================================================
        for entry, raw in zip(dispatch, responses):
            kind = entry[0]

            if kind == "ner":
                si = entry[1]
                self._parse_ner_response(raw, all_sessions[si].turns)

            elif kind == "mood":
                pid, si = entry[1], entry[2]
                if pid in agent_map and raw:
                    mood_text = raw.strip().strip('"').strip("'").lower()
                    agent_map[pid].current_mood = mood_text
                    sess = all_sessions[si]
                    feeling = KnownFeeling(
                        feeling_id=f"{pid}_feel_{len(agent_map[pid].knowledge.feelings):06d}",
                        content=mood_text,
                        trigger_agent="self",
                        session_id=sess.session_id,
                        timestamp=sess.end_time,
                    )
                    agent_map[pid].knowledge.add_feeling(feeling)

            elif kind == "impression":
                si = entry[1]
                sess = all_sessions[si]
                try:
                    result = parse_llm_json(raw)
                    impressions = {
                        k: [str(obs) for obs in v if obs]
                        for k, v in result.items()
                        if k in sess.participants and isinstance(v, list)
                    }
                except (ValueError, TypeError, KeyError, AttributeError):
                    impressions = {}

                if len(sess.participants) == 2:
                    a_id, b_id = sess.participants
                    if a_id in impressions and b_id in agent_map:
                        agent_map[b_id].knowledge.update_person_impression(
                            a_id, impressions[a_id]
                        )
                    if b_id in impressions and a_id in agent_map:
                        agent_map[a_id].knowledge.update_person_impression(
                            b_id, impressions[b_id]
                        )

        # ================================================================
        # Knowledge update (in-memory, after NER results are parsed)
        # ================================================================
        for sess in all_sessions:
            for pid in sess.participants:
                if pid in agent_map:
                    agent_map[pid].knowledge.update_from_session(sess, pid)

    @staticmethod
    def _build_group_context(
        speaker: EntityAgent,
        others: List[EntityAgent],
        event: Optional[WorldEvent],
        recent_turns: List[DialogueTurn],
        context_prefix: str = "",
        modality: str = "face_to_face",
        no_token_limit: bool = False,
    ) -> tuple:
        """Build system and user prompts for a group conversation turn."""
        import re as _re
        from MASim.utils.tokens import count_tokens, truncate_to_tokens

        system_prompt = speaker._build_system_prompt(modality=modality)

        other_names = ", ".join(a.persona.name for a in others)
        parts = []
        if context_prefix:
            parts.append(context_prefix)
        parts.append(f"You're in a group conversation with: {other_names}.")

        # Current emotional state
        if speaker.current_mood:
            parts.append(f"Right now you're feeling: {speaker.current_mood}")

        # Behavioural impressions of each group member
        for other in others:
            imp = speaker.knowledge.impression_for_prompt(other.agent_id)
            if imp:
                parts.append(f"\nHow you see {other.persona.name}:\n{imp}")

        if event:
            parts.append(f"\nSomething you all know about: {event.content}")

        # Entity-targeted memory retrieval
        entity_hints = [a.persona.name for a in others]
        if event:
            entity_hints += _re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', event.content)[:6]
        for turn in recent_turns[-3:]:
            entity_hints += turn.entities_mentioned[:3]

        past_summary = speaker.knowledge.summarise_for_context(entity_hints, 8)
        if past_summary:
            parts.append(
                "\nRelevant things you remember (use sparingly and only if natural):\n"
                + past_summary
            )

        if recent_turns:
            parts.append("")
            all_agents = [speaker] + others
            name_map = {a.agent_id: a.persona.name for a in all_agents}
            for turn in recent_turns[-12:]:
                name = name_map.get(turn.speaker_id, turn.speaker_id)
                parts.append(f"{name}: {turn.text}")

        # Open the next line for transcript completion
        parts.append(f"\n{speaker.persona.name}:")

        user_prompt = "\n".join(parts)

        if not no_token_limit:
            total = count_tokens(system_prompt + user_prompt)
            if total > 4096:
                user_prompt = truncate_to_tokens(user_prompt, 4096 - count_tokens(system_prompt) - 100)

        return system_prompt, user_prompt

    def _batch_extract_session_knowledge(self, turns: List[DialogueTurn]) -> None:
        """Extract entities AND memorable facts — thin wrapper around module-level function."""
        batch_extract_session_knowledge(self.llm_client, turns)


    def _batch_extract_impressions(
        self,
        turns: List[DialogueTurn],
        participants: List[str],
    ) -> Dict[str, List[str]]:
        """Extract behavioural impressions about each participant via a single LLM call.

        Returns a dict mapping agent_id -> list of 1-2 behavioural observations.
        These are then cross-distributed: observations about A go into B's memory, etc.
        """
        if not turns or not participants:
            return {}

        transcript = "\n".join(f"{t.speaker_id}: {t.text}" for t in turns)

        # Build an example output structure so the LLM knows the expected keys
        example = json.dumps(
            {pid: ["observation 1", "observation 2"] for pid in participants},
            indent=2,
        )
        prompt = (
            f"Read this conversation and note 1-2 specific behavioural observations "
            f"about each speaker — how they communicate, what they reveal about their "
            f"personality, their verbal habits, emotional cues, etc.\n\n"
            f"{transcript}\n\n"
            f"Output a JSON object with exactly these keys and string-array values:\n"
            f"{example}"
        )

        try:
            response = self.llm_client.generate(
                IMPRESSION_SYSTEM_PROMPT, prompt, max_tokens=32768,
                tags={"phase": "pp_impression"},
            )
            result = parse_llm_json(response)
            return {
                k: [str(obs) for obs in v if obs]
                for k, v in result.items()
                if k in participants and isinstance(v, list)
            }
        except (ValueError, TypeError, KeyError, AttributeError) as e:
            log.warning("Impression extraction failed: %s", e)
            return {}

    def _update_agent_mood(self, agent: "EntityAgent", session: "Session") -> None:
        """Update an agent's emotional state after a session via a small LLM call.

        Uses only the last 6 turns for efficiency. The updated mood is injected
        into the user prompt of the agent's next conversation turn.
        """
        relevant_turns = [t for t in session.turns[-6:] if t.text]
        if not relevant_turns:
            return
        transcript = "\n".join(
            f"{t.speaker_id}: {t.text}" for t in relevant_turns
        )
        try:
            mood = self.llm_client.generate(
                "Assess how a person feels after a conversation. Output one short phrase only — no explanation, no punctuation.",
                f"After this conversation, how is {agent.persona.name} feeling?\n\n{transcript}\n\nOne phrase:",
                max_tokens=32768,
                tags={"phase": "pp_mood"},
            )
            agent.current_mood = mood.strip().strip('"').strip("'").lower()
            log.debug("Mood update for %s: %s", agent.agent_id, agent.current_mood)
        except Exception as e:
            log.debug("Mood update failed for %s: %s", agent.agent_id, e)


def _extract_entities_regex(text: str) -> List[str]:
    """Fallback regex NER: extract multi-word capitalized phrases (proper nouns).

    Only matches sequences of 2+ capitalized words mid-sentence,
    filtering out common English words that happen to be capitalized.
    """
    stopwords = {
        "I", "The", "A", "An", "It", "If", "But", "And", "Or", "So", "Just",
        "You", "Your", "He", "She", "We", "They", "My", "His", "Her", "Our",
        "That", "This", "There", "Here", "How", "What", "When", "Where", "Why",
        "Can", "Could", "Would", "Should", "Have", "Has", "Had", "Do", "Does",
        "Not", "No", "Yes", "Oh", "Ah", "Well", "Now", "Then", "Also", "Very",
        "Really", "Maybe", "Perhaps", "Honestly", "Seriously", "Anyway",
        "Thanks", "Thank", "Please", "Sorry", "Sure", "Right", "Okay",
        "Go", "Come", "Get", "Let", "Make", "Take", "Keep", "Find", "Tell",
        "After", "Before", "Because", "Since", "Although", "However",
        "Count", "Perfect", "Makes", "Coffee", "Numbers",
    }
    # Match multi-word proper nouns (e.g. "Portland Rain", "Dr. Whitaker")
    pattern = r'\b(?:Dr\.|Prof\.|Mr\.|Ms\.|Mrs\.)?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)'
    matches = re.findall(pattern, text)
    entities = [m for m in matches if m.split()[0] not in stopwords]
    return list(set(entities))


def _is_conversation_ending(text: str) -> bool:
    """Heuristic check if a turn signals conversation end."""
    ending_signals = [
        "goodbye", "bye", "see you", "talk later", "gotta go",
        "nice talking", "catch you later", "take care",
        "ttyl", "g2g", "cya", "nite",
    ]
    lower = text.lower()
    return any(signal in lower for signal in ending_signals)
