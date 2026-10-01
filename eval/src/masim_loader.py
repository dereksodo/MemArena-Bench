"""
masim_loader.py — bridge from MASim run-directory output to the eval pipeline's
MessageEntry / QAItem types.

Usage:
    from eval.src.masim_loader import load_masim_messages, load_masim_qa

    messages = load_masim_messages(Path("MASim/runs/my_run/"))
    qas      = load_masim_qa(Path("MASim/runs/my_run/"), dimensions=["d4_permission", "d7_qa"])
"""
from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from memarena.text_cleaning import clean_llm_text

from .types import MessageEntry, QAItem


def corpus_sessions_path(run_dir: Path) -> Path:
    """corpus_sessions.jsonl, or the .jsonl.gz the released dataset ships."""
    for name in ("corpus_sessions.jsonl", "corpus_sessions.jsonl.gz"):
        if (run_dir / name).exists():
            return run_dir / name
    raise FileNotFoundError(f"neither corpus_sessions.jsonl nor corpus_sessions.jsonl.gz found in {run_dir}")


def open_corpus_sessions(run_dir: Path):
    path = corpus_sessions_path(run_dir)
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open("r", encoding="utf-8")

# Treat sim-time 0.0 as this reference date so ordering is stable.
_SIM_BASE = datetime(2025, 1, 1, tzinfo=timezone.utc)

# Ordered list of every dimension file we try to load by default.
ALL_DIMENSIONS = [
    "d1_conflict",
    "d2_anaphora",
    "d3_confabulation",
    "d4_permission",
    "d5_cloze",
    "d6_metadata",
    "d7_qa",
    "d8_temporal",
    "d9_negation",
    "d10_counterfactual",
    "d11_exception",
]

# Fields we try in order when looking for a plain-text answer in a ground_truth dict.
_ANSWER_FIELDS = [
    "answer",
    "correct_answer",
    "resolved_text",
    "value",
    "fact",
    "response",
    "event",
    "claim",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _float_ts_to_iso(ts: float) -> str:
    """Simulation float (days since epoch 0) → ISO 8601 string."""
    dt = _SIM_BASE + timedelta(days=float(ts))
    return dt.isoformat()


def _slug_to_display_name(slug: str) -> str:
    """'maya_chen' → 'Maya Chen', '__assistant__' → 'AI Assistant'."""
    if slug.startswith("assistant_") or slug == "__assistant__":
        return "AI Assistant"
    return " ".join(p.capitalize() for p in slug.split("_"))


def _load_agent_index(run_dir: Path) -> Dict[str, int]:
    """Build {agent_slug: small_int} from agents_personas.jsonl for stable user_id values."""
    p = run_dir / "agents_personas.jsonl"
    if not p.exists():
        return {}
    index: Dict[str, int] = {}
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                slug = str(obj.get("agent_id", "")).strip()
                if slug:
                    index[slug] = i
            except Exception:
                pass
    return index


def _extract_answer(gt: dict, dim: str) -> str:
    """Best-effort plain-text answer extraction from a ground_truth dict."""
    for field in _ANSWER_FIELDS:
        v = gt.get(field)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # Last resort: compact JSON of the entire dict.
    return json.dumps(gt, ensure_ascii=False)


def _resolve_turn_text(
    corpus_sessions: Dict[str, dict],
    session_id: str,
    turn_id: Optional[str] = None,
    speaker_id: Optional[str] = None,
) -> str:
    """Look up turn text from corpus_sessions dict.

    If turn_id is given, find that exact turn. If speaker_id is given
    instead, return the first turn by that speaker. Falls back to
    concatenating first few turns in the session.
    """
    sess = corpus_sessions.get(session_id)
    if not sess:
        return ""
    turns = sess.get("turns") or []
    if turn_id:
        for t in turns:
            if t.get("turn_id") == turn_id:
                return str(t.get("text", "")).strip()
    if speaker_id:
        for t in turns:
            if t.get("speaker_id") == speaker_id:
                return str(t.get("text", "")).strip()
    # Fallback: concatenate all turns (injected conflict turns are appended at
    # the end, so we cannot truncate to just the first few turns here).
    texts = [str(t.get("text", "")).strip() for t in turns if t.get("text")]
    return " ".join(texts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_masim_messages(run_dir: Path) -> List[MessageEntry]:
    """
    Load corpus_sessions.jsonl from a MASim run directory and convert every
    DialogueTurn into a MessageEntry suitable for the eval pipeline.

    Text format per turn:
        [Location: Office | Modality: face to face]
        [Maya Chen]: actual dialogue text here
    """

    slug_index = _load_agent_index(run_dir)
    messages: List[MessageEntry] = []

    with open_corpus_sessions(run_dir) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            session = json.loads(line)
            session_id = str(session.get("session_id", ""))
            modality = str(session.get("modality", "face_to_face")).replace("_", " ")
            location_id = str(session.get("location_id", "")).replace("loc_", "").replace("_", " ").title()
            group_id = str(session.get("group_id", "")).replace("grp_", "").replace("_", " ").title()
            participants = session.get("participants", [])

            # Build a short context prefix shared by all turns in this session.
            meta_parts: List[str] = []
            if location_id:
                meta_parts.append(f"Location: {location_id}")
            if group_id:
                meta_parts.append(f"Group: {group_id}")
            meta_parts.append(f"Modality: {modality}")
            meta_prefix = " | ".join(meta_parts)

            for turn in session.get("turns", []):
                speaker_slug = str(turn.get("speaker_id", "unknown"))
                speaker_name = _slug_to_display_name(speaker_slug)
                ts = float(turn.get("timestamp", 0.0))
                occur_ts = _float_ts_to_iso(ts)
                text_body = str(turn.get("text", ""))

                # "[Location: … | Group: … | Modality: …]\n[Maya Chen]: …"
                formatted_text = f"[{meta_prefix}]\n[{speaker_name}]: {text_body}"

                messages.append(
                    MessageEntry(
                        msg_id=str(turn.get("turn_id", f"{session_id}_{len(messages)}")),
                        occur_ts=occur_ts,
                        deliver_ts=occur_ts,
                        user_id=slug_index.get(speaker_slug),
                        thread_id=session_id,
                        text=formatted_text,
                        refs={"session_id": session_id, "speaker_slug": speaker_slug},
                        meta={
                            "source": "masim",
                            "session_id": session_id,
                            "speaker_slug": speaker_slug,
                            "speaker_name": speaker_name,
                            "modality": modality,
                            "location_id": location_id,
                            "group_id": group_id,
                            "participants": participants,
                        },
                    )
                )

    return messages


def load_corpus_sessions_dict(run_dir: Path) -> Dict[str, dict]:
    """Load corpus_sessions.jsonl keyed by session_id.

    Returns {session_id: session_dict} for use by evidence-grounded scoring.
    """
    index: Dict[str, dict] = {}
    with open_corpus_sessions(run_dir) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                sess = json.loads(line)
                sid = sess.get("session_id", "")
                if sid:
                    index[sid] = sess
            except Exception:
                pass
    return index


def load_masim_qa(
    run_dir: Path,
    dimensions: Optional[List[str]] = None,
    corpus_sessions: Optional[Dict[str, dict]] = None,
) -> List[QAItem]:
    """
    Load eval_instances/d*.jsonl from a MASim run directory and convert every
    EvalInstance into a QAItem suitable for the eval pipeline.

    The ground_truth is mapped per dimension:
    - D4 permission: expected_answer_mode = "answer" | "abstain"
    - D11 exception/impostor: expected_answer_mode = "abstain"
    - All others: open_ended with token-F1 scoring against extracted answer string

    If corpus_sessions is provided, it is used to resolve actual turn text
    for dimensions that previously only stored session/turn ID references.
    """
    eval_dir = run_dir / "eval_instances"
    if not eval_dir.exists():
        raise FileNotFoundError(f"eval_instances/ not found in {run_dir}")

    dims = dimensions if dimensions else ALL_DIMENSIONS
    cs = corpus_sessions or {}
    items: List[QAItem] = []

    for dim in dims:
        f = eval_dir / f"{dim}.jsonl"
        if not f.exists():
            continue
        with f.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    instance = json.loads(line)
                    qas = _instance_to_qa(instance, dim, cs)
                    items.extend(qas)
                except Exception as exc:
                    # Skip malformed instances; don't abort the whole load.
                    import warnings
                    warnings.warn(f"masim_loader: skipping malformed {dim} instance: {exc}")

    _attach_owner_contacts(items, run_dir)
    return items


# Closeness labels of the Dunbar layers (1 = innermost) as the owner's contact list
# names them; layers 1-2 are the "friends" a friends_only fact may reach.
CONTACT_LABELS = {1: "close friend", 2: "friend", 3: "acquaintance"}


def owner_contacts(run_dir: Path) -> Dict[str, Dict[str, str]]:
    """``{agent: {contact display name: closeness label}}`` from the run's social graph.

    The layers come from ``graph_edges.jsonl`` exactly as the D6 generator assigns
    friends_only access (``MASim.ground_truth.d4_permission.dunbar_layers``).
    Empty when the run directory has no graph.
    """
    import gzip

    from MASim.ground_truth.d4_permission import dunbar_layers
    from eval.src.scoring_core import _slug_to_name

    for path in (run_dir / "graph_edges.jsonl", run_dir / "graph_edges.jsonl.gz"):
        if path.exists():
            opener = gzip.open if path.suffix == ".gz" else open
            with opener(path, "rt", encoding="utf-8") as fh:
                edges = [json.loads(line) for line in fh if line.strip()]
            break
    else:
        return {}
    return {
        agent: {_slug_to_name(n): CONTACT_LABELS.get(layer, "acquaintance")
                for n, layer in sorted(nbrs.items())}
        for agent, nbrs in dunbar_layers(edges).items()
    }


def _attach_owner_contacts(items: List[QAItem], run_dir: Path) -> None:
    """D6 items: ``metadata["owner_contacts"]``, the fact owner's contact list.

    A personal assistant knows its owner's contacts; the D6 answer prompt gives the
    list to every backend alike (``AnswerConfig.d6_owner_contacts``). It is a fact
    about the owner, not a sharing policy.
    """
    d6 = [qa for qa in items if (qa.metadata or {}).get("dimension") == "d4_permission"]
    if not d6:
        return
    contacts = owner_contacts(run_dir)
    for qa in d6:
        owner = str(qa.metadata.get("ego_agent_id") or "")
        if owner in contacts:
            qa.metadata["owner_contacts"] = contacts[owner]


def dump_masim_transcript(messages: List[MessageEntry], out_path: Path) -> None:
    """
    Write messages as JSONL in a format readable by eval/src/io.py's
    load_messages_jsonl().  Each line is a dict matching the canonicalize_message
    contract.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for m in messages:
            row = {
                "msg_id": m.msg_id,
                "occur_ts": m.occur_ts,
                "deliver_ts": m.deliver_ts,
                "user_id": m.user_id,
                "thread_id": m.thread_id,
                "text": m.text,
                "refs": m.refs,
                "meta": m.meta,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def dump_masim_qa(qas: List[QAItem], out_path: Path) -> None:
    """Write QAItems as {"qars": [...]} JSON readable by eval/src/io.py's load_qa()."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "qars": [
            {
                "id": qa.question_id,
                "Q": qa.question,
                "A": qa.answer,
                "options": (
                    {chr(65 + i): o for i, o in enumerate(qa.options)}
                    if qa.options
                    else None
                ),
                "meta": qa.metadata,
            }
            for qa in qas
        ]
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Per-dimension instance → QAItem conversion
# ---------------------------------------------------------------------------

def _instance_to_qa(instance: dict, dim: str, corpus_sessions: Optional[Dict[str, dict]] = None) -> List[QAItem]:
    gt = instance.get("ground_truth") or {}
    instance_id = str(instance.get("instance_id", ""))
    query = clean_llm_text(instance.get("query", ""))
    difficulty = str(instance.get("difficulty", "medium"))
    inst_meta = instance.get("metadata") or {}
    cs = corpus_sessions or {}

    if not instance_id or not query:
        return []

    # Extract ego_context / anchor early so D5 can use it
    ego_context = instance.get("ego_context") or []
    anchor_msg_id: Optional[str] = None
    if ego_context:
        last_turn = ego_context[-1]
        anchor_msg_id = str(last_turn.get("turn_id") or "").strip() or None

    answer = ""
    expected_answer_mode: Optional[str] = None
    policy_expected: Optional[str] = None

    # ------------------------------------------------------------------
    if dim == "d1_conflict":
        # Prefer extracted key details over raw turn text
        orig_detail = gt.get("original_detail", "")
        changed_detail = gt.get("changed_detail", "")
        if orig_detail and changed_detail:
            source_name = _slug_to_display_name(gt.get("source_agent", ""))
            conflict_name = _slug_to_display_name(gt.get("conflicting_agent", ""))
            answer = (
                f"{source_name} said {orig_detail}, "
                f"but {conflict_name} said {changed_detail}."
            )
        # Fallback: raw turn text (backward compat with old runs)
        elif gt.get("original_fact") and gt.get("contradictory_fact"):
            answer = f"Original claim: {gt['original_fact']} Contradictory claim: {gt['contradictory_fact']}"
        elif gt.get("original_fact"):
            answer = gt["original_fact"]
        elif cs:
            # Fallback: resolve full session text when facts are absent.
            # Do NOT filter by speaker_id — the conflicting agent may be
            # "__assistant__" which has no speaker entry in the corpus.
            orig_text = _resolve_turn_text(cs, gt.get("original_session", ""),
                                           speaker_id=gt.get("source_agent"))
            # For conflicting session: grab all turns (no speaker filter) to
            # capture the injected contradictory turn regardless of speaker.
            contra_text = _resolve_turn_text(cs, gt.get("conflicting_session", ""))
            if orig_text and contra_text:
                answer = f"Original claim: {orig_text} Contradictory claim: {contra_text}"
            elif orig_text or contra_text:
                answer = (orig_text or contra_text).strip()
            else:
                answer = _extract_answer(gt, dim)
        else:
            answer = _extract_answer(gt, dim)
        expected_answer_mode = "answer"

    elif dim == "d2_anaphora":
        # D2 gold: antecedent text is the key answer
        ant_text = gt.get("antecedent_text", "")
        if ant_text:
            referent = str(gt.get("referent") or "").strip()
            answer = f"{referent}: {ant_text}" if referent else ant_text
        elif cs:
            resolved = _resolve_turn_text(cs, gt.get("antecedent_session", ""),
                                          turn_id=gt.get("antecedent_turn"))
            referent = str(gt.get("referent") or "").strip()
            answer = f"{referent}: {resolved}" if (referent and resolved) else (resolved or _extract_answer(gt, dim))
        else:
            referent = str(gt.get("referent") or "").strip()
            answer = referent if referent else _extract_answer(gt, dim)

    elif dim == "d3_confabulation":
        expected_resp = str(gt.get("expected_response") or "").strip().lower()
        if expected_resp == "abstain":
            answer = "abstain"
            expected_answer_mode = "abstain"
        else:
            correct_fact = gt.get("correct_fact", "")
            if correct_fact:
                answer = correct_fact
            elif cs:
                resolved = _resolve_turn_text(cs, gt.get("source_session", ""),
                                              turn_id=gt.get("source_turn"))
                answer = resolved if resolved else _extract_answer(gt, dim)
            else:
                answer = _extract_answer(gt, dim)
            expected_answer_mode = "answer"

    elif dim == "d4_permission":
        fact = str(gt.get("fact") or "")
        expected_disclosure = bool(gt.get("expected_disclosure", True))
        if expected_disclosure:
            answer = fact
            # All ALLOW instances: score deterministically as "disclose".
            # We only care about whether the model privacy-refused; the
            # factual content is judged elsewhere via D7 QA. Avoiding the
            # LLM-judge fallback here makes D6/permission scoring fully
            # reproducible from pre-registered keyword lists.
            expected_answer_mode = "disclose"
            policy_expected = "ALLOW"
        else:
            # Model should decline/withhold — gold answer records what was withheld.
            answer = fact
            expected_answer_mode = "abstain"
            policy_expected = "DENY_NO_ACCESS"

    elif dim == "d5_cloze":
        # D5 cloze: keep as a single QA item per instance.
        # The query already contains the cloze passage + all options.
        # Context should include all corpus EXCEPT the tested session
        # (thread_id = session_id), so the model recalls from surrounding
        # conversations but cannot simply look up the answer verbatim.
        blanks = gt.get("blanks") or {}
        choices = gt.get("choices") or {}
        n_blanks = len(blanks)
        evidence_session_id = gt.get("session_id", "")

        # New single-blank format: flat answer_letter/blank_word/options
        if not blanks and gt.get("answer_letter"):
            answer_letter = gt["answer_letter"]
            blank_word = gt.get("blank_word", "")
            options = gt.get("options", [])
            gold_answer = answer_letter
            n_blanks = 1
            blanks = {"blank_1": blank_word}
            choices = {"blank_1": {
                "answer_letter": answer_letter,
                "blank_word": blank_word,
                "options": options,
            }}
        else:
            # Legacy multi-blank format: "1C 2A 3B ..."
            answer_parts = []
            for label in sorted(blanks.keys(), key=lambda k: int(k.split("_")[1])):
                blank_num = label.split("_")[1]
                choice_data = choices.get(label, {})
                answer_letter = choice_data.get("answer_letter", "A")
                answer_parts.append(f"{blank_num}{answer_letter}")
            gold_answer = " ".join(answer_parts)

        cloze_metadata: Dict = {
            "task_family": dim,
            "dimension": dim,
            "difficulty": difficulty,
            "source": "masim",
            "n_blanks": n_blanks,
            "blanks": blanks,
            "choices": choices,
        }
        # Exclude the tested session from context (thread_id = session_id)
        if evidence_session_id:
            cloze_metadata["exclude_thread_ids"] = [evidence_session_id]
        if anchor_msg_id:
            cloze_metadata["anchor_msg_id"] = anchor_msg_id
        if ego_context:
            cloze_metadata["ego_context_turn_ids"] = [
                str(t.get("turn_id", "")) for t in ego_context
            ]
        for k, v in inst_meta.items():
            cloze_metadata.setdefault(f"instance_{k}", v)

        cloze_metadata["gold_ground_truth"] = gt
        return [QAItem(
            question_id=instance_id,
            question=query,  # original self-contained query from D5 generator
            answer=gold_answer,
            question_type="cloze",
            metadata=cloze_metadata,
        )]

    elif dim == "d6_metadata":
        attr = gt.get("attribute", "")
        if attr == "context":
            src_turn = gt.get("source_turn", "")
            src_sess = gt.get("source_session", "")
            answer = f"See turn {src_turn} in session {src_sess}"
        else:
            answer = str(gt.get("value") or gt.get("attribute_value") or _extract_answer(gt, dim))

    elif dim == "d7_qa":
        ans_text = gt.get("answer", "")
        if ans_text:
            answer = ans_text
        elif cs:
            resolved = _resolve_turn_text(cs, gt.get("source_session", ""),
                                          turn_id=gt.get("source_turn"))
            answer = resolved if resolved else _extract_answer(gt, dim)
        else:
            answer = _extract_answer(gt, dim)

    elif dim == "d8_temporal":
        ans_text = gt.get("answer", "")
        if ans_text:
            answer = ans_text
        else:
            answer = _extract_answer(gt, dim)

    elif dim == "d9_negation":
        expected_resp = str(gt.get("expected_response") or "").strip().lower()
        if expected_resp == "no_knowledge":
            absent = gt.get("absent_agent", "")
            absent_name = _slug_to_display_name(absent) if absent else "the agent"
            answer = f"No, {absent_name} was not present in that conversation and would not know about it."
            expected_answer_mode = "answer"
        else:
            # Presence query: resolve actual turn text
            ans_text = gt.get("actual_fact", "") or gt.get("answer", "")
            if ans_text:
                answer = ans_text
            elif cs:
                resolved = _resolve_turn_text(cs, gt.get("actual_session", ""),
                                              turn_id=gt.get("actual_turn"))
                answer = resolved if resolved else _extract_answer(gt, dim)
            else:
                answer = _extract_answer(gt, dim)
            expected_answer_mode = "answer"

    elif dim == "d10_counterfactual":
        real_fact = gt.get("real_fact", "")
        if real_fact:
            altered = gt.get("altered_detail", "")
            answer = f"{real_fact} (altered: {altered})" if altered else real_fact
        elif cs:
            resolved = _resolve_turn_text(cs, gt.get("source_session", ""),
                                          turn_id=gt.get("source_turn"))
            altered = gt.get("altered_detail", "")
            if resolved:
                answer = f"{resolved} (altered: {altered})" if altered else resolved
            else:
                answer = _extract_answer(gt, dim)
        else:
            answer = _extract_answer(gt, dim)
        expected_answer_mode = "answer"

    elif dim == "d11_exception":
        is_anomalous = gt.get("is_anomalous", True)
        if is_anomalous:
            answer = "No"
        else:
            answer = "Yes"
        expected_answer_mode = "answer"

    else:
        answer = _extract_answer(gt, dim)

    metadata: Dict = {
        "task_family": dim,
        "dimension": dim,
        "difficulty": difficulty,
        "source": "masim",
    }
    if expected_answer_mode:
        metadata["expected_answer_mode"] = expected_answer_mode
    if policy_expected:
        metadata["policy_expected"] = policy_expected
    if anchor_msg_id:
        metadata["anchor_msg_id"] = anchor_msg_id
    if ego_context:
        metadata["ego_context_turn_ids"] = [
            str(t.get("turn_id", "")) for t in ego_context
        ]
    # Carry forward top-level instance fields needed for context reconstruction.
    for field in ("evidence_session_ids", "ego_agent_id", "context_tier",
                  "asker_agent_id", "answerer_agent_id"):
        val = instance.get(field)
        if val is not None:
            metadata.setdefault(field, val)
    # Carry forward any instance-level metadata (e.g., query_agent, target_agent).
    for k, v in inst_meta.items():
        metadata.setdefault(f"instance_{k}", v)

    # Scoring-only copy of the gold (the answering path reads metadata["ground_truth"]
    # to build context, so the gold must not go there).
    metadata["gold_ground_truth"] = gt
    return [QAItem(
        question_id=instance_id,
        question=query,
        answer=answer,
        question_type="open_ended",
        metadata=metadata,
    )]
