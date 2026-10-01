from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from memarena.text_cleaning import clean_llm_text

from .types import MessageEntry, QAItem


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_messages_jsonl(path: Path, limit: Optional[int] = None) -> List[MessageEntry]:
    """
    Load messages from either:
    1) transcript jsonl (one json object per line), or
    2) EverMemBench-style groupchat json ({"dialogues": {...}}).
    """
    suffix = path.suffix.lower()
    if suffix == ".json":
        return load_groupchat_json(path, limit_days=None, limit_messages=limit)

    # Heuristic fallback: if first non-space char is "{" and file has top-level dialogues, parse as groupchat json.
    first = path.read_text(encoding="utf-8", errors="ignore")[:2048].lstrip()
    if first.startswith("{"):
        try:
            obj = _read_json(path)
            if isinstance(obj, dict) and isinstance(obj.get("dialogues"), dict):
                return load_groupchat_json(path, limit_days=None, limit_messages=limit)
        except Exception:
            pass

    rows: List[MessageEntry] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            rows.append(canonicalize_message(obj, source="jsonl", fallback_id=f"jsonl_{i:08d}"))
            if limit is not None and len(rows) >= max(0, int(limit)):
                break
    return rows


def load_groupchat_json(
    path: Path,
    limit_days: Optional[int] = None,
    limit_messages: Optional[int] = None,
) -> List[MessageEntry]:
    data = _read_json(path)
    if not isinstance(data, dict) or not isinstance(data.get("dialogues"), dict):
        raise ValueError(f"Unsupported groupchat json format: {path}")

    dialogues = data["dialogues"]
    dates = sorted([d for d in dialogues.keys() if isinstance(d, str)])
    if limit_days is not None:
        dates = dates[: max(0, int(limit_days))]

    out: List[MessageEntry] = []
    seq = 0

    for date_str in dates:
        groups = dialogues.get(date_str)
        if not isinstance(groups, dict):
            continue

        # keep deterministic order
        for group_name in sorted(groups.keys()):
            messages = groups.get(group_name)
            if not isinstance(messages, list):
                continue
            for msg in messages:
                if not isinstance(msg, dict):
                    continue

                raw = {
                    "msg_id": msg.get("msg_id") or f"{date_str}_{group_name}_{seq}",
                    "occur_ts": _parse_groupchat_time(msg.get("time"), date_str),
                    "deliver_ts": _parse_groupchat_time(msg.get("time"), date_str),
                    "thread_id": str(group_name),
                    "speaker": {"user_id": msg.get("speaker_id")},
                    "text": msg.get("dialogue") or msg.get("text") or msg.get("content") or "",
                    "meta": {
                        "source": "groupchat_json",
                        "speaker_name": msg.get("speaker"),
                        "date": date_str,
                    },
                }
                out.append(canonicalize_message(raw, source="groupchat_json", fallback_id=f"group_{seq:08d}"))
                seq += 1
                if limit_messages is not None and len(out) >= max(0, int(limit_messages)):
                    return out

    return out


def canonicalize_message(raw: dict, source: str, fallback_id: str = "") -> MessageEntry:
    speaker = raw.get("speaker") if isinstance(raw.get("speaker"), dict) else {}
    occur_ts = str(raw.get("occur_ts") or raw.get("time") or "")
    if not occur_ts:
        occur_ts = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    deliver_ts = str(raw.get("deliver_ts") or occur_ts)
    msg_id = str(raw.get("msg_id") or raw.get("id") or fallback_id)

    meta = dict(raw.get("meta") or {})
    meta.setdefault("source", source)

    return MessageEntry(
        msg_id=msg_id,
        occur_ts=occur_ts,
        deliver_ts=deliver_ts,
        user_id=_opt_int(speaker.get("user_id") if isinstance(speaker, dict) else raw.get("user_id")),
        thread_id=str(raw.get("thread_id") or raw.get("group") or ""),
        text=str(raw.get("text") or raw.get("dialogue") or raw.get("content") or ""),
        refs=dict(raw.get("refs") or {}),
        meta=meta,
    )


def load_qa(path: Path, limit: Optional[int] = None) -> List[QAItem]:
    data = _read_json(path)

    # Prefer qars if present, but allow mixed payloads by parser fallback.
    if isinstance(data, dict) and isinstance(data.get("qars"), list):
        raw_items = data["qars"]
    elif isinstance(data, dict) and isinstance(data.get("questions"), list):
        raw_items = data["questions"]
    elif isinstance(data, list):
        raw_items = data
    else:
        raise ValueError(f"Unsupported QA format: {path}")

    if limit is not None:
        raw_items = raw_items[: max(0, int(limit))]

    out: List[QAItem] = []
    for i, item in enumerate(raw_items):
        if not isinstance(item, dict):
            raise ValueError(f"QA item at index {i} is not an object")
        if "Q" in item or "A" in item:
            out.append(_parse_qars_item(item, i))
        else:
            out.append(_parse_legacy_item(item, i))
    return out


def validate_qa_schema(items: List[QAItem]) -> List[str]:
    errs: List[str] = []
    seen = set()
    for i, q in enumerate(items):
        if not q.question_id:
            errs.append(f"[{i}] missing question_id")
        if q.question_id in seen:
            errs.append(f"[{i}] duplicated question_id: {q.question_id}")
        seen.add(q.question_id)

        if not q.question.strip():
            errs.append(f"[{q.question_id}] empty question")
        if q.question_type not in {"open_ended", "multiple_choice"}:
            errs.append(f"[{q.question_id}] invalid question_type: {q.question_type}")

        if q.question_type == "multiple_choice":
            if not q.options:
                errs.append(f"[{q.question_id}] multiple_choice missing options")
            if not (q.correct_option or _infer_option_from_answer(q.answer)):
                errs.append(f"[{q.question_id}] multiple_choice missing correct option")
    return errs


def _parse_qars_item(item: Dict[str, Any], idx: int) -> QAItem:
    qid = str(item.get("id") or f"q_{idx:04d}")
    question = clean_llm_text(item.get("Q") or item.get("question") or "")
    answer = str(item.get("A") or item.get("answer") or "").strip()
    options_raw = item.get("options")

    if not question:
        raise ValueError(f"QA item missing Q at index {idx}")

    raw_meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    metadata: Dict[str, Any] = dict(raw_meta)
    for k, v in item.items():
        if k in {"id", "Q", "A", "question", "answer", "options", "meta"}:
            continue
        metadata[k] = v

    if isinstance(options_raw, dict) and options_raw:
        options = [f"{k}. {v}" for k, v in sorted(options_raw.items(), key=lambda kv: kv[0])]
        # Find the option letter whose value matches the answer text.
        correct_letter = None
        ans_lower = answer.strip().lower()
        for k, v in options_raw.items():
            if str(v).strip().lower() == ans_lower:
                correct_letter = k.strip().upper()[:1]
                break
        if correct_letter is None:
            correct_letter = answer.strip().upper()[:1] if answer else None
        return QAItem(
            question_id=qid,
            question=question,
            answer=answer,
            question_type="multiple_choice",
            options=options,
            correct_option=correct_letter,
            metadata=metadata,
        )
    if isinstance(options_raw, list) and options_raw:
        return QAItem(
            question_id=qid,
            question=question,
            answer=answer,
            question_type="multiple_choice",
            options=[str(x) for x in options_raw],
            correct_option=_infer_option_from_answer(answer),
            metadata=metadata,
        )

    return QAItem(
        question_id=qid,
        question=question,
        answer=answer,
        question_type="open_ended",
        metadata=metadata,
    )


def _parse_legacy_item(item: Dict[str, Any], idx: int) -> QAItem:
    qid = str(item.get("question_id") or item.get("id") or f"q_{idx:04d}")
    question = clean_llm_text(item.get("question") or item.get("Q") or "")
    answer = str(item.get("answer") or item.get("A") or "").strip()
    options = item.get("options")
    correct_option = item.get("correct_option")

    if not question:
        raise ValueError(f"QA item missing question at index {idx}")

    if isinstance(options, list) and options:
        return QAItem(
            question_id=qid,
            question=question,
            answer=answer,
            question_type="multiple_choice",
            options=[str(x) for x in options],
            correct_option=(str(correct_option).strip().upper()[:1] if correct_option else _infer_option_from_answer(answer)),
            metadata={k: v for k, v in item.items() if k not in {"question_id", "id", "question", "Q", "answer", "A", "options", "correct_option"}},
        )

    return QAItem(
        question_id=qid,
        question=question,
        answer=answer,
        question_type="open_ended",
        metadata={k: v for k, v in item.items() if k not in {"question_id", "id", "question", "Q", "answer", "A", "options", "correct_option"}},
    )


def _infer_option_from_answer(answer: str) -> Optional[str]:
    s = (answer or "").strip().upper()
    if s[:1] in {"A", "B", "C", "D"}:
        return s[:1]
    return None


def _opt_int(v: Any) -> Optional[int]:
    try:
        return int(v)
    except Exception:
        return None


def _parse_groupchat_time(v: Any, date_str: str) -> str:
    s = str(v or "").strip()
    if not s:
        return f"{date_str}T00:00:00Z"
    # Normalize common formats
    s = s.replace(" ", "T")
    if s.endswith("Z") or "+" in s[10:]:
        return s
    # likely naive timestamp
    return s + "Z"


def message_to_context_line(m: MessageEntry) -> str:
    who = f"u_{m.user_id}" if m.user_id is not None else "u_unknown"
    ts = m.deliver_ts or m.occur_ts
    return f"[{ts}][{m.thread_id}][{who}] {m.text}"


def messages_to_context_block(messages: Iterable[MessageEntry]) -> str:
    return "\n".join(message_to_context_line(m) for m in messages)
