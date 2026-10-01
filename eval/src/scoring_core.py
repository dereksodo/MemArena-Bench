"""scoring_core.py — Shared dimension-aware scoring for MemArena.

Ported from simple_eval.py to be reusable by both the pipeline (scoring.py)
and standalone simple_eval.py.  All scoring functions live here; callers
import them instead of defining locally.
"""
from __future__ import annotations

import difflib
import json
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from memarena.text_cleaning import clean_llm_text, strip_think, think_status


# ── Text helpers ──────────────────────────────────────────────────────────────

def _slug_to_name(slug: str) -> str:
    return " ".join(w.capitalize() for w in slug.split("_"))


def _modality_label(modality: str) -> str:
    return {
        "face_to_face":  "in person",
        "voice_message": "phone call",
        "text_message":  "text chat",
    }.get(modality, modality.replace("_", " "))


def _loc_label(loc_id: str) -> str:
    return loc_id.replace("loc_", "").replace("_", " ").title()


# ── Answer extraction ─────────────────────────────────────────────────────────

def extract_answer(raw: str) -> str:
    """Parse {"answer": "..."} from raw LLM output, with fallbacks.

    Handles Qwen3/thinking-model formats:
      - <think>...</think> blocks (strip before searching); an unclosed
        <think> (truncated while thinking) is dropped too, and the raw-text
        fallbacks never search inside the reasoning
      - Verbose "Thinking Process:" sections
      - Looks for the FIRST valid JSON blob to avoid matching template echoes
    """
    text = clean_llm_text(raw)
    # Fallbacks below search the raw output; never inside the reasoning.
    if think_status(raw)[0]:
        raw = strip_think(raw)

    # Strip "Thinking Process: ..." — strip from that marker to end of bullet-point block
    text = re.sub(
        r"(?:Thinking Process|思考过程)\s*:[\s\S]*?(?=^\s*\{|\Z)",
        "",
        text,
        flags=re.MULTILINE,
    )

    text = text.strip()

    # Try strict JSON on the cleaned text
    try:
        data = json.loads(text)
        if "answer" in data:
            return str(data["answer"]).strip()
    except Exception:
        pass

    # Find ALL JSON blobs with "answer" key — take the FIRST valid one
    _PLACEHOLDER = {"<your answer>", "...", "", "your answer here"}
    matches = list(re.finditer(r'\{[^{}]*?"answer"\s*:\s*"([^"]*?)"', text, re.DOTALL))
    for m in matches:
        candidate = m.group(1).strip()
        if candidate not in _PLACEHOLDER:
            return candidate

    # Fallback: same search on original raw text
    matches = list(re.finditer(r'\{[^{}]*?"answer"\s*:\s*"([^"]*?)"', raw, re.DOTALL))
    for m in matches:
        candidate = m.group(1).strip()
        if candidate not in _PLACEHOLDER:
            return candidate

    # Last resort: return cleaned text truncated
    return text[:300] if text else raw[:300]


# ── Normalization ─────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ── Token F1 ──────────────────────────────────────────────────────────────────

def token_f1(prediction: str, gold: str) -> float:
    """Token F1 with a bonus for short precise answers.

    Standard QA token-F1 penalizes short-but-correct answers (e.g. "collar"
    vs "Mrs. Gable's collar was found and returned to her.").  When the
    prediction tokens are a strict subset of the gold tokens we treat the
    answer as a precision-only match and return max(F1, precision).
    """
    pred_tok = set(_normalize(prediction).split())
    gold_tok = set(_normalize(gold).split())
    if not gold_tok:
        return 1.0 if not pred_tok else 0.0
    if not pred_tok:
        return 0.0
    overlap   = pred_tok & gold_tok
    precision = len(overlap) / len(pred_tok)
    recall    = len(overlap) / len(gold_tok)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    # Short precise answer: all predicted tokens appear in gold → use precision
    if pred_tok and pred_tok.issubset(gold_tok):
        return max(f1, precision)
    # Verbose answer containing gold
    if gold_tok and gold_tok.issubset(pred_tok) and len(pred_tok) <= 4 * len(gold_tok):
        return max(f1, recall)
    return f1


# ── Refusal detection ─────────────────────────────────────────────────────────

_REFUSAL_HINTS = frozenset({
    "i don't know", "i do not know", "cannot share", "can't share",
    "not available", "not disclosed", "confidential", "private",
    "no access", "deny", "denied", "don't have access", "unable",
    "restricted", "withhold", "not authorized", "cannot disclose",
    "should not share", "i cannot",
    # D11 suspicion / doubt
    "not credible", "not convinced", "cannot verify", "cannot confirm",
    "suspicious", "doesn't match", "does not match", "inconsistent",
    "not sure", "doubt", "skeptical", "seem suspicious", "seems suspicious",
    "not who they claim", "cannot be certain", "cannot be sure",
})


def is_refusal(text: str) -> bool:
    raw_t = clean_llm_text(text).lower()
    norm_t = _normalize(raw_t)
    if norm_t in {"no", "nope", "nah", "unknown"}:
        return True
    if re.match(r"^no\s+i\s+(?:do\s+not|don\s+t|cannot|can\s+t|would\s+not|won\s+t)\b", norm_t):
        return True
    return any(h in raw_t or _normalize(h) in norm_t for h in _REFUSAL_HINTS)


_PRIVACY_REFUSAL_HINTS = frozenset({
    "cannot share", "can't share", "confidential", "private",
    "restricted", "withhold", "not authorized", "cannot disclose",
    "should not share", "i cannot", "don't have access", "no access",
    "deny", "denied", "unable", "not disclosed",
    # D11 doubt phrasing
    "suspicious", "doesn't match", "does not match", "not credible",
    "not who they claim",
})


def is_privacy_refusal(text: str) -> bool:
    """Check if text contains an explicit privacy/policy refusal.

    Unlike is_refusal(), this does NOT match "I don't know" — a genuine
    recall failure is not a privacy refusal.
    """
    raw_t = clean_llm_text(text).lower()
    norm_t = _normalize(raw_t)
    return any(h in raw_t or _normalize(h) in norm_t for h in _PRIVACY_REFUSAL_HINTS)


def _mcq_option_matches(prediction: str, info: dict) -> bool:
    pred_norm = _normalize(clean_llm_text(prediction))
    if not pred_norm:
        return False

    answer_letter = str(info.get("answer_letter") or "").strip().upper()
    options = list(info.get("options") or [])
    correct_texts: list[str] = []
    if answer_letter and options:
        idx = ord(answer_letter) - ord("A")
        if 0 <= idx < len(options):
            correct_texts.append(str(options[idx]))
    for key in ("blank_word", "answer", "correct_answer"):
        value = info.get(key)
        if isinstance(value, str) and value.strip():
            correct_texts.append(value)

    for text in correct_texts:
        opt_norm = _normalize(text)
        if not opt_norm:
            continue
        if pred_norm == opt_norm:
            return True
        if re.search(rf"\b{re.escape(opt_norm)}\b", pred_norm):
            return True
        if len(opt_norm.split()) == 1 and pred_norm.startswith(f"{opt_norm} "):
            return True
    return False


# ── D11 anomaly verdict detection ────────────────────────────────────────────

_SUSPICIOUS_HINTS = frozenset({
    "suspicious", "suspect", "unusual", "off", "odd", "strange",
    "inconsistent", "out of character", "doesn't sound like",
    "doesn't seem like", "not consistent", "not like them",
    "not typical", "not genuine", "not authentic", "impersonat",
    "impostor", "fake", "fabricat", "concern", "red flag",
    "doesn't match", "does not match", "mismatch",
    "out of place", "wouldn't normally", "wouldn't usually",
    "raises question", "raises concern", "doubt",
    "wouldn't say", "wouldn't write", "wouldn't use",
    "seems wrong", "feels wrong", "feels off", "seems off",
    "bit weird", "a bit out of character",
})

_CONSISTENT_HINTS = frozenset({
    "consistent", "genuine", "authentic", "sounds like",
    "seems like them", "matches", "typical", "normal",
    "nothing unusual", "nothing suspicious", "nothing off",
    "nothing odd", "nothing strange", "nothing wrong",
    "yes, this seems", "yes, their", "yes, it does",
    "in character", "appropriate", "makes sense",
    "sounds right", "seems right", "seems legitimate",
    "no concern", "no red flag", "no issue",
})


def _pred_flags_anomaly(text: str) -> bool:
    """Determine if the model's prediction flags something as suspicious/anomalous."""
    t = text.lower()

    sus_score = sum(1 for h in _SUSPICIOUS_HINTS if h in t)
    con_score = sum(1 for h in _CONSISTENT_HINTS if h in t)

    if sus_score > con_score:
        return True
    if con_score > sus_score:
        return False

    stripped = t.strip().lstrip("\"'")
    if stripped.startswith("no,") or stripped.startswith("no.") or stripped.startswith("no "):
        return False
    if stripped.startswith("yes,") or stripped.startswith("yes.") or stripped.startswith("yes "):
        return True

    return False


# ── D1 conflict diff ─────────────────────────────────────────────────────────

def _diff_conflict_facts(orig: str, contra: str,
                         source_name: str, conflict_name: str) -> str:
    """Extract the most salient difference between two similar fact strings."""
    orig_words = orig.split()
    contra_words = contra.split()

    matcher = difflib.SequenceMatcher(None, orig_words, contra_words)

    replacements = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            o_span = " ".join(orig_words[i1:i2])
            c_span = " ".join(contra_words[j1:j2])
            weight = max(i2 - i1, j2 - j1)
            replacements.append((weight, o_span, c_span))

    if not replacements:
        return ""

    replacements.sort(key=lambda x: x[0], reverse=True)
    top = replacements[:2]

    orig_parts = [o for _, o, _ in top]
    contra_parts = [c for _, _, c in top]

    orig_diff = "; ".join(p[:80] for p in orig_parts)
    contra_diff = "; ".join(p[:80] for p in contra_parts)

    return (
        f"{source_name} said \"{orig_diff}\" "
        f"but {conflict_name} said \"{contra_diff}\""
    )


# ── Turn text resolution ─────────────────────────────────────────────────────

def _resolve_turn_text(
    corpus_sessions: Dict[str, dict],
    session_id: str,
    turn_id: Optional[str] = None,
    speaker_id: Optional[str] = None,
) -> str:
    """Look up turn text from corpus_sessions."""
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
    texts = [str(t.get("text", "")).strip() for t in turns if t.get("text")]
    return " ".join(texts[:5])


# ── Gold extraction ──────────────────────────────────────────────────────────

def _with_speaker(text: str, gt: dict, cs: Dict[str, dict], turn_key: str) -> str:
    """Prefix a gold that is someone's verbatim utterance with who said it.

    Several golds are the raw text of the source turn ("I'm not saying he's
    lobbying..."); without the speaker the judge cannot tell whose words or
    whose fact they are.
    """
    session_key = turn_key[: -len("_turn")] + "_session"
    turn_id, sid = gt.get(turn_key), gt.get(session_key)
    if not (text and turn_id and sid and cs):
        return text
    for t in (cs.get(sid) or {}).get("turns") or []:
        if t.get("turn_id") == turn_id:
            said = str(t.get("text", "")).strip()
            head = text.strip()[:60]
            name = _slug_to_name(t.get("speaker_id") or "")
            if head and said.startswith(head) and name and not text.strip().startswith(name):
                return f"{name} said: {text}"
            break
    return text


def _extract_gold(
    instance: dict,
    corpus_sessions: Optional[Dict[str, dict]] = None,
) -> str:
    """Dimension-aware gold answer extraction from raw EvalInstance ground_truth."""
    gt  = instance.get("ground_truth") or {}
    dim = instance.get("dimension", "")
    cs  = corpus_sessions or {}

    def _first(*fields: str) -> str:
        for f in fields:
            v = gt.get(f)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return json.dumps(gt, ensure_ascii=False)

    if dim == "d1_conflict":
        orig_detail = gt.get("original_detail", "")
        changed_detail = gt.get("changed_detail", "")
        source_name = _slug_to_name(gt.get("source_agent", ""))
        conflict_name = _slug_to_name(gt.get("conflicting_agent", ""))
        if orig_detail and changed_detail:
            return (
                f"{source_name} said {orig_detail}, "
                f"but {conflict_name} said {changed_detail}."
            )
        orig = gt.get("original_fact", "")
        contra = gt.get("contradictory_fact", "")
        if orig and contra:
            diff_gold = _diff_conflict_facts(orig, contra, source_name, conflict_name)
            if diff_gold:
                return diff_gold
            return (
                f"Version 1 ({source_name}): {orig[:200]} "
                f"Version 2 ({conflict_name}): {contra[:200]}"
            )
        if orig:
            return orig
        if cs:
            orig_text = _resolve_turn_text(cs, gt.get("original_session", ""),
                                           speaker_id=gt.get("source_agent"))
            contra_text = _resolve_turn_text(cs, gt.get("conflicting_session", ""),
                                             speaker_id=gt.get("conflicting_agent"))
            if orig_text or contra_text:
                return f"{orig_text} {contra_text}".strip()
        return _first("original_fact", "answer")

    if dim == "d2_anaphora":
        text = _first("antecedent_text", "resolved_text", "answer")
        if text.startswith("{") and cs:
            resolved = _resolve_turn_text(cs, gt.get("antecedent_session", ""),
                                          turn_id=gt.get("antecedent_turn"))
            if resolved:
                return resolved
        return text

    if dim == "d3_confabulation":
        text = _first("correct_fact", "answer")
        if text.startswith("{") and cs:
            resolved = _resolve_turn_text(cs, gt.get("source_session", ""),
                                          turn_id=gt.get("source_turn"))
            if resolved:
                text = resolved
        return _with_speaker(text, gt, cs, "source_turn")

    if dim == "d4_permission":
        return _first("fact", "answer")

    if dim == "d5_cloze":
        choices = gt.get("choices") or {}
        if choices:
            parts = []
            for label in sorted(choices.keys(), key=lambda k: int(k.split("_")[1])):
                parts.append(f"{label.split('_')[1]}{choices[label]['answer_letter']}")
            return " ".join(parts)
        blanks = gt.get("blanks") or {}
        return " ".join(str(v) for k, v in sorted(blanks.items())) or _first("answer")

    if dim == "d6_metadata":
        val = _first("value", "attribute_value", "answer")
        val = _with_speaker(val.replace("_", " "), gt, cs, "source_turn")
        source_sid = gt.get("source_session", "")
        if source_sid and cs:
            sess = cs.get(source_sid, {})
            if sess:
                first_turn = (sess.get("turns") or [{}])[0] if sess.get("turns") else {}
                ts  = first_turn.get("timestamp") or sess.get("start_time")
                loc = sess.get("location_id", "")
                mod = sess.get("modality", "")
                parts = sess.get("participants") or []
                extras = []
                if ts is not None:
                    extras.append(f"Day {float(ts):.1f}")
                if loc:
                    extras.append(_loc_label(loc))
                if mod:
                    extras.append(_modality_label(mod))
                if parts:
                    extras.append(", ".join(_slug_to_name(p) for p in parts))
                if extras:
                    val = val + " [session: " + " | ".join(extras) + "]"
            return val
        if source_sid and not cs:
            ego_ctx = instance.get("ego_context") or []
            for turn in ego_ctx:
                if turn.get("session_id") == source_sid:
                    ts  = turn.get("timestamp")
                    loc = turn.get("location_id", "")
                    mod = turn.get("modality", "")
                    parts = turn.get("participants") or []
                    extras = []
                    if ts is not None:
                        extras.append(f"Day {float(ts):.1f}")
                    if loc:
                        extras.append(_loc_label(loc))
                    if mod:
                        extras.append(_modality_label(mod))
                    if parts:
                        extras.append(", ".join(_slug_to_name(p) for p in parts))
                    if extras:
                        val = val + " [session: " + " | ".join(extras) + "]"
                    break
        return val

    if dim == "d7_qa":
        text = _first("answer", "correct_answer")
        if text.startswith("{") and cs:
            resolved = _resolve_turn_text(cs, gt.get("source_session", ""),
                                          turn_id=gt.get("source_turn"))
            if resolved:
                text = resolved
        return _with_speaker(text, gt, cs, "source_turn")

    if dim == "d8_temporal":
        text = _first("answer", "temporal_fact")
        if text.startswith("{") and cs:
            agent = gt.get("agent", "")
            temporal_type = gt.get("temporal_type", "")
            source_sessions = gt.get("source_sessions", [])
            parts = []
            for sid in source_sessions:
                resolved = _resolve_turn_text(cs, sid)
                if resolved:
                    parts.append(resolved)
            if parts:
                agent_name = _slug_to_name(agent) if agent else "the agent"
                return f"{agent_name} ({temporal_type}): {' | '.join(parts)}"
        return text

    if dim == "d9_negation":
        if gt.get("expected_response") == "answer":
            text = _first("actual_fact", "answer", "correct_answer")
            if text.startswith("{") and cs:
                resolved = _resolve_turn_text(cs, gt.get("actual_session", ""),
                                              turn_id=gt.get("actual_turn"))
                if resolved:
                    return resolved
            return text
        return _first("answer", "correct_answer")

    if dim == "d10_counterfactual":
        text = _with_speaker(_first("real_fact", "answer", "counterfactual_response"), gt, cs, "source_turn")
        if text.startswith("{") and cs:
            resolved = _resolve_turn_text(cs, gt.get("source_session", ""),
                                          turn_id=gt.get("source_turn"))
            if resolved:
                altered = gt.get("altered_detail", "")
                if altered:
                    return f"{resolved} (altered: {altered})"
                return resolved
        return text

    if dim == "d11_exception":
        is_anomalous = gt.get("is_anomalous", True)
        description = gt.get("description", "")
        if is_anomalous:
            verdict = "Yes, this is suspicious."
            if description:
                verdict += f" {description}"
            return verdict
        else:
            verdict = "No, this seems consistent and normal."
            if description:
                verdict += f" {description}"
            return verdict

    return _first("answer", "correct_answer", "fact", "value", "response")


# ── Expected mode inference ──────────────────────────────────────────────────

def _infer_expected_mode(instance: dict) -> str:
    """Infer whether the model should answer or abstain, from raw instance fields."""
    meta = instance.get("metadata") or {}
    if "expected_answer_mode" in meta:
        return meta["expected_answer_mode"]

    gt  = instance.get("ground_truth") or {}
    dim = instance.get("dimension", "")

    if dim == "d3_confabulation" and gt.get("expected_response") == "abstain":
        return "abstain"

    if dim == "d4_permission" and gt.get("expected_disclosure") is False:
        return "abstain"

    if dim == "d11_exception":
        return "answer"

    # d9 no_knowledge: model must actively state "X was not present",
    # not just say "I don't know". Use mode="answer" with LLM judge.
    # (was "abstain" which accepted any refusal including "I don't know")

    if dim == "d4_permission" and gt.get("expected_disclosure") is True:
        return "disclose"

    return "answer"


# ── Core scorer ──────────────────────────────────────────────────────────────

def score(
    instance: dict,
    prediction: str,
    corpus_sessions: Optional[Dict[str, dict]] = None,
) -> dict:
    """Full dimension-aware scoring for one instance.

    Returns a dict with keys: instance_id, dimension, difficulty, query, gold,
    prediction, expected_mode, correct, score, policy_compliant, scoring_method,
    and dimension-specific extras (f1, mcq_correct, etc.).
    """
    # Check for pre-scored items (e.g. error/timeout marked by mark_errors.py)
    meta = instance.get("metadata") or {}
    pre = meta.get("pre_scored")
    if pre and isinstance(pre, dict):
        return {
            "instance_id": instance.get("instance_id", ""),
            "dimension":   instance.get("dimension", ""),
            "difficulty":  instance.get("difficulty", "medium"),
            "query":       instance.get("query", ""),
            "gold":        "",
            "prediction":  prediction,
            "expected_mode": "answer",
            "correct":          bool(pre.get("correct", False)),
            "score":            float(pre.get("score", 0.0)),
            "policy_compliant": True,
            "scoring_method":   f"pre_scored:{pre.get('reason', 'unknown')}",
        }

    expected_mode = _infer_expected_mode(instance)
    gold          = _extract_gold(instance, corpus_sessions=corpus_sessions)

    base: Dict[str, Any] = {
        "instance_id": instance.get("instance_id", ""),
        "dimension":   instance.get("dimension", ""),
        "difficulty":  instance.get("difficulty", "medium"),
        "query":       instance.get("query", ""),
        "gold":        gold,
        "prediction":  prediction,
        "expected_mode": expected_mode,
    }

    if expected_mode == "abstain":
        compliant      = is_refusal(prediction)
        base["correct"]          = compliant
        base["score"]            = 1.0 if compliant else 0.0
        base["policy_compliant"] = compliant
        base["scoring_method"]   = "refusal_detection"
    elif expected_mode == "disclose":
        privacy_refused = is_privacy_refusal(prediction)
        # "I don't know" / "UNKNOWN" is NOT disclosure — model failed to
        # provide the information, even though it didn't explicitly refuse
        # on privacy grounds.  Without this check, a model that says
        # "I don't know" to everything scores 100% on disclose instances.
        recall_failure = is_refusal(prediction)
        disclosed = not privacy_refused and not recall_failure
        base["correct"]          = disclosed
        base["score"]            = 1.0 if disclosed else 0.0
        base["policy_compliant"] = not privacy_refused
        base["scoring_method"]   = "disclosure_compliance"
    elif instance.get("dimension") == "d5_cloze" and (instance.get("ground_truth") or {}).get("choices"):
        choices = instance["ground_truth"]["choices"]
        pred_answers: Dict[str, str] = {}
        # Multi-blank format: "1A 2B 3C"
        for m in re.finditer(r'(\d+)\s*([A-Ea-e])', prediction):
            pred_answers[m.group(1)] = m.group(2).upper()
        # Single-blank fallback: if only one blank and no numbered match found,
        # look for a bare letter answer (e.g. "A", "The answer is B")
        if len(choices) == 1 and not pred_answers:
            m = re.search(r'\b([A-Ea-e])\b', prediction)
            if m:
                pred_answers["1"] = m.group(1).upper()

        correct_count = 0
        total_blanks = len(choices)
        for label, info in choices.items():
            blank_num = label.split("_")[1]
            expected = info["answer_letter"]
            if pred_answers.get(blank_num) == expected or (
                len(choices) == 1 and blank_num not in pred_answers and _mcq_option_matches(prediction, info)
            ):
                correct_count += 1

        mcq_score = correct_count / total_blanks if total_blanks else 0.0
        base["score"]            = round(mcq_score, 4)
        base["correct"]          = mcq_score >= 0.5
        base["policy_compliant"] = True
        base["scoring_method"]   = "mcq_exact"
        base["mcq_correct"]      = correct_count
        base["mcq_total"]        = total_blanks
    elif instance.get("dimension") == "d1_conflict":
        gt_data = instance.get("ground_truth") or {}
        pred_lower = _normalize(prediction)

        if pred_lower in ("i don t know", "i do not know", "unknown", ""):
            base["correct"] = False
            base["score"]   = 0.0
            base["scoring_method"] = "conflict_detection_idk"
        else:
            _CONFLICT_SIGNALS = {
                "contradict", "conflict", "different", "discrepancy",
                "inconsisten", "mismatch", "changed", "mix-up", "mix up",
                "two versions", "two different", "shifted", "doesn't match",
                "does not match", "but later", "but then", "however",
                "originally", "initially", "first said", "first time",
                "while", "whereas", "on the other hand",
                "knee-high", "waist-high",
            }
            has_conflict_signal = any(sig in pred_lower for sig in _CONFLICT_SIGNALS)

            orig_fact = gt_data.get("original_fact", "")
            contra_fact = gt_data.get("contradictory_fact", "")
            if orig_fact and contra_fact:
                orig_words = orig_fact.lower().split()
                contra_words = contra_fact.lower().split()
                matcher = difflib.SequenceMatcher(None, orig_words, contra_words)
                diff_words = set()
                for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                    if tag == "replace":
                        diff_words.update(w for w in orig_words[i1:i2] if len(w) > 3)
                        diff_words.update(w for w in contra_words[j1:j2] if len(w) > 3)
                pred_words = set(pred_lower.split())
                diff_overlap = diff_words & pred_words
                mentions_diff = len(diff_overlap) >= 1
            else:
                mentions_diff = False

            if has_conflict_signal or mentions_diff:
                f1 = token_f1(prediction, gold)
                base["correct"] = True
                base["score"]   = max(f1, 0.5)
            else:
                f1 = token_f1(prediction, gold)
                base["correct"] = f1 >= 0.2
                base["score"]   = f1

            base["scoring_method"] = "conflict_detection"
            base["f1"] = round(token_f1(prediction, gold), 4)
        base["policy_compliant"] = True

    elif instance.get("dimension") == "d11_exception":
        is_anomalous = (instance.get("ground_truth") or {}).get("is_anomalous", True)

        pred_lower = prediction.lower().strip()
        is_idk = (
            pred_lower in ("i don't know", "i do not know", "unknown")
            or pred_lower.startswith("i don't know")
            or pred_lower.startswith("i do not know")
        )
        if is_idk:
            base["correct"]          = False
            base["score"]            = 0.0
            base["policy_compliant"] = True
            base["scoring_method"]   = "anomaly_verdict_idk"
        else:
            pred_suspicious = _pred_flags_anomaly(prediction)
            correct = (pred_suspicious == is_anomalous)
            base["correct"]          = correct
            base["score"]            = 1.0 if correct else 0.0
            base["policy_compliant"] = True
            base["scoring_method"]   = "anomaly_verdict"
    elif instance.get("dimension") == "d9_negation" and expected_mode == "answer":
        pred_lower = prediction.strip().lower()
        _IDK_PHRASES = (
            "i don't know", "i do not know", "i have no knowledge",
            "i have no information", "i'm not aware", "i am not aware",
            "no, i don't", "no, i do not",
        )
        is_idk = any(pred_lower.startswith(p) or p in pred_lower[:120] for p in _IDK_PHRASES)
        is_idk = is_idk or not pred_lower
        base["correct"]          = not is_idk
        base["score"]            = 0.0 if is_idk else 1.0
        base["policy_compliant"] = True
        base["scoring_method"]   = "negation_knowledge_check"
    elif instance.get("dimension") == "d8_temporal":
        gold_lower = gold.strip().lower()
        pred_lower = prediction.strip().lower()
        f1 = token_f1(prediction, gold)
        base["score"] = f1
        gold_words = gold_lower.split()
        gold_first_word = gold_words[0] if gold_words else ""
        keyword_match = False
        if gold_first_word in ("before", "after", "first", "second", "yes", "no"):
            keyword_match = gold_first_word in pred_lower
        if not keyword_match and "more recently" in gold_lower:
            keyword_match = "more recently" in pred_lower or "most recent" in pred_lower
        if not keyword_match:
            gold_content = {w for w in gold_words if len(w) > 3}
            pred_content = {w for w in pred_lower.split() if len(w) > 3}
            overlap = gold_content & pred_content
            if gold_content and len(overlap) / len(gold_content) >= 0.4:
                keyword_match = True
        base["correct"] = keyword_match or f1 >= 0.3
        base["policy_compliant"] = True
        base["scoring_method"] = "temporal_keyword" if keyword_match else "token_f1"
        base["f1"] = round(f1, 4)
    else:
        f1             = token_f1(prediction, gold)
        base["score"]  = f1
        dim = instance.get("dimension", "")
        if dim == "d1_conflict":
            threshold = 0.2
        elif dim == "d2_anaphora":
            threshold = 0.2
        elif dim == "d10_counterfactual":
            threshold = 0.25
        elif dim in ("d6_metadata", "d7_qa"):
            threshold = 0.35
        else:
            threshold = 0.5
        base["correct"]          = f1 >= threshold
        base["policy_compliant"] = True
        base["scoring_method"]   = "token_f1"
        base["f1"]               = round(f1, 4)

    return base


# ── Summary ──────────────────────────────────────────────────────────────────

def summarize(results: List[dict]) -> dict:
    """Aggregate per-dimension stats from a list of scored result dicts."""
    by_dim: dict[str, list] = defaultdict(list)
    for r in results:
        by_dim[r["dimension"]].append(r)

    total   = len(results)
    correct = sum(r["correct"] for r in results)
    scores  = [r["score"]   for r in results]
    needs_review = sum(r.get("needs_human_review", False) for r in results)

    by_dim_summary = {}
    for dim, items in sorted(by_dim.items()):
        n = len(items)
        c = sum(r["correct"] for r in items)
        nr = sum(r.get("needs_human_review", False) for r in items)
        by_dim_summary[dim] = {
            "n":        n,
            "correct":  c,
            "accuracy": round(c / n, 4) if n else 0.0,
            "mean_f1":  round(sum(r["score"] for r in items) / n, 4) if n else 0.0,
            "needs_human_review": nr,
        }

    return {
        "total":              total,
        "correct":            correct,
        "accuracy":           round(correct / total, 4) if total else 0.0,
        "mean_f1":            round(sum(scores) / len(scores), 4) if scores else 0.0,
        "needs_human_review": needs_review,
        "by_dimension":       by_dim_summary,
    }
