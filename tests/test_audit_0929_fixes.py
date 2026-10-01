"""Regression tests for the code defects found by the 2026-09-29 result audit."""
from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from eval.src.adapters.inmemory import InMemoryAdapter, _coerce_day
from eval.src.adapters.memobase_adapter import MemobaseAdapter, MemobaseAPIError
from eval.src.adapters.oracle_with_distractors import OracleWithDistractorsAdapter
from eval.src.answering import AnswerConfig, AnswerEngine, ThinkingEnabledError
from eval.src.adapters.base import PromptStyle
from eval.src.scoring import evaluate_answers
from eval.src.types import AnswerRecord, MessageEntry, QAItem, SearchHit
from MASim.ground_truth.query_time import query_timestamp, session_end_times
from memarena.text_cleaning import clean_llm_text, strip_think, think_status

REPO = Path(__file__).resolve().parents[1]


# ── 1. Qwen3 thinking ────────────────────────────────────────────────────────

def test_think_status_and_strip() -> None:
    assert think_status('{"answer": "x"}') == (False, False)
    assert think_status('<think>a</think>\n{"answer": "x"}') == (True, False)
    assert think_status("<think>\nstill reasoning, then max_tokens") == (True, True)
    assert think_status("reasoning from a template-opened block</think>") == (True, True)
    assert strip_think('<think>a</think>{"answer": "x"}') == '{"answer": "x"}'
    assert strip_think("<think>\nOkay, the answer is {\"answer\": \"leak\"}") == ""
    assert clean_llm_text("<think>") == ""


def _qa(qid: str = "q1", dim: str = "d7_qa") -> QAItem:
    return QAItem(question_id=qid, question="What?", answer="gold", metadata={"dimension": dim})


def _engine(**cfg) -> AnswerEngine:
    eng = AnswerEngine(AnswerConfig(api_key="EMPTY", endpoint="http://127.0.0.1:9/v1", **cfg))
    return eng


def _fake_llm(eng: AnswerEngine, raw: str) -> None:
    async def fake(system_prompt, payload, *, context_prefix=None):
        return raw, {"elapsed_ms": 1.0}
    eng._complete_json = fake  # type: ignore[method-assign]


def test_truncated_think_is_recorded_empty_not_scored_as_think() -> None:
    eng = _engine()
    _fake_llm(eng, "<think>\nLet me think about who said what")
    rec = asyncio.run(eng.answer_one(qa=_qa(), ctx_hits=[]))
    assert rec.prediction == ""
    assert rec.think_detected is True and rec.think_truncated is True
    assert eng.thinking_report()["think_truncated"] == 1


def test_closed_think_keeps_answer_after_block() -> None:
    eng = _engine()
    _fake_llm(eng, '<think>short</think>\n{"answer": "blue notebook"}')
    rec = asyncio.run(eng.answer_one(qa=_qa(), ctx_hits=[]))
    assert rec.prediction == "blue notebook"
    assert rec.think_detected is True and rec.think_truncated is False


def test_thinking_guard_raises_behind_flag() -> None:
    eng = _engine(fail_on_thinking=True, think_min_answers=3)
    _fake_llm(eng, '<think>x</think>{"answer": "a"}')
    for i in range(2):
        asyncio.run(eng.answer_one(qa=_qa(f"q{i}"), ctx_hits=[]))
    with pytest.raises(ThinkingEnabledError):
        asyncio.run(eng.answer_one(qa=_qa("q3"), ctx_hits=[]))


def test_scoring_treats_legacy_literal_think_prediction_as_empty() -> None:
    qa = _qa()
    rec = AnswerRecord(question_id="q1", question="What?", answer="gold", prediction="<think>",
                       model="m", raw_response="<think>\nreasoning")
    out = asyncio.run(evaluate_answers(qas=[qa], answers=[rec]))
    row = out["details"][0]
    assert row["prediction"] == ""
    assert row["think_truncated"] is True
    assert out["summary"]["think_truncated"] == 1


# ── 2. adapters that must use their own context ─────────────────────────────

def _corpus() -> dict:
    def sess(sid, t, text):
        return {"session_id": sid, "participants": ["alice", "bob"], "modality": "text_message",
                "location_id": "", "start_time": t, "end_time": t + 0.01,
                "turns": [{"turn_id": f"{sid}_t0", "speaker_id": "alice", "text": text, "timestamp": t}]}
    return {"ev": sess("ev", 1.0, "EVIDENCE TURN"), "dis": sess("dis", 2.0, "DISTRACTOR TURN")}


def test_text_sessions_prompt_uses_hits_not_gold_evidence() -> None:
    corpus = _corpus()
    eng = AnswerEngine(AnswerConfig(), prompt_style=PromptStyle.TEXT_SESSIONS, corpus_sessions=corpus,
                       ego_session_map={"alice": ["ev", "dis"]}, adapter_name="x", context_from_hits=True)
    qa = QAItem(question_id="q", question="Q?", answer="a",
                metadata={"dimension": "d7_qa", "ego_agent_id": "alice", "evidence_session_ids": ["ev"],
                          "evidence_sessions": ["ev"], "query_agent": "alice"})
    hit = SearchHit(msg_id="dis_t0", score=1.0, text="x", occur_ts="", thread_id="dis", user_id=None)
    _, user = eng._build_text_prompt(qa, [hit])
    assert "DISTRACTOR TURN" in user and "EVIDENCE TURN" not in user
    _, user_empty = eng._build_text_prompt(qa, [])
    assert "EVIDENCE TURN" not in user_empty and "no conversation history" in user_empty


def test_oracle_gated_withholds_deny_evidence() -> None:
    corpus = _corpus()
    eng = AnswerEngine(AnswerConfig(), prompt_style=PromptStyle.TEXT_SESSIONS, corpus_sessions=corpus,
                       adapter_name="oracle_gated")
    meta = {"dimension": "d4_permission", "ego_agent_id": "alice", "evidence_session_ids": ["ev"],
            "context_tier": "evidence_only"}
    deny = QAItem("q", "Q?", "a", metadata={**meta, "policy_expected": "DENY_NO_ACCESS"})
    allow = QAItem("q", "Q?", "a", metadata={**meta, "policy_expected": "ALLOW"})
    assert "EVIDENCE TURN" not in eng._build_text_prompt(deny)[1]
    assert "EVIDENCE TURN" in eng._build_text_prompt(allow)[1]


def test_distractor_adapter_marks_hit_context() -> None:
    assert OracleWithDistractorsAdapter.text_context_from_hits is True


# ── 3. query time ────────────────────────────────────────────────────────────

def test_query_time_after_last_evidence_and_ego_fallback() -> None:
    sessions = [
        {"session_id": "a", "start_time": 1.2, "end_time": 1.3, "turns": [{"timestamp": 1.35}]},
        {"session_id": "b", "start_time": 5.0, "end_time": 5.1, "turns": []},
        {"session_id": "c", "start_time": 9.0, "end_time": 9.2, "turns": []},
    ]
    ends = session_end_times(sessions)
    assert query_timestamp(["a", "b"], ["a", "b", "c"], ends) == (5.1, "after_last_evidence")
    assert query_timestamp([], ["a", "b"], ends) == (5.1, "after_last_ego_session")
    assert query_timestamp([], None, ends) == (9.2, "after_last_session")


def test_fix_query_timestamps_refuses_in_place(tmp_path: Path) -> None:
    (tmp_path / "eval_instances").mkdir()
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "fix_query_timestamps.py"), str(tmp_path), str(tmp_path)],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "never rewrites a release in place" in r.stderr


# ── 5. retriever scope ───────────────────────────────────────────────────────

def test_coerce_day_reads_iso_message_timestamps() -> None:
    assert _coerce_day("2025-01-11T09:30:00+00:00") == 10
    assert _coerce_day("7.9") == 7


def test_bm25_cutoff_applies_to_iso_timestamps() -> None:
    def m(mid, day, text, sid):
        iso = f"2025-01-{day + 1:02d}T10:00:00+00:00"
        return MessageEntry(msg_id=mid, occur_ts=iso, deliver_ts=iso, user_id=None, thread_id=sid, text=text)
    a = InMemoryAdapter()
    a.set_ego_session_map({"alice": ["s0", "s9"]})

    async def run():
        await a.add(namespace="n", messages=[m("0", 0, "red notebook early", "s0"), m("9", 9, "red notebook late", "s9")])
        return await a.search(namespace="n", query="red notebook", top_k=5, ego_id="alice", query_timestamp=3.5)
    assert [h.msg_id for h in asyncio.run(run())] == ["0"]


def test_dense_adapter_applies_scope(monkeypatch) -> None:
    from eval.src.adapters.dense_e5 import DenseE5Adapter
    a = DenseE5Adapter()
    monkeypatch.setattr(a, "_encode_passages", lambda texts: np.ones((len(texts), 2), dtype=np.float32) / np.sqrt(2))
    monkeypatch.setattr(a, "_encode_query", lambda text: np.ones((1, 2), dtype=np.float32) / np.sqrt(2))
    a.set_ego_session_map({"alice": ["s0", "s9"]})
    msgs = [MessageEntry("0", "0.5", "0.5", None, "s0", "early"), MessageEntry("9", "9.5", "9.5", None, "s9", "late"),
            MessageEntry("b", "0.5", "0.5", None, "bob", "other ego")]

    async def run(**kw):
        await a.add(namespace="n", messages=msgs) if not a._messages else None
        return sorted(h.msg_id for h in await a.search(namespace="n", query="q", top_k=5, ego_id="alice", **kw))
    assert asyncio.run(run()) == ["0", "9"]
    assert asyncio.run(run(query_timestamp=3.0)) == ["0"]
    assert asyncio.run(run(query_timestamp=10.0, exclude_thread_ids=["s0"])) == ["9"]


# ── 4. Memobase body errno ───────────────────────────────────────────────────

def test_memobase_errno_in_http_200_is_retried_then_counted() -> None:
    adapter = MemobaseAdapter(cfg={"base_url": "http://localhost:8019", "api_key": "secret", "errno_retries": 3})
    adapter._errno_backoff = lambda attempt: 0.0  # type: ignore[method-assign]
    replies = [{"data": None, "errno": 500, "errmsg": "too many clients already"}, {"data": {}, "errno": 0}]

    async def fake_request(method, path, *, json_body=None, headers=None, timeout_s=None):
        return replies.pop(0) if replies else {"data": None, "errno": 500, "errmsg": "too many clients"}
    adapter._request_with_retry = fake_request  # type: ignore[method-assign]

    assert asyncio.run(adapter._call("POST", "/x", op="insert"))["errno"] == 0
    with pytest.raises(MemobaseAPIError):
        asyncio.run(adapter._call("POST", "/x", op="insert"))
    st = adapter.api_stats()["insert"]
    assert st["calls"] == 2 and st["ok"] == 1 and st["failed"] == 1 and st["retries"] == 3


def test_memobase_add_reports_errno_failures() -> None:
    adapter = MemobaseAdapter(cfg={"base_url": "http://localhost:8019", "api_key": "secret", "errno_retries": 1})
    adapter._user_ids["alice"] = "uid"

    async def fake_request(method, path, *, json_body=None, headers=None, timeout_s=None):
        if "blobs/insert" in path:
            return {"data": None, "errno": 500, "errmsg": "too many clients"}
        return {"data": {}, "errno": 0}
    adapter._request_with_retry = fake_request  # type: ignore[method-assign]
    msg = MessageEntry("m", "1.0", "1.0", None, "s1", "hello", meta={"speaker": "alice"})
    res = asyncio.run(adapter.add(namespace="alice", messages=[msg]))
    assert res["indexed_messages"] == 0 and res["n_errors"] == 1 and res["submitted_messages"] == 1
