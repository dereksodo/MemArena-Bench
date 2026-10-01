from __future__ import annotations

import pytest

from eval.src.scoring import JudgeScoringError, _llm_judge_score


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Message(content)


class _Response:
    def __init__(self, content: str) -> None:
        self.choices = [_Choice(content)]


class _Completions:
    def __init__(self, *, content: str | None = None, exc: Exception | None = None) -> None:
        self.content = content
        self.exc = exc
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.exc is not None:
            raise self.exc
        return _Response(self.content or '{"correct": true, "score": 1.0, "reason": "ok"}')


class _Client:
    def __init__(self, completions: _Completions) -> None:
        self.chat = type("Chat", (), {"completions": completions})()


def test_llm_judge_does_not_send_sglang_extra_body() -> None:
    completions = _Completions()
    ok, score, reason = _llm_judge_score(
        _Client(completions),
        "openai/gpt-4o-mini",
        "question",
        "gold",
        "prediction",
        "d7_qa",
    )

    assert ok is True
    assert score == 1.0
    assert reason == "llm_judge: ok"
    assert completions.kwargs is not None
    assert "extra_body" not in completions.kwargs


def test_llm_judge_failure_is_not_token_f1_fallback() -> None:
    completions = _Completions(exc=RuntimeError("rate limited"))

    with pytest.raises(JudgeScoringError, match="judge request failed"):
        _llm_judge_score(
            _Client(completions),
            "openai/gpt-4o-mini",
            "question",
            "gold",
            "prediction",
            "d7_qa",
        )


def test_llm_judge_bad_json_is_not_token_f1_fallback() -> None:
    completions = _Completions(content="not json")

    with pytest.raises(JudgeScoringError, match="not valid JSON"):
        _llm_judge_score(
            _Client(completions),
            "openai/gpt-4o-mini",
            "question",
            "gold",
            "prediction",
            "d7_qa",
        )


def test_d6_judge_message_names_the_fact_owner_and_question() -> None:
    from eval.src.scoring import _build_armB_user

    msg = _build_armB_user("Hiro took up photography.", "What has Hiro been up to lately?",
                           {"fact": "I've been getting into photography.", "target_agent": "hiro_tanaka"})
    assert "Question: What has Hiro been up to lately?" in msg
    assert "about Hiro Tanaka" in msg and "I've been getting into photography." in msg


def test_verbatim_gold_is_attributed_to_its_speaker() -> None:
    from eval.src.scoring_core import _extract_gold

    cs = {"s1": {"turns": [{"turn_id": "s1_t0", "speaker_id": "mehmet_yilmaz",
                            "text": "I'm not saying he's lobbying, just asking."}]}}
    inst = {"dimension": "d3_confabulation",
            "ground_truth": {"correct_fact": "I'm not saying he's lobbying, just asking.",
                             "source_session": "s1", "source_turn": "s1_t0"}}
    assert _extract_gold(inst, cs) == "Mehmet Yilmaz said: I'm not saying he's lobbying, just asking."
    inst["ground_truth"]["correct_fact"] = "Mehmet doubted the lobbying story."
    assert _extract_gold(inst, cs) == "Mehmet doubted the lobbying story."


def test_gold_sessions_come_first_in_the_judge_evidence() -> None:
    from eval.src.scoring import _gold_sessions_first

    assert _gold_sessions_first(["a", "b", "c", "d"], {"source_session": "d"}) == ["d", "a", "b", "c"]
    assert _gold_sessions_first(["a", "b"], {"source_session": "z"}) == ["a", "b"]


def test_gold_already_naming_its_speaker_is_left_alone() -> None:
    from eval.src.scoring_core import _extract_gold

    cs = {"s1": {"turns": [{"turn_id": "s1_t0", "speaker_id": "emeka_eze", "text": "Emeka Eze: lime curing is real."}]}}
    inst = {"dimension": "d3_confabulation",
            "ground_truth": {"correct_fact": "Emeka Eze: lime curing is real.", "source_session": "s1", "source_turn": "s1_t0"}}
    assert _extract_gold(inst, cs) == "Emeka Eze: lime curing is real."


def test_keep_going_scores_an_unjudgeable_record_as_judge_failed(monkeypatch) -> None:
    import asyncio
    import eval.src.scoring as S
    from eval.src.types import AnswerRecord, QAItem

    def boom(*a, **k):
        raise S.JudgeScoringError("empty reply")

    monkeypatch.setattr(S, "_score_one", boom)
    qa = QAItem(question_id="d7_x", question="q", answer="a", question_type="open_ended",
                metadata={"dimension": "d7_qa"})
    ans = [AnswerRecord(question_id="d7_x", question="q", answer="a", model="m", raw_response="p", prediction="p")]
    with pytest.raises(S.JudgeScoringError):
        asyncio.run(S.evaluate_answers(qas=[qa], answers=ans, judge_client=object(), judge_model="m"))
    out = asyncio.run(S.evaluate_answers(qas=[qa], answers=ans, judge_client=object(), judge_model="m",
                                         judge_errors="keep_going"))
    assert out["details"][0]["reason"].startswith(S.JUDGE_FAILED_PREFIX)


def test_text_prompt_names_whose_memory_is_asked() -> None:
    from eval.simple_eval import build_prompt

    inst = {"dimension": "d3_confabulation", "query": "Do you remember Mehmet talking about Tucker?",
            "ego_agent_id": "owen_murphy", "evidence_session_ids": [],
            "metadata": {"query_agent": None, "instance_query_agent": "owen_murphy"}}
    _sys, user = build_prompt(inst, {}, {})
    assert "== Question being asked to Owen Murphy ==" in user and "the person" not in user
    d6 = {"dimension": "d4_permission", "query": "How has Kyaw been?", "ego_agent_id": "kyaw_zin_win",
          "evidence_session_ids": [], "metadata": {"instance_query_agent": "mei_lin_chen", "asker_agent_id": "mei_lin_chen"}}
    sys6, user6 = build_prompt(d6, {}, {}, owner_contacts={"Mei Lin Chen": "acquaintance"})
    assert "== Question from Mei Lin Chen ==" in user6 and "Mei Lin Chen: acquaintance" in user6
    assert sys6.startswith("You are Kyaw Zin Win's personal assistant")


def test_d6_identity_ablation_switches() -> None:
    from eval.simple_eval import build_prompt

    d6 = {"dimension": "d4_permission", "query": "How has Kyaw been?", "ego_agent_id": "kyaw_zin_win",
          "evidence_session_ids": [], "metadata": {"instance_query_agent": "mei_lin_chen", "asker_agent_id": "mei_lin_chen"}}
    _s, l0 = build_prompt(d6, {}, {}, owner_contacts=None, show_asker=False)
    _s, l1 = build_prompt(d6, {}, {}, owner_contacts=None, show_asker=True)
    _s, l2 = build_prompt(d6, {}, {}, owner_contacts={"Mei Lin Chen": "acquaintance"}, show_asker=True)
    assert "Question from Someone" in l0 and "contacts ==" not in l0
    assert "Question from Mei Lin Chen" in l1 and "contacts ==" not in l1
    assert "Question from Mei Lin Chen" in l2 and "Mei Lin Chen: acquaintance" in l2
