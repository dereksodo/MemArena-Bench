from __future__ import annotations

import json
from pathlib import Path

from eval.src.masim_loader import load_masim_qa
from eval.src.scoring_core import score
from scripts.stage_paper_accuracy_inputs import _find_eval_result


def _judged_eval(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "details": [
                {
                    "question_id": "d7_example",
                    "reason": "evidence_judge: ok",
                    "correct": True,
                    "answer_scored": True,
                }
            ]
        }),
        encoding="utf-8",
    )


def test_masim_loader_strips_qwen3_thinking_from_queries(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    eval_dir = run_dir / "eval_instances"
    eval_dir.mkdir(parents=True)
    (run_dir / "corpus_sessions.jsonl").write_text("", encoding="utf-8")
    (eval_dir / "d7_qa.jsonl").write_text(
        json.dumps({
            "instance_id": "d7_example",
            "dimension": "d7_qa",
            "query": "<think>rephrase the question</think>\nWhat did Alice plan?",
            "ground_truth": {"answer": "The lantern plan."},
            "metadata": {},
        }) + "\n",
        encoding="utf-8",
    )

    qas = load_masim_qa(run_dir, dimensions=["d7_qa"])

    assert len(qas) == 1
    assert qas[0].question == "What did Alice plan?"


def test_stage_paper_inputs_does_not_swap_memobase_and_memos(tmp_path: Path) -> None:
    root = tmp_path / "matrix"
    base = root / "eval_results_s2" / "memory_cache"
    memobase = base / "evaluation_results_memobase_0_6b_s2_judge_remote.json"
    memos = base / "evaluation_results_memos_0_6b_s2_judge_remote.json"
    _judged_eval(memobase)
    _judged_eval(memos)
    entries = [{"status": "complete", "source_type": "manifest", "run_dir": str(root)}]

    found_memobase = _find_eval_result(entries, "memobase", "0_6b", "s2", require_llm_judge=True)
    found_memos = _find_eval_result(entries, "memos", "0_6b", "s2", require_llm_judge=True)

    assert found_memobase == memobase
    assert found_memos == memos


def test_cloze_word_answer_and_short_no_refusal_score() -> None:
    cloze = {
        "instance_id": "d5_example",
        "dimension": "d5_cloze",
        "query": "Pick the missing word.",
        "ground_truth": {
            "choices": {
                "blank_1": {
                    "answer_letter": "E",
                    "blank_word": "oak",
                    "options": ["maple", "pine", "cedar", "birch", "oak"],
                }
            }
        },
    }
    abstain = {
        "instance_id": "d3_example",
        "dimension": "d3_confabulation",
        "query": "Do you know the private detail?",
        "ground_truth": {"expected_response": "abstain", "answer": ""},
    }

    assert score(cloze, "oak")["correct"] is True
    assert score(abstain, "No.")["correct"] is True
