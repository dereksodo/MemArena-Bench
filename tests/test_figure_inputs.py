from __future__ import annotations

import json
from pathlib import Path

from memarena.figures.gen_pvalues import _load_qid_to_agent


def test_pvalues_reads_seed_scoped_masim_qa(tmp_path: Path) -> None:
    qa_dir = tmp_path / "eval_results_s2" / "oracle"
    qa_dir.mkdir(parents=True)
    (qa_dir / "masim_qa_oracle_8b_s2_judge_remote.json").write_text(
        json.dumps({
            "qars": [
                {
                    "id": "d7_example",
                    "meta": {"ego_agent_id": "agent_a"},
                }
            ]
        }),
        encoding="utf-8",
    )

    assert _load_qid_to_agent(tmp_path) == {"d7_example": "agent_a"}


def test_pvalues_falls_back_to_eval_instances(tmp_path: Path) -> None:
    inst_dir = tmp_path / "eval_instances"
    inst_dir.mkdir()
    (inst_dir / "d7_qa.jsonl").write_text(
        json.dumps({
            "instance_id": "d7_example",
            "dimension": "d7_qa",
            "ego_agent_id": "agent_b",
        }) + "\n",
        encoding="utf-8",
    )

    assert _load_qid_to_agent(tmp_path) == {"d7_example": "agent_b"}
