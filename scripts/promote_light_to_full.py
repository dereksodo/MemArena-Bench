#!/usr/bin/env python3
"""Promote ``answer_results_light_*.json`` files to the full
``answer_results_*.json`` format expected by ``scripts/llmjudge.py``.

The light files contain only the prediction surface
(``question_id, prediction, model, raw_response``). The full
``AnswerRecord`` adds ``question`` and ``answer`` (mandatory), plus
optional timing/token fields. We reconstruct ``question`` and ``answer``
by joining against ``masim_qa`` via ``question_id``; timing/token
fields are filled with ``None``. Predictions are NOT regenerated,
preserving the exact outputs the original chain produced.

Usage::

    python scripts/promote_light_to_full.py \
        out/accuracy_memarena_l_*/eval_results_*/memory_cache/answer_results_light_*.json
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.src.masim_loader import load_corpus_sessions_dict, load_masim_qa
from eval.src.types import AnswerRecord


def promote_one(light_path: Path, qa_by_id: dict) -> tuple[Path, str]:
    full_name = light_path.name.replace("answer_results_light_", "answer_results_", 1)
    full_path = light_path.with_name(full_name)
    if full_path.exists() and full_path.stat().st_size > 0:
        return full_path, "exists"

    light = json.loads(light_path.read_text(encoding="utf-8"))
    if not isinstance(light, list):
        return full_path, "bad-format"

    out_rows = []
    missing = 0
    for row in light:
        qid = row.get("question_id")
        qa = qa_by_id.get(qid)
        if qa is None:
            missing += 1
            continue
        rec = AnswerRecord(
            question_id=qid,
            question=qa.question,
            answer=qa.answer,
            prediction=row.get("prediction"),
            model=row.get("model", ""),
            raw_response=row.get("raw_response", ""),
        )
        out_rows.append(asdict(rec))

    full_path.write_text(json.dumps(out_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return full_path, f"wrote {len(out_rows)} rows (missing={missing})"


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: promote_light_to_full.py <light files...>", file=sys.stderr)
        return 2

    run_dir = REPO_ROOT / "data" / "benchmark"
    print(f"loading masim_qa from {run_dir}...", file=sys.stderr)
    corpus = load_corpus_sessions_dict(run_dir)
    qas = load_masim_qa(run_dir, dimensions=None, corpus_sessions=corpus)
    qa_by_id = {q.question_id: q for q in qas}
    print(f"loaded {len(qa_by_id)} QA items", file=sys.stderr)

    n_ok = n_skip = n_err = 0
    for arg in argv:
        light_path = Path(arg).resolve()
        if not light_path.exists():
            print(f"  MISSING: {light_path}")
            n_err += 1
            continue
        try:
            full_path, status = promote_one(light_path, qa_by_id)
            tag = "skip" if status == "exists" else "ok"
            n_ok += int(tag == "ok")
            n_skip += int(tag == "skip")
            print(f"  [{tag}] {full_path.name}: {status}")
        except Exception as e:
            print(f"  [err] {light_path.name}: {type(e).__name__}: {e}")
            n_err += 1

    print(f"\ntotal: ok={n_ok} skip={n_skip} err={n_err}")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
