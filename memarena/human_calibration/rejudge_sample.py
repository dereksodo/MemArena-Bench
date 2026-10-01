#!/usr/bin/env python3
"""Re-judge the human-calibration sample with a chosen judge.

``sample.json`` (local, gitignored) carries the gpt-4o-mini verdicts of the
submitted paper, which were made before the gold answer reached the evidence
judge (commit 337b277). This script recomputes every verdict on the same 500
records with the current scoring code, i.e. with the gold passed, for one
judge of memarena/judges.py:

* binary items (D2-D5): ``eval.src.scoring._score_one`` on the dataset's
  QAItem (evidence-grounded judge, gold in ``gold_ground_truth``), exactly as
  scripts/llmjudge.py scores them;
* D6 items: the 5-label relabel of scripts/rerun_d6_judge.py (``_judge_record``,
  gold fact = the record's gold), exactly as scripts/judge_cell.sh step 2 does.

DeepSeek judges run with reasoning on (4096-token budget); prompts, temperature 0 and parsing
are unchanged. The output has the layout of sample.json (judge_correct /
judge_label / judge_reason replaced, the originals kept as *_original), so
analyze_judge_human.py reads it with ``--pool``:

    python3 -m memarena.human_calibration.rejudge_sample --judge gpt4omini
    python3 -m memarena.human_calibration.rejudge_sample --judge deepseek
    python3 -m memarena.human_calibration.analyze_judge_human \\
        --labels memarena/human_calibration/labels.json \\
        --pool memarena/human_calibration/sample_judge_deepseek.json --judge-name deepseek

This makes one or two API calls per item (500 items). The key is
OPENROUTER_API_KEY from the environment or .env; it is never printed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from memarena.judges import JUDGES, PRIMARY_JUDGE, judge_spec  # noqa: E402

HERE = Path(__file__).resolve().parent
JUDGE_REASON_PREFIXES = ("evidence_judge:", "llm_judge:")


def rejudge_row(row: Mapping, qa: Any, client: Any, model: str, corpus: Optional[Dict[str, dict]]) -> dict:
    """New verdict fields for one sample row."""
    from eval.src.scoring import _armB_correctness, _score_one
    from scripts.rerun_d6_judge import _judge_record

    if row.get("dim_kind") == "d6" or str(row.get("id", "")).startswith("d4_perm"):
        rec = {
            "prediction": row.get("prediction") or "",
            "question_id": row.get("id"),
            "expected_answer_mode": row.get("expected_answer_mode") or "",
        }
        gt = (qa.metadata.get("gold_ground_truth") or qa.metadata.get("ground_truth")) if qa is not None else None
        label, fact_in, rationale, _reason = _judge_record(
            client, model, rec, str(row.get("gold") or ""),
            question=qa.question if qa is not None else None, gt=gt)
        ok, reason = _armB_correctness(label, str(rec["expected_answer_mode"]).lower())
        return {"judge_correct": bool(ok), "judge_label": label, "judge_reason": rationale or reason,
                "leaked_fact_in_output": bool(fact_in), "judge_scored": label != "PARSE_ERROR"}
    ok, _score, reason = _score_one(qa, row.get("prediction") or "", client, model, corpus)
    return {"judge_correct": bool(ok), "judge_label": "CORRECT" if ok else "INCORRECT",
            "judge_reason": reason, "judge_scored": str(reason).startswith(JUDGE_REASON_PREFIXES)}


def rejudge_sample(
    sample: Mapping,
    qas_by_id: Mapping[str, Any],
    client: Any,
    judge_tag: str,
    corpus: Optional[Dict[str, dict]] = None,
    workers: int = 32,
    progress: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """A copy of ``sample`` (sample.json layout) with this judge's verdicts."""
    spec = judge_spec(judge_tag)
    rows: List[dict] = [dict(r) for r in sample.get("sample", [])]
    missing = [r["id"] for r in rows if r.get("dim_kind") != "d6" and r["id"] not in qas_by_id]
    if missing:
        raise ValueError(f"{len(missing)} sample ids are not in the dataset, e.g. {missing[:5]}")

    done = 0
    lock = threading.Lock()

    def one(row: dict) -> dict:
        nonlocal done
        new = rejudge_row(row, qas_by_id.get(row["id"]), client, spec.model, corpus)
        with lock:
            done += 1
            n = done
        if progress is not None:
            progress(n, len(rows))
        return new

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        verdicts = list(pool.map(one, rows))

    changed = 0
    for row, new in zip(rows, verdicts):
        for key in ("judge_correct", "judge_label", "judge_reason", "leaked_fact_in_output"):
            if key in row:
                row[f"{key}_original"] = row[key]
        changed += int(bool(row.get("judge_correct")) != bool(new["judge_correct"])
                       or str(row.get("judge_label")) != str(new["judge_label"]))
        row.update(new)
    meta = dict(sample.get("metadata") or {})
    meta.update({
        "judge_tag": spec.tag,
        "judge_model": spec.model,
        "rejudged_by": "memarena/human_calibration/rejudge_sample.py",
        "rejudged_at": datetime.now().isoformat(timespec="seconds"),
        "gold_passed": True,
        "n_verdicts_changed_vs_original": changed,
        "n_not_judge_scored": sum(1 for r in rows if not r.get("judge_scored", True)),
    })
    return {"metadata": meta, "sample": rows}


def _load_dataset(dataset: Path):
    from eval.src.masim_loader import load_corpus_sessions_dict, load_masim_qa

    corpus = load_corpus_sessions_dict(dataset)
    qas = load_masim_qa(dataset, corpus_sessions=corpus)
    return {q.question_id: q for q in qas}, corpus


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge", default=PRIMARY_JUDGE, choices=sorted(JUDGES))
    ap.add_argument("--sample", type=Path, default=HERE / "sample.json")
    ap.add_argument("--dataset", type=Path,
                    default=Path(os.getenv("MEMARENA_DATASET_DIR") or REPO_ROOT / "data" / "benchmark"))
    ap.add_argument("--out", type=Path, help="default: sample_judge_<judge>.json next to --sample")
    ap.add_argument("--workers", type=int, default=32, help="judge requests in flight")
    args = ap.parse_args(argv)

    from scripts.rerun_d6_judge import _make_client

    sample = json.loads(args.sample.read_text(encoding="utf-8"))
    qas_by_id, corpus = _load_dataset(args.dataset)
    client = _make_client(args.workers)  # OPENROUTER_API_KEY / OPENROUTER_BASE_URL from env or .env
    out = args.out or args.sample.with_name(f"sample_judge_{args.judge}.json")
    t0 = time.time()
    print(f"[rejudge] {len(sample.get('sample', []))} items with {judge_spec(args.judge).model} -> {out}")

    def progress(done: int, total: int) -> None:
        if done % 50 == 0 or done == total:
            print(f"[rejudge] {done}/{total} ({time.time() - t0:.0f}s)", flush=True)

    result = rejudge_sample(sample, qas_by_id, client, args.judge, corpus, args.workers, progress)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    meta = result["metadata"]
    print(f"[rejudge] verdicts changed vs the original sample: {meta['n_verdicts_changed_vs_original']}; "
          f"not judge-scored: {meta['n_not_judge_scored']}")
    print(f"[rejudge] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
