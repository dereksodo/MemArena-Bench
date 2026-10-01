"""Restore the original Memobase main-grid run that the paper reports.

On 2026-05-04 an unattended loop agent on the H100 misread the Memobase cells
as missing (``scripts/run_missing_memobase_memos.sh``, commit 882fd61) and
re-ran them from a fresh memory ingest. Its outputs overwrote the original
files in git (c737786 for Mistral-7B/Qwen3-8B/Qwen3-32B; 67de61e for the
Qwen3-0.6B answer files). Every Memobase number in the paper comes from the
original run, so the main grid is pinned to it here.

Each file is restored from the last commit that still held the original run
and is accepted only if its predictions match the original evaluation file
(``a5ea918`` -- original predictions with the 5-label D6 rubric). The re-run
stays recoverable from git history (c737786 / 67de61e).

    python3 scripts/restore_memobase_original_run.py          # dry run
    python3 scripts/restore_memobase_original_run.py --apply
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ORIGINAL_EVAL = "a5ea918"
SEEDS = ("s2", "s3", "s4")

# (path template, git revision holding the original run)
TARGETS = []
for m in ("7b", "8b", "32b"):
    for s in SEEDS:
        base = f"out/accuracy_memarena_l_{m}/eval_results_{s}/memory_cache"
        TARGETS.append((f"{base}/evaluation_results_memobase_{m}_{s}_judge_remote.json", ORIGINAL_EVAL))
        TARGETS.append((f"{base}/answer_results_memobase_{m}_{s}.json", "254b2e4" if (m == "8b" and s != "s2") else "9c69565"))
for s in SEEDS:
    base = f"out/accuracy_memarena_l_0_6b/eval_results_{s}/memory_cache"
    TARGETS.append((f"{base}/answer_results_memobase_0_6b_{s}.json", "9c69565"))
    TARGETS.append((f"{base}/run_meta_memobase_0_6b_{s}.json", "9c69565"))


def git_show(rev: str, path: str) -> str:
    return subprocess.run(["git", "show", f"{rev}:{path}"], cwd=REPO, capture_output=True, text=True, check=True).stdout


def predictions(doc) -> dict:
    rows = doc if isinstance(doc, list) else doc.get("details", [])
    return {r.get("question_id"): r.get("prediction") for r in rows if isinstance(r, dict) and "prediction" in r}


def original_eval_predictions(path: str) -> dict:
    """Predictions of the original evaluation file for the cell that `path` belongs to."""
    p = Path(path)
    m, s = p.parts[1].removeprefix("accuracy_memarena_l_"), p.parts[2].removeprefix("eval_results_")
    eval_path = f"out/accuracy_memarena_l_{m}/eval_results_{s}/memory_cache/evaluation_results_memobase_{m}_{s}_judge_remote.json"
    return predictions(json.loads(git_show(ORIGINAL_EVAL, eval_path)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true", help="write the files (default: dry run)")
    args = ap.parse_args()

    ok = True
    for path, rev in TARGETS:
        text = git_show(rev, path)
        if "run_meta" not in path:
            want = original_eval_predictions(path)
            got = predictions(json.loads(text))
            same = sum(got.get(q) == v for q, v in want.items())
            if same != len(want):
                print(f"REFUSE {path}@{rev}: {same}/{len(want)} predictions match the original run")
                ok = False
                continue
            note = f"{same}/{len(want)} predictions = original"
        else:
            note = "run metadata"
        current = (REPO / path).read_text(encoding="utf-8") if (REPO / path).exists() else None
        state = "unchanged" if current == text else ("restore" if current is not None else "create")
        print(f"{state:9s} {path} <- {rev} ({note})")
        if args.apply and state != "unchanged":
            (REPO / path).write_text(text, encoding="utf-8")
    if not ok:
        print("some targets failed verification; nothing written for them", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
