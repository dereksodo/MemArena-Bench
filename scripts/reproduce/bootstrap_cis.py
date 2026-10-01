#!/usr/bin/env python3
"""Compute bootstrap 95% confidence intervals for every (backend, model)
trial on \\benchL. Reads per-instance details from the 4o-mini judge files
and resamples to estimate CIs on overall and per-dimension accuracy.

Output: scoring_summary_ci.json sidecar that the table generator can read.
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
import random

EVAL_DIR = Path("MASim/runs/l_20260408_111046/eval_results")
JUDGE_TAG = os.environ.get("JUDGE_TAG", "4omini")
N_BOOTSTRAP = int(os.environ.get("N_BOOTSTRAP", "1000"))
SEED = int(os.environ.get("SEED", "12345"))

# Same dim filter as compute_all_metrics.py
EXCLUDED_DIMS = {"d9_negation", "d11_exception"}


def determine_backend(trial_name: str):
    """trial -> (backend, model_slug)"""
    for bk in ("vanilla", "oracle", "rag"):
        if trial_name.startswith(bk + "_"):
            return bk, trial_name[len(bk) + 1:]
    return None, trial_name


def bootstrap_accuracy(scores, n_resample=1000, ci=0.95, rng=None):
    """Return (point_estimate, lower, upper) for the mean of scores."""
    if not scores:
        return (0.0, 0.0, 0.0)
    if rng is None:
        rng = random.Random(SEED)
    n = len(scores)
    point = sum(scores) / n
    means = []
    for _ in range(n_resample):
        sample = [scores[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = int((1 - ci) / 2 * n_resample)
    hi_idx = int((1 + ci) / 2 * n_resample) - 1
    return (round(point, 4), round(means[lo_idx], 4), round(means[hi_idx], 4))


def main():
    out = {}  # trial_name -> { dimension -> {acc, lo, hi, n} }
    rng = random.Random(SEED)

    for sub in ("vanilla", "oracle", "inmem"):
        eval_subdir = EVAL_DIR / sub
        if not eval_subdir.is_dir():
            continue
        for f in sorted(eval_subdir.glob(f"evaluation_results_*_{JUDGE_TAG}.json")):
            trial = f.stem.replace("evaluation_results_", "")
            if trial.endswith(f"_{JUDGE_TAG}"):
                trial = trial[: -(len(JUDGE_TAG) + 1)]
            backend, model_slug = determine_backend(trial)
            if backend is None:
                continue

            data = json.loads(f.read_text())
            details = data.get("details", [])
            if not details:
                continue

            # Group by dimension
            by_dim = defaultdict(list)
            for d in details:
                dim = d.get("dimension", "")
                if not dim or dim in EXCLUDED_DIMS:
                    continue
                # Use judge_correct as a binary score (0 or 1)
                by_dim[dim].append(1 if d.get("judge_correct") else 0)

            trial_out = {}
            # Per-dim CIs
            for dim, scores in by_dim.items():
                point, lo, hi = bootstrap_accuracy(scores, N_BOOTSTRAP, rng=rng)
                trial_out[dim] = {
                    "n": len(scores),
                    "accuracy": point,
                    "ci_lo": lo,
                    "ci_hi": hi,
                }
            # Overall CI (across all included dims)
            all_scores = [s for scores in by_dim.values() for s in scores]
            point, lo, hi = bootstrap_accuracy(all_scores, N_BOOTSTRAP, rng=rng)
            trial_out["overall"] = {
                "n": len(all_scores),
                "accuracy": point,
                "ci_lo": lo,
                "ci_hi": hi,
            }
            out[trial] = trial_out

    out_path = EVAL_DIR / f"scoring_summary_ci_{JUDGE_TAG}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved bootstrap CIs ({N_BOOTSTRAP} resamples, seed={SEED}, judge={JUDGE_TAG}) to {out_path}")
    print()
    print(f"{'trial':30s} {'dim':18s} {'n':>5s} {'acc':>8s} {'ci_lo':>8s} {'ci_hi':>8s}")
    print("-" * 80)
    for trial in sorted(out):
        for dim in ("overall", "d1_conflict", "d4_permission", "d6_metadata", "d7_qa"):
            if dim not in out[trial]:
                continue
            v = out[trial][dim]
            print(f"{trial:30s} {dim:18s} {v['n']:>5d} {v['accuracy']:>8.4f} {v['ci_lo']:>8.4f} {v['ci_hi']:>8.4f}")


if __name__ == "__main__":
    main()
