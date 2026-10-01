#!/usr/bin/env python3
"""mark_errors.py — Pre-mark error/timeout items in answer result JSONs.

Scans all answer result files for items with:
  - empty raw_response
  - "ERROR" in raw_response
  - model == "error"
  - empty prediction AND empty raw_response

Marks them in-place with a `pre_scored` field so the judge pipeline can
skip the LLM call and use score=0 directly.

Idempotent: safe to run multiple times.
"""
import glob
import json
import sys
from pathlib import Path


def _is_bad_item(item: dict) -> str | None:
    """Return a reason string if the item should be pre-scored as error, else None."""
    raw = str(item.get("raw_response", ""))
    pred = str(item.get("prediction", ""))
    model = str(item.get("model", ""))

    if model == "error":
        return "model_error"
    if "ERROR" in raw:
        return "error_in_raw_response"
    if not raw and not pred:
        return "empty_response"
    if not raw and pred == "UNKNOWN":
        return "empty_raw_unknown_pred"
    return None


def mark_file(filepath: str) -> int:
    """Mark bad items in a single answer result JSON. Returns count of newly marked items."""
    path = Path(filepath)
    data = json.loads(path.read_text())

    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("results", data.get("details", []))
    else:
        return 0

    marked = 0
    for item in items:
        reason = _is_bad_item(item)
        if reason and "pre_scored" not in item:
            item["pre_scored"] = {
                "score": 0.0,
                "correct": False,
                "reason": reason,
            }
            # Also normalize prediction so the judge sees a clear marker
            item["prediction"] = "[ERROR: no valid response]"
            marked += 1

    if marked > 0:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")

    return marked


def main():
    # Discover all answer result files (same patterns as run_judge.sh)
    patterns = [
        "MASim/runs/*/eval_results/vanilla_*.json",
        "MASim/runs/*/eval_results/oracle_*.json",
        "MASim/runs/*/eval_results/*/answer_results_*.json",
        "MASim/runs/*/eval_results/*/runs/*/answer_results_*.json",
    ]

    files = []
    for p in patterns:
        files.extend(glob.glob(p))

    # Filter out non-answer files
    files = [
        f for f in files
        if "light" not in Path(f).name
        and "judged" not in Path(f).name
        and "evaluation" not in Path(f).name
    ]
    files = sorted(set(files))

    total_marked = 0
    for f in files:
        try:
            n = mark_file(f)
            if n > 0:
                print(f"  {n:3d} items marked in {f}")
                total_marked += n
        except Exception as e:
            print(f"  ERROR processing {f}: {e}", file=sys.stderr)

    if total_marked == 0:
        print("  No new error items to mark (all clean or already marked).")
    else:
        print(f"\n  Total: {total_marked} items marked as pre_scored across {len(files)} files.")


if __name__ == "__main__":
    main()
