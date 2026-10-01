#!/usr/bin/env python3
"""Pure-Python rules baseline for d4_permission (paper dimension D6).

Applies a deterministic rule on MASim ground-truth permission tags
(authorized + permission_level) to produce predictions for each of the 200
d4_permission instances. Then runs the LLM judge against this synthetic
answer file to obtain the same evaluation metrics as the LLM backends.

This gives the "ceiling on access-control logic given correct permission
tags" — the gap between this 100% by-construction ceiling and the LLM
backends frames whether the residual D4_permission gap is a contextual-
integrity reasoning gap (vs missing-evidence gap).

Output:
    out/d4_rules_baseline/answer_results_rules_d4_paper.json
    out/d4_rules_baseline/evaluation_results_rules_d4_paper_judge_remote.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def build_prediction(gt: dict) -> str:
    """Map ground-truth permission tags to a concrete prediction text.

    Rule:
      - authorized=True   → disclose the fact verbatim
      - authorized=False  → refuse politely with permission-aware reasoning
    """
    authorized = bool(gt.get("authorized"))
    if authorized:
        return str(gt.get("fact") or "").strip()
    # Refuse — match the abstain phrasing the judge tends to accept.
    target = gt.get("target_agent") or "the other person"
    level = gt.get("permission_level") or "private"
    return (
        f"I'd rather not share that — that information from {target} is {level}, "
        f"and I don't have permission to disclose it."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--instances",
        type=Path,
        default=REPO_ROOT / "data/benchmark/eval_instances/d4_permission.jsonl",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "out/d4_rules_baseline",
    )
    ap.add_argument("--namespace", default="rules_d4_paper")
    ap.add_argument("--judge-only", action="store_true",
                    help="Skip rebuilding the answer file; only run the judge.")
    ap.add_argument("--no-judge", action="store_true",
                    help="Build the answer file but skip judge invocation.")
    ap.add_argument("--judge-concurrency", type=int, default=64)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    answer_path = args.out_dir / f"answer_results_{args.namespace}.json"

    if not args.judge_only:
        items = []
        with args.instances.open() as f:
            for line in f:
                inst = json.loads(line)
                gt = inst.get("ground_truth") or {}
                pred = build_prediction(gt)
                items.append({
                    "question_id": inst["instance_id"],
                    "question": inst.get("query") or inst.get("question") or "",
                    "answer": str(gt.get("fact") or "").strip(),
                    "prediction": pred,
                    "model": "rules_baseline",
                    "raw_response": pred,
                    "answer_time_ms": 0.0,
                    "ttft_ms": 0.0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "search_time_ms": None,
                    "timing_reliable": True,
                })
        with answer_path.open("w") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        print(f"[rules-baseline] wrote {len(items)} predictions → {answer_path}")

    if args.no_judge:
        return 0

    # Run llmjudge.py on the synthetic answer file.
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        env_file = REPO_ROOT / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line.startswith("OPENROUTER_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    break
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY missing; set in env or .env")

    cmd = [
        str(REPO_ROOT / ".venv/bin/python"),
        str(REPO_ROOT / "scripts/llmjudge.py"),
        "--run-dir", str(REPO_ROOT / "data/benchmark"),
        "--answer-path", str(answer_path),
        "--judge-preset", "remote",
        "--judge-model", "openai/gpt-4o-mini-2024-07-18",
        "--judge-endpoint", "https://openrouter.ai/api/v1",
        "--judge-api-key", api_key,
        "--concurrency", str(args.judge_concurrency),
        "--force",
    ]
    print(f"[rules-baseline] running judge: {' '.join(cmd[:6])} ...")
    rc = subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode
    if rc != 0:
        print(f"[rules-baseline] judge exited {rc}", file=sys.stderr)
        return rc
    print(f"[rules-baseline] done; results in {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
