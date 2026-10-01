#!/usr/bin/env python3
"""
quick_test.py — Fast smoke test for modified eval dimensions.

Runs the first N instances of each specified dimension through the LLM
and prints a per-dimension accuracy summary.

Usage:
    python3 eval/quick_test.py \
        --run-dir MASim/runs/1m_20260308_221634/ \
        --dims d1_conflict d8_temporal d9_negation d11_exception d4_permission \
        --n 50 \
        --model qwen3-235B-A22B \
        --endpoint http://127.0.0.1:8000/v1

Defaults:
    dims   = d1_conflict d4_permission d8_temporal d9_negation d11_exception
    n      = 50
    model  = qwen3-235B-A22B
    endpoint = http://127.0.0.1:8000/v1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import List

# Re-use everything from simple_eval
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.simple_eval import (
    build_prompt,
    call_llm,
    eval_one,
    extract_answer,
    load_corpus_sessions,
    load_ego_session_map,
    load_instances,
    score,
    summarize,
)

_DEFAULT_DIMS = [
    "d1_conflict",
    "d4_permission",
    "d8_temporal",
    "d9_negation",
    "d11_exception",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quick smoke test for modified eval dimensions",
    )
    parser.add_argument("--run-dir", required=True, help="MASim run directory")
    parser.add_argument(
        "--dims", nargs="*", default=_DEFAULT_DIMS,
        help=f"Dimensions to test (default: {' '.join(_DEFAULT_DIMS)})",
    )
    parser.add_argument("--n", type=int, default=50, help="Max instances per dimension (default: 50)")
    parser.add_argument("--model", default="qwen3-235B-A22B")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--context-length", type=int, default=16384)
    parser.add_argument("--judge", action="store_true", default=True,
                        help="Use LLM-as-judge scoring (default: on)")
    parser.add_argument("--no-judge", dest="judge", action="store_false",
                        help="Disable LLM-as-judge, use token_f1 only")
    parser.add_argument("--judge-model", default=None,
                        help="Judge model (default: same as --model)")
    parser.add_argument("--judge-endpoint", default=None,
                        help="Judge endpoint (default: same as --endpoint)")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: <run-dir>/eval_results/quick_test.json)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)

    # Compute context budget
    overhead_tokens = 300
    remaining = args.context_length - args.max_tokens - overhead_tokens
    budget_tokens = max(4096, remaining)
    budget_tokens = min(budget_tokens, args.context_length - overhead_tokens)
    max_context_chars = int(budget_tokens * 3.0)

    # Load data
    instances = load_instances(run_dir, args.dims, args.n)
    corpus_sessions = load_corpus_sessions(run_dir)
    ego_session_map = load_ego_session_map(run_dir)

    by_dim: dict = defaultdict(int)
    for inst in instances:
        by_dim[inst["dimension"]] += 1

    print(f"Quick test: {len(instances)} instances from {run_dir}", file=sys.stderr)
    for dim, n in sorted(by_dim.items()):
        print(f"  {dim:<25} {n}", file=sys.stderr)
    print(file=sys.stderr)

    judge_model = args.judge_model or args.model
    judge_endpoint = args.judge_endpoint or args.endpoint

    if args.judge:
        print(f"Judge: {judge_model} @ {judge_endpoint}", file=sys.stderr)
    else:
        print("Judge: disabled (token_f1 only)", file=sys.stderr)
    print(file=sys.stderr)

    # Wrapper that captures the prompt alongside eval_one output
    def eval_one_with_prompt(inst):
        system, user = build_prompt(
            inst, corpus_sessions, ego_session_map,
            max_context_chars=max_context_chars,
        )
        result = eval_one(
            inst,
            endpoint=args.endpoint, model=args.model, api_key=args.api_key,
            temperature=args.temperature, max_tokens=args.max_tokens,
            use_judge=args.judge,
            judge_model=judge_model,
            judge_endpoint=judge_endpoint,
            corpus_sessions=corpus_sessions,
            ego_session_map=ego_session_map,
            max_context_chars=max_context_chars,
        )
        result["prompt_system"] = system
        result["prompt_user"] = user
        return result

    # Run evaluation
    results: List[dict] = []
    done = 0
    start = time.monotonic()

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(eval_one_with_prompt, inst): inst
            for inst in instances
        }
        for fut in as_completed(futures):
            try:
                result = fut.result()
            except Exception as exc:
                inst = futures[fut]
                result = {
                    "instance_id": inst.get("instance_id", ""),
                    "dimension": inst.get("dimension", ""),
                    "correct": False,
                    "score": 0.0,
                    "error": str(exc),
                }
            results.append(result)
            done += 1
            if done % 10 == 0 or done == len(instances):
                acc = sum(r["correct"] for r in results) / done
                elapsed = time.monotonic() - start
                rate = done / elapsed if elapsed > 0 else 0
                remaining_n = len(instances) - done
                eta = remaining_n / rate if rate > 0 else 0
                eta_str = f"  ETA {eta:.0f}s" if done < len(instances) else ""
                print(
                    f"  {done}/{len(instances)}  acc={acc:.1%}"
                    f"  ({elapsed:.0f}s{eta_str})",
                    file=sys.stderr,
                )

    elapsed_total = time.monotonic() - start
    summary = summarize(results)

    # Print summary
    print("\n" + "=" * 60)
    print(f"  Quick Test Results  ({args.model})")
    print(f"  Run dir : {run_dir}")
    print(f"  N/dim   : {args.n}")
    print(f"  Elapsed : {elapsed_total:.0f}s")
    print("=" * 60)
    print(f"  Overall : {summary['accuracy']:.1%}  ({summary['correct']}/{summary['total']})")
    print(f"  Mean F1 : {summary['mean_f1']:.3f}")
    print()
    print("  By dimension:")
    for dim, s in summary["by_dimension"].items():
        bar = "#" * int(s["accuracy"] * 20)
        print(f"    {dim:<25} {s['accuracy']:>5.1%}  {bar}  ({s['correct']}/{s['n']})")
    print("=" * 60)

    # Save results (JSON)
    output_path = Path(args.output) if args.output else run_dir / "eval_results" / "quick_test.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary,
        "model": args.model,
        "dims_tested": args.dims,
        "n_per_dim": args.n,
        "run_dir": str(run_dir),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": round(elapsed_total, 1),
        "results": results,
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n  Saved to: {output_path}")

    # Save detailed human-readable log (one file per dimension)
    detail_dir = output_path.parent / "quick_test_details"
    detail_dir.mkdir(parents=True, exist_ok=True)
    by_dim_results: dict = defaultdict(list)
    for r in results:
        by_dim_results[r["dimension"]].append(r)

    for dim, dim_results in sorted(by_dim_results.items()):
        lines: List[str] = []
        dim_correct = sum(r["correct"] for r in dim_results)
        lines.append(f"{'=' * 80}")
        lines.append(f"  {dim}  —  {dim_correct}/{len(dim_results)} correct")
        lines.append(f"{'=' * 80}\n")

        for i, r in enumerate(dim_results, 1):
            status = "CORRECT" if r.get("correct") else "WRONG"
            lines.append(f"--- [{i}/{len(dim_results)}] {r.get('instance_id', '?')}  [{status}]  score={r.get('score', 0):.3f}  method={r.get('scoring_method', '?')} ---\n")
            lines.append(f"QUERY:\n{r.get('query', '(none)')}\n")
            lines.append(f"GOLD:\n{r.get('gold', '(none)')}\n")
            lines.append(f"PREDICTION:\n{r.get('prediction', '(none)')}\n")
            lines.append(f"RAW LLM RESPONSE:\n{r.get('raw_response', '(none)')}\n")
            sys_prompt = r.get("prompt_system", "(not captured)")
            usr_prompt = r.get("prompt_user", "(not captured)")
            lines.append(f"SYSTEM PROMPT:\n{sys_prompt}\n")
            lines.append(f"USER PROMPT:\n{usr_prompt}\n")
            lines.append("")

        detail_path = detail_dir / f"{dim}.txt"
        detail_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"  Details : {detail_dir}/")


if __name__ == "__main__":
    main()
