#!/usr/bin/env python3
"""
latency_experiment.py — Compare per-question latency: sequential vs token-budget batched.

Samples 50 random questions (seed=42), builds prompts, then:
  Run A: sequential (one-by-one, no concurrency)
  Run B: token-budget batched (60% of KV cache max)

Reports paired comparison: correlation, mean/median/p95, scatter summary.

Usage:
    python3 -m eval.latency_experiment \
        --run-dir  MASim/runs/l_20260408_111046 \
        --model    /models/Qwen3-0.6B \
        --endpoint http://127.0.0.1:8050/v1 \
        --context-length 20000
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from eval.simple_eval import (
    _CHARS_PER_TOKEN,
    _load_tokenizer,
    _count_tokens,
    build_prompt,
    call_llm,
    load_instances,
    load_corpus_sessions,
    load_ego_session_map,
    TokenBudgetSemaphore,
)


def _build_one_prompt(
    inst: dict,
    corpus_sessions: dict,
    ego_session_map: dict,
    max_context_chars: int,
    hard_cap_chars: int,
) -> Tuple[str, str]:
    dim = inst.get("dimension", "")
    tier = inst.get("context_tier", "single_session")
    if tier == "full_ego" or dim == "d11_exception":
        eff = min(max_context_chars * 4, hard_cap_chars) if hard_cap_chars > 0 else max_context_chars * 4
    else:
        eff = max_context_chars
    return build_prompt(inst, corpus_sessions, ego_session_map, max_context_chars=eff)


def run_sequential(
    prompts: List[Tuple[str, str]],
    endpoint: str,
    model: str,
    api_key: str,
    max_tokens: int,
) -> List[dict]:
    """Run A: one request at a time."""
    results = []
    for i, (system, user) in enumerate(prompts):
        raw, meta = call_llm(
            system, user,
            endpoint=endpoint, model=model, api_key=api_key,
            temperature=0.0, max_tokens=max_tokens,
        )
        results.append({
            "idx": i,
            "elapsed_ms": meta.get("elapsed_ms"),
            "ttft_ms": meta.get("ttft_ms"),
            "prompt_tokens": meta.get("prompt_tokens"),
            "completion_tokens": meta.get("completion_tokens"),
        })
        print(f"  seq {i+1}/50  elapsed={meta.get('elapsed_ms', 0):.0f}ms  "
              f"prompt={meta.get('prompt_tokens')}  completion={meta.get('completion_tokens')}",
              file=sys.stderr)
    return results


def run_batched(
    prompts: List[Tuple[str, str]],
    token_counts: List[int],
    token_budget: int,
    endpoint: str,
    model: str,
    api_key: str,
    max_tokens: int,
) -> List[dict]:
    """Run B: token-budget concurrency."""
    sem = TokenBudgetSemaphore(token_budget)
    results: List[Optional[dict]] = [None] * len(prompts)

    def _run(i: int) -> None:
        system, user = prompts[i]
        n_tok = token_counts[i]
        sem.acquire(n_tok)
        try:
            raw, meta = call_llm(
                system, user,
                endpoint=endpoint, model=model, api_key=api_key,
                temperature=0.0, max_tokens=max_tokens,
            )
            results[i] = {
                "idx": i,
                "elapsed_ms": meta.get("elapsed_ms"),
                "ttft_ms": meta.get("ttft_ms"),
                "prompt_tokens": meta.get("prompt_tokens"),
                "completion_tokens": meta.get("completion_tokens"),
            }
        finally:
            sem.release(n_tok)

    done = 0
    with ThreadPoolExecutor(max_workers=200) as pool:
        futures = {pool.submit(_run, i): i for i in range(len(prompts))}
        for fut in as_completed(futures):
            fut.result()
            done += 1
            r = results[futures[fut]]
            if r:
                print(f"  bat {done}/50  elapsed={r['elapsed_ms']:.0f}ms  "
                      f"prompt={r['prompt_tokens']}  completion={r['completion_tokens']}",
                      file=sys.stderr)

    return [r for r in results if r is not None]


def analyze(seq: List[dict], bat: List[dict]) -> None:
    """Paired comparison of sequential vs batched latency."""
    # Align by idx
    seq_by_idx = {r["idx"]: r for r in seq}
    bat_by_idx = {r["idx"]: r for r in bat}
    common = sorted(set(seq_by_idx) & set(bat_by_idx))

    if len(common) < 2:
        print("ERROR: not enough paired results to compare", file=sys.stderr)
        return

    seq_times = [seq_by_idx[i]["elapsed_ms"] for i in common]
    bat_times = [bat_by_idx[i]["elapsed_ms"] for i in common]
    seq_ttft = [seq_by_idx[i]["ttft_ms"] or 0 for i in common]
    bat_ttft = [bat_by_idx[i]["ttft_ms"] or 0 for i in common]

    # Correlation
    n = len(common)
    mean_s = statistics.mean(seq_times)
    mean_b = statistics.mean(bat_times)
    std_s = statistics.stdev(seq_times) if n > 1 else 1
    std_b = statistics.stdev(bat_times) if n > 1 else 1
    if std_s > 0 and std_b > 0:
        cov = sum((s - mean_s) * (b - mean_b) for s, b in zip(seq_times, bat_times)) / (n - 1)
        pearson = cov / (std_s * std_b)
    else:
        pearson = float("nan")

    # Rank correlation (Spearman)
    def _rank(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0.0] * len(vals)
        for r, i in enumerate(order):
            ranks[i] = r + 1
        return ranks

    rank_s = _rank(seq_times)
    rank_b = _rank(bat_times)
    d_sq = sum((a - b) ** 2 for a, b in zip(rank_s, rank_b))
    spearman = 1 - (6 * d_sq) / (n * (n ** 2 - 1)) if n > 1 else float("nan")

    # Diffs
    diffs = [b - s for s, b in zip(seq_times, bat_times)]
    pct_diffs = [(b - s) / s * 100 if s > 0 else 0 for s, b in zip(seq_times, bat_times)]

    print("\n" + "=" * 60)
    print("  LATENCY EXPERIMENT RESULTS")
    print("=" * 60)
    print(f"  Paired samples: {n}")
    print()

    print("  elapsed_ms (end-to-end):")
    print(f"    Sequential  — mean={mean_s:.0f}  median={statistics.median(seq_times):.0f}  "
          f"p95={sorted(seq_times)[int(n*0.95)]:.0f}")
    print(f"    Batched     — mean={mean_b:.0f}  median={statistics.median(bat_times):.0f}  "
          f"p95={sorted(bat_times)[int(n*0.95)]:.0f}")
    print(f"    Diff (bat-seq) — mean={statistics.mean(diffs):+.0f}ms  "
          f"median={statistics.median(diffs):+.0f}ms  "
          f"mean%={statistics.mean(pct_diffs):+.1f}%")
    print()

    print("  ttft_ms (time to first token):")
    print(f"    Sequential  — mean={statistics.mean(seq_ttft):.0f}  "
          f"median={statistics.median(seq_ttft):.0f}")
    print(f"    Batched     — mean={statistics.mean(bat_ttft):.0f}  "
          f"median={statistics.median(bat_ttft):.0f}")
    print()

    print(f"  Correlation (elapsed_ms):")
    print(f"    Pearson  r = {pearson:.4f}")
    print(f"    Spearman ρ = {spearman:.4f}")
    print()

    if abs(statistics.mean(pct_diffs)) > 10:
        print("  ⚠ WARNING: batched latency deviates >10% from sequential on average")
    else:
        print("  ✓ Batched latency within 10% of sequential — timing is reliable")
    print("=" * 60)

    # Per-question scatter (text)
    print("\n  idx | seq_ms | bat_ms | diff_ms | diff_%")
    print("  " + "-" * 50)
    for i in common:
        s = seq_by_idx[i]["elapsed_ms"]
        b = bat_by_idx[i]["elapsed_ms"]
        d = b - s
        p = (d / s * 100) if s > 0 else 0
        print(f"  {i:3d} | {s:6.0f} | {b:6.0f} | {d:+7.0f} | {p:+5.1f}%")


def main():
    ap = argparse.ArgumentParser(description="Latency experiment: sequential vs token-budget batched")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--endpoint", default="http://127.0.0.1:8050/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--context-length", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--n-samples", type=int, default=50)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)

    # Load data
    instances = load_instances(run_dir)
    corpus_sessions = load_corpus_sessions(run_dir)
    ego_session_map = load_ego_session_map(run_dir)

    # Sample
    random.seed(42)
    samples = random.sample(instances, min(args.n_samples, len(instances)))
    print(f"Sampled {len(samples)} questions (seed=42)", file=sys.stderr)

    # Context budget
    overhead_tokens = 300
    remaining = args.context_length - args.max_tokens - overhead_tokens
    budget_tokens = max(4096, remaining)
    hard_cap = args.context_length - overhead_tokens
    budget_tokens = min(budget_tokens, hard_cap)
    max_context_chars = int(budget_tokens * _CHARS_PER_TOKEN)
    hard_cap_chars = int(hard_cap * _CHARS_PER_TOKEN)

    # Build prompts
    tokenizer = _load_tokenizer(args.model)
    if tokenizer:
        print(f"Loaded tokenizer for {args.model}", file=sys.stderr)
    else:
        print(f"No tokenizer — using char estimate", file=sys.stderr)

    prompts: List[Tuple[str, str]] = []
    token_counts: List[int] = []
    for inst in samples:
        system, user = _build_one_prompt(inst, corpus_sessions, ego_session_map,
                                         max_context_chars, hard_cap_chars)
        prompts.append((system, user))
        token_counts.append(_count_tokens(tokenizer, system, user))

    total_tokens = sum(token_counts)
    print(f"Prompt tokens: min={min(token_counts)} max={max(token_counts)} "
          f"mean={total_tokens//len(token_counts)} total={total_tokens}", file=sys.stderr)

    token_budget = int(args.context_length * 0.6)
    print(f"Token budget: {token_budget} (60% of {args.context_length})", file=sys.stderr)

    # Run A: Sequential
    print("\n--- Run A: Sequential ---", file=sys.stderr)
    seq_results = run_sequential(prompts, args.endpoint, args.model, args.api_key, args.max_tokens)

    # Brief pause to let server settle
    time.sleep(2)

    # Run B: Token-budget batched
    print("\n--- Run B: Token-budget batched ---", file=sys.stderr)
    bat_results = run_batched(prompts, token_counts, token_budget,
                              args.endpoint, args.model, args.api_key, args.max_tokens)

    # Analyze
    analyze(seq_results, bat_results)

    # Save raw data
    out = {
        "config": {
            "model": args.model,
            "context_length": args.context_length,
            "token_budget": token_budget,
            "n_samples": len(samples),
            "seed": 42,
        },
        "token_counts": token_counts,
        "sequential": seq_results,
        "batched": bat_results,
    }
    out_path = run_dir / "eval_results" / "latency_experiment.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nRaw data saved to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
