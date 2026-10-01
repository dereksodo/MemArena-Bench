#!/usr/bin/env python3
"""Report LLM calls whose wall time exceeds 3x the median for their phase.

Usage:
    python -m MASim.tools.report_slow_calls <run_dir> [--threshold 3.0] [--output slow_calls.jsonl]

Reads llm_calls_detail.jsonl (full prompts/completions) from the run directory.
Falls back to llm_timing.jsonl (metadata only) if detail log doesn't exist.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Report slow LLM calls (>Nx median)")
    parser.add_argument("run_dir", help="Path to the run directory")
    parser.add_argument("--threshold", type=float, default=3.0,
                        help="Multiplier over median to flag (default: 3.0)")
    parser.add_argument("--output", default=None,
                        help="Output file path (default: <run_dir>/slow_calls.jsonl)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    detail_path = run_dir / "llm_calls_detail.jsonl"
    timing_path = run_dir / "llm_timing.jsonl"

    # Pick the best available source
    if detail_path.exists() and detail_path.stat().st_size > 0:
        src_path = detail_path
        has_detail = True
    elif timing_path.exists():
        src_path = timing_path
        has_detail = False
    else:
        print(f"Error: no llm_calls_detail.jsonl or llm_timing.jsonl in {run_dir}",
              file=sys.stderr)
        sys.exit(1)

    # Load all records
    records = []
    with open(src_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: skipping malformed line {line_no}", file=sys.stderr)

    if not records:
        print("No records found.", file=sys.stderr)
        sys.exit(0)

    # Group by phase
    by_phase: dict[str, list] = {}
    for rec in records:
        phase = rec.get("phase", "unknown")
        by_phase.setdefault(phase, []).append(rec)

    # Compute medians
    medians: dict[str, float] = {}
    for phase, recs in by_phase.items():
        times = [r["wall_s"] for r in recs]
        medians[phase] = statistics.median(times)

    # Find slow calls
    slow = []
    for rec in records:
        phase = rec.get("phase", "unknown")
        median = medians[phase]
        if median > 0 and rec["wall_s"] > median * args.threshold:
            entry = {
                "phase": phase,
                "wall_s": rec["wall_s"],
                "median_s": round(median, 4),
                "ratio": round(rec["wall_s"] / median, 2),
                "prompt_tokens": rec.get("prompt_tokens", 0),
                "completion_tokens": rec.get("completion_tokens", 0),
                "ts": rec.get("ts", 0),
            }
            # Per-field token counts (whitespace-split approximation)
            if has_detail:
                sys_text = rec.get("system_prompt", "")
                usr_text = rec.get("user_prompt", "")
                cmp_text = rec.get("completion", "")
                entry["system_prompt_tokens"] = len(sys_text.split()) if sys_text else 0
                entry["user_prompt_tokens"] = len(usr_text.split()) if usr_text else 0
                entry["completion_word_tokens"] = len(cmp_text.split()) if cmp_text else 0
            # Copy all tag fields (day_idx, step, session_type, batch_idx, etc.)
            for k in ("day_idx", "step", "session_type", "batch_size", "batch_idx"):
                if k in rec:
                    entry[k] = rec[k]
            # Include full prompts/completion if available
            if has_detail:
                entry["system_prompt"] = sys_text
                entry["user_prompt"] = usr_text
                entry["completion"] = cmp_text
            slow.append(entry)

    slow.sort(key=lambda x: x["ratio"], reverse=True)

    # Output
    out_path = Path(args.output) if args.output else run_dir / "slow_calls.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for entry in slow:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Print summary
    print(f"Source: {src_path}")
    print(f"Total calls: {len(records)}")
    print(f"Slow calls (>{args.threshold}x median): {len(slow)}")
    print(f"Output: {out_path}")
    print()
    print(f"{'Phase':<35s} {'Median':>8s} {'Count':>6s} {'Slow':>5s}")
    print("-" * 60)
    for phase in sorted(by_phase.keys()):
        n_total = len(by_phase[phase])
        n_slow = sum(1 for s in slow if s["phase"] == phase)
        print(f"{phase:<35s} {medians[phase]:>7.2f}s {n_total:>6d} {n_slow:>5d}")

    if slow:
        print()
        print(f"Top 10 slowest (by ratio):")
        for entry in slow[:10]:
            line = (f"  {entry['phase']:<30s} {entry['wall_s']:>7.1f}s "
                    f"({entry['ratio']:.1f}x median={entry['median_s']:.2f}s) "
                    f"tokens={entry['prompt_tokens']}+{entry['completion_tokens']}")
            if "system_prompt_tokens" in entry:
                line += (f"  [sys={entry['system_prompt_tokens']} "
                         f"usr={entry['user_prompt_tokens']} "
                         f"out={entry['completion_word_tokens']}]")
            print(line)


if __name__ == "__main__":
    main()
