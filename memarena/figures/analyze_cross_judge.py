#!/usr/bin/env python3
"""Analyze cross-judge agreement between Qwen3-235B and GPT-4o-mini.

Reads secondary_judge JSON files and produces:
1. Per-file agreement table (raw agreement, Cohen's κ, accuracies)
2. Per-dimension agreement breakdown
3. Aggregate statistics across all files
4. Per-backend and per-model breakdowns
5. Disagreement analysis (where judges disagree most)

Usage:
    python3 analyze_cross_judge.py
    python3 analyze_cross_judge.py --run-dir MASim/runs/l_20260408_111046
    python3 analyze_cross_judge.py --latex  # output LaTeX table
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


def cohens_kappa(paired):
    """Compute Cohen's kappa from list of (primary, secondary) bool pairs."""
    if not paired:
        return None
    n = len(paired)
    agree = sum(1 for p, s in paired if p == s)
    raw = agree / n
    p1 = sum(1 for p, _ in paired if p) / n
    s1 = sum(1 for _, s in paired if s) / n
    pe = p1 * s1 + (1 - p1) * (1 - s1)
    if (1 - pe) == 0:
        return 1.0
    return (raw - pe) / (1 - pe)


def analyze_file(results):
    """Analyze one file's cross-judge results."""
    paired = []
    by_dim = defaultdict(list)
    disagree_examples = []

    for r in results:
        if r is None:
            continue
        pc = r.get("primary_correct")
        sc = r.get("secondary_correct")
        if pc is None or sc is None:
            continue
        paired.append((pc, sc))
        dim = r.get("dimension", "unknown")
        by_dim[dim].append((pc, sc))
        if pc != sc:
            disagree_examples.append(r)

    if not paired:
        return None

    n = len(paired)
    agree = sum(1 for p, s in paired if p == s)

    return {
        "n": n,
        "raw_agreement": round(agree / n, 4),
        "cohens_kappa": round(cohens_kappa(paired), 4),
        "primary_accuracy": round(sum(1 for p, _ in paired if p) / n, 4),
        "secondary_accuracy": round(sum(1 for _, s in paired if s) / n, 4),
        "n_agree": agree,
        "n_disagree": n - agree,
        "both_correct": sum(1 for p, s in paired if p and s),
        "both_incorrect": sum(1 for p, s in paired if not p and not s),
        "primary_only": sum(1 for p, s in paired if p and not s),
        "secondary_only": sum(1 for p, s in paired if not p and s),
        "by_dimension": {
            dim: {
                "n": len(pairs),
                "agreement": round(sum(1 for p, s in pairs if p == s) / len(pairs), 4),
                "kappa": round(cohens_kappa(pairs), 4) if cohens_kappa(pairs) is not None else None,
                "primary_acc": round(sum(1 for p, _ in pairs if p) / len(pairs), 4),
                "secondary_acc": round(sum(1 for _, s in pairs if s) / len(pairs), 4),
            }
            for dim, pairs in sorted(by_dim.items())
        },
        "disagree_examples": disagree_examples[:5],
    }


def parse_filename(key):
    """Extract backend and model from filename key."""
    # e.g. "inmem_evaluation_results_rag_8b" -> backend=rag, model=8b
    parts = key.replace("evaluation_results_", "").split("_")
    if len(parts) >= 3:
        backend_dir = parts[0]  # inmem, oracle, vanilla
        model = parts[-1]  # 0_6b, 3b, 7b, 8b, 32b
        backend = parts[1] if len(parts) > 2 else backend_dir
        return backend, model, backend_dir
    return "unknown", "unknown", "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        default=os.getenv("MEMARENA_RUN_DIR", "MASim/runs/l_20260408_111046"),
    )
    parser.add_argument("--latex", action="store_true", help="Output LaTeX table")
    args = parser.parse_args()

    # Find the secondary judge results
    judge_dir = Path(args.run_dir) / "eval_results" / "secondary_judge"
    judge_files = list(judge_dir.glob("secondary_judge_*.json"))

    if not judge_files:
        print(
            f"[cross-judge] skipped: no secondary judge results in {judge_dir}. "
            "Run an evaluation with --secondary-judge-preset to generate this optional analysis."
        )
        return

    for jf in judge_files:
        print(f"Loading {jf.name}...")
        data = json.load(open(jf))
        model = data.get("model", "unknown")

        print(f"\n{'='*80}")
        print(f"  CROSS-JUDGE AGREEMENT: Qwen3-235B vs {model}")
        print(f"  Corpus: {args.run_dir}")
        print(f"  Timestamp: {data.get('timestamp', '?')}")
        print(f"{'='*80}\n")

        # ── Per-file table ──
        all_paired = []
        all_by_dim = defaultdict(list)
        by_backend = defaultdict(list)
        by_model_size = defaultdict(list)

        print(f"{'File':<55} {'n':>5} {'Agree':>6} {'κ':>6} {'1°':>6} {'2°':>6}")
        print("-" * 90)

        for key, results in sorted(data.get("details", {}).items()):
            stats = analyze_file(results)
            if stats is None:
                print(f"{key:<55} {'(no data)':>5}")
                continue

            print(f"{key:<55} {stats['n']:>5} {stats['raw_agreement']:>6.3f} "
                  f"{stats['cohens_kappa']:>6.3f} {stats['primary_accuracy']:>6.3f} "
                  f"{stats['secondary_accuracy']:>6.3f}")

            # Collect for aggregation
            for r in results:
                if r and r.get("primary_correct") is not None and r.get("secondary_correct") is not None:
                    pair = (r["primary_correct"], r["secondary_correct"])
                    all_paired.append(pair)
                    dim = r.get("dimension", "unknown")
                    all_by_dim[dim].append(pair)

                    backend, model_size, _ = parse_filename(key)
                    by_backend[backend].append(pair)
                    by_model_size[model_size].append(pair)

        # ── Aggregate ──
        if not all_paired:
            print("\n[cross-judge] skipped: secondary judge files contained no paired rows")
            continue

        print(f"\n{'AGGREGATE':<55} {len(all_paired):>5} "
              f"{sum(1 for p,s in all_paired if p==s)/len(all_paired):>6.3f} "
              f"{cohens_kappa(all_paired):>6.3f} "
              f"{sum(1 for p,_ in all_paired if p)/len(all_paired):>6.3f} "
              f"{sum(1 for _,s in all_paired if s)/len(all_paired):>6.3f}")

        # ── Confusion matrix ──
        both_c = sum(1 for p, s in all_paired if p and s)
        both_i = sum(1 for p, s in all_paired if not p and not s)
        p_only = sum(1 for p, s in all_paired if p and not s)
        s_only = sum(1 for p, s in all_paired if not p and s)
        n = len(all_paired)

        print(f"\n  Confusion Matrix (n={n}):")
        print(f"  {'':>20} {'2° Correct':>12} {'2° Incorrect':>12}")
        print(f"  {'1° Correct':<20} {both_c:>12} {p_only:>12}")
        print(f"  {'1° Incorrect':<20} {s_only:>12} {both_i:>12}")

        # ── Per-dimension ──
        print(f"\n  Per-Dimension Agreement:")
        print(f"  {'Dimension':<25} {'n':>5} {'Agree':>6} {'κ':>6} {'1°':>6} {'2°':>6}")
        print(f"  {'-'*60}")
        for dim in sorted(all_by_dim.keys()):
            pairs = all_by_dim[dim]
            nn = len(pairs)
            ag = sum(1 for p, s in pairs if p == s) / nn
            kp = cohens_kappa(pairs)
            p_acc = sum(1 for p, _ in pairs if p) / nn
            s_acc = sum(1 for _, s in pairs if s) / nn
            print(f"  {dim:<25} {nn:>5} {ag:>6.3f} {kp:>6.3f} {p_acc:>6.3f} {s_acc:>6.3f}")

        # ── Per-backend ──
        print(f"\n  Per-Backend Agreement:")
        print(f"  {'Backend':<15} {'n':>6} {'Agree':>6} {'κ':>6}")
        print(f"  {'-'*40}")
        for backend in sorted(by_backend.keys()):
            pairs = by_backend[backend]
            nn = len(pairs)
            ag = sum(1 for p, s in pairs if p == s) / nn
            kp = cohens_kappa(pairs)
            print(f"  {backend:<15} {nn:>6} {ag:>6.3f} {kp:>6.3f}")

        # ── Per-model-size ──
        print(f"\n  Per-Model-Size Agreement:")
        print(f"  {'Model':<15} {'n':>6} {'Agree':>6} {'κ':>6}")
        print(f"  {'-'*40}")
        for ms in sorted(by_model_size.keys()):
            pairs = by_model_size[ms]
            nn = len(pairs)
            ag = sum(1 for p, s in pairs if p == s) / nn
            kp = cohens_kappa(pairs)
            print(f"  {ms:<15} {nn:>6} {ag:>6.3f} {kp:>6.3f}")

        # ── LaTeX table ──
        if args.latex:
            print(f"\n  LaTeX table:")
            print(r"  \begin{table}[h]")
            print(r"  \centering\small")
            print(r"  \caption{Cross-judge agreement: Qwen3-235B vs.\ GPT-4o-mini.}")
            print(r"  \label{tab:cross-judge}")
            print(r"  \begin{tabular}{@{}lcccc@{}}")
            print(r"  \toprule")
            print(r"  Dimension & $n$ & Agreement & Cohen's $\kappa$ & $\Delta$Accuracy \\")
            print(r"  \midrule")
            for dim in sorted(all_by_dim.keys()):
                pairs = all_by_dim[dim]
                nn = len(pairs)
                ag = sum(1 for p, s in pairs if p == s) / nn
                kp = cohens_kappa(pairs)
                p_acc = sum(1 for p, _ in pairs if p) / nn
                s_acc = sum(1 for _, s in pairs if s) / nn
                delta = abs(p_acc - s_acc)
                dim_label = dim.replace("_", r"\_")
                print(f"  {dim_label} & {nn} & {ag:.3f} & {kp:.3f} & {delta:.3f} \\\\")
            # Aggregate row
            ag_all = sum(1 for p, s in all_paired if p == s) / len(all_paired)
            kp_all = cohens_kappa(all_paired)
            print(r"  \midrule")
            print(f"  \\textbf{{Overall}} & {len(all_paired)} & {ag_all:.3f} & {kp_all:.3f} & --- \\\\")
            print(r"  \bottomrule")
            print(r"  \end{tabular}")
            print(r"  \end{table}")

        print()


if __name__ == "__main__":
    main()
