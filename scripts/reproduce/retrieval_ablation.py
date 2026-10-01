#!/usr/bin/env python3
"""
Retrieval Budget Ablation Analysis (TODO #8)

Analyze the effect of varying RAG top_k on per-dimension accuracy.
Reads evaluation_results_*.json files from rag_ablation/ subdirectory.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_results(path: Path) -> dict:
    """Load evaluation results JSON."""
    return json.loads(path.read_text())


def bootstrap_ci(scores: np.ndarray, n_boot: int = 5000, seed: int = 42) -> tuple[float, float]:
    """Compute 95% bootstrap CI for mean."""
    rng = np.random.default_rng(seed)
    n = len(scores)
    if n == 0:
        return (0.0, 0.0)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = scores[idx].mean()
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def analyze_topk(eval_dir: Path, topk_values: list[int]) -> dict:
    """Load results for each top_k and compute per-dimension accuracy."""
    results = {}
    for k in topk_values:
        # Try multiple naming patterns and subdirectories
        candidates = [
            eval_dir / f"evaluation_results_rag_8b_topk{k}.json",
            eval_dir / "inmem" / f"evaluation_results_rag_8b_topk{k}.json",
            eval_dir / "rag_ablation" / f"evaluation_results_rag_8b_topk{k}.json",
            eval_dir / "rag_ablation" / "inmem" / f"evaluation_results_rag_8b_topk{k}.json",
        ]
        found = None
        for c in candidates:
            if c.exists():
                found = c
                break

        # Also try recursive glob
        if found is None:
            matches = list(eval_dir.rglob(f"evaluation_results*topk{k}*.json"))
            # Prefer non-runs/ paths
            non_runs = [m for m in matches if "/runs/" not in str(m)]
            if non_runs:
                found = non_runs[0]
            elif matches:
                found = matches[0]

        if found is None:
            print(f"  [SKIP] No results for top_k={k}", file=sys.stderr)
            continue

        print(f"  Loading top_k={k} from {found.name}", file=sys.stderr)
        data = load_results(found)
        details = data.get("details", [])

        # Per-dimension breakdown
        dim_map = {
            "d1": "d1_conflict", "d2": "d2_anaphora", "d3": "d3_confabulation",
            "d4": "d4_permission", "d5": "d5_cloze", "d6": "d6_metadata",
            "d7": "d7_qa", "d8": "d8_temporal", "d9": "d9_negation",
            "d10": "d10_counterfactual", "d11": "d11_exception",
        }
        by_dim = defaultdict(list)
        for d in details:
            score = d.get("judge_score", d.get("score", 0.0))
            if score is None:
                score = 0.0
            # Try explicit dimension field first, then extract from question_id
            dim = d.get("dimension")
            if not dim or dim == "unknown":
                qid = d.get("question_id", d.get("instance_id", ""))
                prefix = qid.split("_")[0] if "_" in qid else ""
                dim = dim_map.get(prefix, d.get("question_type", "unknown"))
            by_dim[dim].append(score)

        all_scores = [d.get("judge_score", d.get("score", 0.0)) or 0.0 for d in details]
        all_arr = np.array(all_scores)
        ci = bootstrap_ci(all_arr)

        dim_results = {}
        for dim in sorted(by_dim):
            arr = np.array(by_dim[dim])
            dim_ci = bootstrap_ci(arr)
            dim_results[dim] = {
                "n": len(arr),
                "accuracy": float(arr.mean()),
                "ci_lo": dim_ci[0],
                "ci_hi": dim_ci[1],
            }

        results[k] = {
            "file": str(found),
            "total": len(details),
            "overall_accuracy": float(all_arr.mean()),
            "overall_ci": {"lo": ci[0], "hi": ci[1]},
            "per_dimension": dim_results,
        }

    return results


def format_markdown(results: dict, topk_values: list[int]) -> str:
    """Format results as markdown tables."""
    lines = [
        "# Retrieval Budget Ablation (top_k)",
        "",
        "Per-dimension accuracy as RAG top_k varies from 3 to 20.",
        "Model: Qwen3-8B-AWQ on MemArena-L scale.",
        "",
    ]

    # Overall accuracy table
    lines.append("## Overall Accuracy")
    lines.append("")
    present_k = [k for k in topk_values if k in results]
    header = "| top_k | " + " | ".join(str(k) for k in present_k) + " |"
    sep = "|-------|" + "|".join("-----" for _ in present_k) + "|"
    vals = "| Accuracy | " + " | ".join(
        f"{results[k]['overall_accuracy']:.3f}" for k in present_k
    ) + " |"
    cis = "| 95% CI | " + " | ".join(
        f"[{results[k]['overall_ci']['lo']:.3f}, {results[k]['overall_ci']['hi']:.3f}]"
        for k in present_k
    ) + " |"
    ns = "| N | " + " | ".join(str(results[k]['total']) for k in present_k) + " |"
    lines.extend([header, sep, vals, cis, ns, ""])

    # Per-dimension table
    if present_k:
        all_dims = sorted(
            set().union(*(results[k]["per_dimension"].keys() for k in present_k))
        )

        lines.append("## Per-Dimension Accuracy")
        lines.append("")
        header = "| Dimension | " + " | ".join(f"k={k}" for k in present_k) + " |"
        sep = "|-----------|" + "|".join("-----" for _ in present_k) + "|"
        lines.extend([header, sep])

        for dim in all_dims:
            cells = []
            for k in present_k:
                if dim in results[k]["per_dimension"]:
                    d = results[k]["per_dimension"][dim]
                    cells.append(f"{d['accuracy']:.3f}")
                else:
                    cells.append("—")
            lines.append(f"| {dim} | " + " | ".join(cells) + " |")
        lines.append("")

        # Find best k per dimension
        lines.append("## Best top_k per Dimension")
        lines.append("")
        lines.append("| Dimension | Best k | Accuracy | Delta vs k=10 |")
        lines.append("|-----------|--------|----------|---------------|")
        for dim in all_dims:
            best_k = None
            best_acc = -1
            acc_at_10 = None
            for k in present_k:
                if dim in results[k]["per_dimension"]:
                    acc = results[k]["per_dimension"][dim]["accuracy"]
                    if acc > best_acc:
                        best_acc = acc
                        best_k = k
                    if k == 10:
                        acc_at_10 = acc
            if best_k is not None:
                delta = f"{best_acc - acc_at_10:+.3f}" if acc_at_10 is not None else "—"
                lines.append(f"| {dim} | {best_k} | {best_acc:.3f} | {delta} |")
        lines.append("")

    # Paper snippet
    lines.append("---")
    lines.append("")
    lines.append("## Paper Text Snippet")
    lines.append("")
    if len(present_k) >= 3:
        accs = {k: results[k]["overall_accuracy"] for k in present_k}
        best_k = max(accs, key=accs.get)
        worst_k = min(accs, key=accs.get)
        spread = accs[best_k] - accs[worst_k]
        lines.append(
            f"Varying top\\_k from {min(present_k)} to {max(present_k)}, "
            f"overall accuracy ranges from {accs[worst_k]:.1%} (k={worst_k}) "
            f"to {accs[best_k]:.1%} (k={best_k}), a spread of {spread:.1%} points. "
        )
        if 10 in accs:
            lines.append(
                f"The default k=10 achieves {accs[10]:.1%}, "
                f"{'matching' if abs(accs[10] - accs[best_k]) < 0.005 else 'within ' + f'{abs(accs[10] - accs[best_k]):.1%} of'} "
                f"the best setting."
            )
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Retrieval budget ablation analysis")
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=Path("MASim/runs/l_20260408_111046/eval_results"),
    )
    parser.add_argument(
        "--top-k-values",
        type=int,
        nargs="+",
        default=[3, 5, 10, 15, 20],
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    results = analyze_topk(args.eval_dir, args.top_k_values)

    report = format_markdown(results, args.top_k_values)
    print(report)

    out_path = args.output or (args.eval_dir / "retrieval_ablation.md")
    out_path.write_text(report)
    print(f"\nReport written to {out_path}", file=sys.stderr)

    json_path = out_path.with_suffix(".json")
    # Convert for JSON serialization
    json_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"Raw data written to {json_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
