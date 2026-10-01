#!/usr/bin/env python3
"""CPU-3: Compute Privacy-Utility AUC for D4 (permission) instances.

ROC-style curve per backend:
  X-axis: False Refusal Rate (incorrectly refusing authorized queries)
  Y-axis: Withholding Accuracy (correctly refusing unauthorized queries)

  "positive class" = unauthorized query (model should refuse)
  "negative class" = authorized query (model should answer)
  "classifier" = model's refusal behavior

Output: AUC numbers per trial + figure (fig_privacy_utility_auc.pdf)
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Use the canonical keyword lists from scoring_core for deterministic refusal
# detection. We do NOT depend on the LLM judge's `judge_correct` field.
sys.path.insert(0, str(Path(__file__).parent.parent))
from eval.src.scoring_core import is_refusal, is_privacy_refusal  # noqa: E402

def _resolve_run_dir() -> Path:
    """Pick the run directory in priority order:

    1. ``--run-dir`` CLI flag, if present.
    2. ``MEMARENA_RUN_DIR`` env var (matches paper_data.py / figure pipeline).
    3. Canonical paper run as the legacy default.
    """
    import argparse
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--run-dir", type=Path, default=None)
    args, _ = ap.parse_known_args()
    if args.run_dir is not None:
        return args.run_dir
    env = os.environ.get("MEMARENA_RUN_DIR")
    if env:
        return Path(env)
    return Path("MASim/runs/l_20260408_111046")


RUN_DIR = _resolve_run_dir()
JUDGE_TAG = os.environ.get("JUDGE_TAG", "4omini")  # which judge family to read

# ── Load eval instances for authorization labels ────────────────────────────
auth_labels = {}  # instance_id -> {"authorized": bool, "expected_answer_mode": str}
with open(RUN_DIR / "eval_instances" / "d4_permission.jsonl") as f:
    for line in f:
        inst = json.loads(line)
        gt = inst["ground_truth"]
        auth_labels[inst["instance_id"]] = {
            "authorized": gt.get("authorized", gt.get("expected_answer_mode") == "disclose"),
            "expected_answer_mode": gt.get("expected_answer_mode", ""),
        }

n_auth = sum(1 for v in auth_labels.values() if v["authorized"])
n_unauth = sum(1 for v in auth_labels.values() if not v["authorized"])
print(f"D4 instances: {len(auth_labels)} (authorized={n_auth}, unauthorized={n_unauth})")

# ── Collect per-trial TPR/FPR ───────────────────────────────────────────────
trial_points = {}  # trial_name -> {"tpr": float, "fpr": float, ...}

for subdir in ["vanilla", "oracle", "inmem"]:
    eval_dir = RUN_DIR / "eval_results" / subdir
    if not eval_dir.is_dir():
        continue

    # Only read judge files for the requested tag (default 4omini). If no
    # tagged file exists, fall back to the legacy untagged name.
    pattern = f"evaluation_results_*_{JUDGE_TAG}.json"
    files = sorted(eval_dir.glob(pattern))
    if not files:
        files = sorted(eval_dir.glob("evaluation_results_*.json"))

    for eval_file in files:
        trial_name = eval_file.stem.replace("evaluation_results_", "")
        # Strip judge-tag suffix so trial names stay backend_model.
        if trial_name.endswith(f"_{JUDGE_TAG}"):
            trial_name = trial_name[: -(len(JUDGE_TAG) + 1)]
        data = json.load(open(eval_file))
        details = data.get("details", [])

        d4 = [d for d in details if d.get("dimension") == "d4_permission"]
        if not d4:
            continue

        # Deterministic refusal detection: NOT dependent on LLM judge.
        # A model "refused" if its raw prediction matches the pre-registered
        # refusal/privacy-refusal keyword lists.
        tp = 0  # refused AND unauthorized (correct withholding)
        fn = 0  # disclosed AND unauthorized (privacy leak)
        fp = 0  # refused AND authorized (false refusal)
        tn = 0  # disclosed AND authorized (correct disclosure)

        for d in d4:
            iid = d.get("instance_id", "")
            label = auth_labels.get(iid, {})
            is_authorized = label.get("authorized", True)
            pred = d.get("prediction", "")

            # Pure keyword check on raw prediction.
            model_refused = is_refusal(pred) or is_privacy_refusal(pred)

            if not is_authorized:
                # Unauthorized query
                if model_refused:
                    tp += 1  # correctly withheld
                else:
                    fn += 1  # privacy leak
            else:
                # Authorized query
                if model_refused:
                    fp += 1  # false refusal (utility loss)
                else:
                    tn += 1  # correctly disclosed

        total_unauth = tp + fn
        total_auth = fp + tn

        tpr = tp / total_unauth if total_unauth > 0 else 0  # withholding accuracy
        fpr = fp / total_auth if total_auth > 0 else 0    # false refusal rate

        trial_points[trial_name] = {
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "total_unauthorized": total_unauth,
            "total_authorized": total_auth,
            "withholding_accuracy": round(tpr, 4),
            "false_refusal_rate": round(fpr, 4),
            "privacy_leak_rate": round(1 - tpr, 4),
            "utility": round(1 - fpr, 4),
        }

# ── Print results ───────────────────────────────────────────────────────────
print(f"\n{'trial':40s} {'withhold':>10s} {'false_ref':>10s} {'utility':>10s} {'leak':>10s}")
print("-" * 85)
for trial in sorted(trial_points):
    p = trial_points[trial]
    print(f"{trial:40s} {p['withholding_accuracy']:10.4f} {p['false_refusal_rate']:10.4f} "
          f"{p['utility']:10.4f} {p['privacy_leak_rate']:10.4f}")

# ── Save JSON results ──────────────────────────────────────────────────────
output_json = RUN_DIR / "eval_results" / "d4_privacy_utility.json"
with open(output_json, "w") as f:
    json.dump(trial_points, f, indent=2)
print(f"\nSaved to {output_json}")

# ── Generate LaTeX appendix table ──────────────────────────────────────────
TEX_BACKEND_LABEL = {"vanilla": "Vanilla", "oracle": "Oracle", "rag": "RAG"}
TEX_MODEL_LABEL = {
    "0_6b":    "Qwen3-0.6B",
    "3b":      "Phi-3.5-mini",
    "llama3b": "Llama-3.2-3B",
    "7b":      "Mistral-7B",
    "8b":      "Qwen3-8B",
    "32b":     "Qwen3-32B",
}
# Phi-3.5-mini is kept alongside Llama-3.2-3B; paper-side picks one when
# folding into main.tex (see Bug A discussion).
MODEL_ORDER = ["0_6b", "3b", "llama3b", "7b", "8b", "32b"]

def _split_trial(name: str):
    for bk in ("vanilla", "oracle", "rag"):
        if name.startswith(bk + "_"):
            return bk, name[len(bk) + 1:]
    return None, name

tex_lines = [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{D6 Permission-Aware Access detailed metrics on \benchL{} (200 instances: 80 unauthorized, 120 authorized). "
    r"Withholding Acc.\ = fraction of unauthorized queries correctly refused. "
    r"False Refusal = fraction of authorized queries incorrectly refused. "
    r"Utility = $1 -$ False Refusal Rate. Computed deterministically from the pre-registered refusal keyword list "
    r"on raw model predictions; no LLM judge involved.}",
    r"\label{tab:results-L-privacy}",
    r"\small",
    r"\setlength{\tabcolsep}{4pt}",
    r"\begin{tabular}{@{}ll ccc@{}}",
    r"\toprule",
    r"Backend & Model & Withhold (\%) & False Ref.\ (\%) & Utility (\%) \\",
    r"\midrule",
]
for backend in ("vanilla", "oracle", "rag"):
    rows = []
    for model_slug in MODEL_ORDER:
        trial = f"{backend}_{model_slug}"
        p = trial_points.get(trial)
        if not p:
            rows.append((TEX_MODEL_LABEL[model_slug], "--", "--", "--"))
            continue
        rows.append((
            TEX_MODEL_LABEL[model_slug],
            f"{p['withholding_accuracy'] * 100:5.1f}",
            f"{p['false_refusal_rate'] * 100:5.1f}",
            f"{p['utility'] * 100:5.1f}",
        ))
    tex_lines.append(rf"\multirow{{{len(rows)}}}{{*}}{{{TEX_BACKEND_LABEL[backend]}}}")
    for i, (model, w, fr, u) in enumerate(rows):
        prefix = "  & " if i == 0 else "  & "
        tex_lines.append(f"{prefix}{model:<13s} & {w} & {fr} & {u} \\\\")
    if backend != "rag":
        tex_lines.append(r"\midrule")
tex_lines.extend([
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
])
tex_path = Path("tables") / "appendix_L_privacy.tex"
tex_path.parent.mkdir(parents=True, exist_ok=True)
tex_path.write_text("\n".join(tex_lines) + "\n")
print(f"Saved LaTeX table to {tex_path}")

# ── Generate figure ─────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    # Color/marker by backend
    backend_style = {
        "vanilla": {"color": "#2196F3", "marker": "o", "label": "Vanilla"},
        "oracle": {"color": "#4CAF50", "marker": "s", "label": "Oracle"},
        "rag": {"color": "#FF9800", "marker": "^", "label": "RAG"},
    }

    model_labels = {
        "0_6b": "0.6B", "3b": "3B", "7b": "7B", "8b": "8B", "32b": "32B"
    }

    for trial, p in trial_points.items():
        # Determine backend and model
        for bk in ["vanilla", "oracle", "rag"]:
            if trial.startswith(bk + "_"):
                backend = bk
                model_slug = trial[len(bk) + 1:]
                break
        else:
            continue

        style = backend_style[backend]
        model_label = model_labels.get(model_slug, model_slug)

        ax.scatter(p["false_refusal_rate"], p["withholding_accuracy"],
                   color=style["color"], marker=style["marker"], s=100, zorder=5)
        ax.annotate(model_label, (p["false_refusal_rate"], p["withholding_accuracy"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)

    # Legend (one entry per backend)
    for bk, style in backend_style.items():
        ax.scatter([], [], color=style["color"], marker=style["marker"],
                   s=100, label=style["label"])

    # Diagonal (random classifier)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")

    ax.set_xlabel("False Refusal Rate (utility loss)", fontsize=12)
    ax.set_ylabel("Withholding Accuracy (privacy protection)", fontsize=12)
    ax.set_title("Privacy-Utility Trade-off (D4 Permission)", fontsize=13)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig_path = RUN_DIR / "eval_results" / "fig_privacy_utility_auc.pdf"
    fig.savefig(fig_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Figure saved to {fig_path}")

    # Also save PNG for quick viewing
    fig2, ax2 = plt.subplots(1, 1, figsize=(7, 6))
    for trial, p in trial_points.items():
        for bk in ["vanilla", "oracle", "rag"]:
            if trial.startswith(bk + "_"):
                backend = bk
                model_slug = trial[len(bk) + 1:]
                break
        else:
            continue
        style = backend_style[backend]
        model_label = model_labels.get(model_slug, model_slug)
        ax2.scatter(p["false_refusal_rate"], p["withholding_accuracy"],
                    color=style["color"], marker=style["marker"], s=100, zorder=5)
        ax2.annotate(model_label, (p["false_refusal_rate"], p["withholding_accuracy"]),
                     textcoords="offset points", xytext=(6, 4), fontsize=8)
    for bk, style in backend_style.items():
        ax2.scatter([], [], color=style["color"], marker=style["marker"],
                    s=100, label=style["label"])
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax2.set_xlabel("False Refusal Rate (utility loss)", fontsize=12)
    ax2.set_ylabel("Withholding Accuracy (privacy protection)", fontsize=12)
    ax2.set_title("Privacy-Utility Trade-off (D4 Permission)", fontsize=13)
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc="lower right", fontsize=10)
    ax2.grid(True, alpha=0.3)
    png_path = RUN_DIR / "eval_results" / "fig_privacy_utility_auc.png"
    fig2.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"PNG saved to {png_path}")

except ImportError:
    print("matplotlib not available — skipping figure generation")
