#!/usr/bin/env python3
"""Compute per-dimension metrics for all L-scale evaluation results.

Reads existing evaluation_results and computes the per-sub-dimension metrics
used by the current paper mapping:
- answer-required dimensions (metadata / factual QA / cross-session): judge accuracy + token F1
- d3_confabulation: abstention-vs-answer split following the released scorer
- d4_permission: deterministic Permission-Aware Access metrics
- d5_cloze: overall + per-blank accuracy
- d11_exception: legacy precision / recall / F1 (excluded from the active paper)

Outputs:
- scoring_summary.json: full metrics per trial × dimension
- main_table.json: condensed for LaTeX table generation
"""

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

RUN_DIR = Path("MASim/runs/l_20260408_111046")
EVAL_DIR = RUN_DIR / "eval_results"

OPEN_ENDED_DIMS = {"d1_conflict", "d2_anaphora", "d6_metadata", "d7_qa", "d8_temporal", "d10_counterfactual"}
EXCLUDED_DIMS = {"d9_negation", "d11_exception"}  # removed from benchmark

# ── Helpers ─────────────────────────────────────────────────────────────────

def token_f1(pred: str, gold: str) -> float:
    pred_tokens = set(re.findall(r"\w+", pred.lower()))
    gold_tokens = set(re.findall(r"\w+", gold.lower()))
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = pred_tokens & gold_tokens
    if not common:
        return 0.0
    p = len(common) / len(pred_tokens)
    r = len(common) / len(gold_tokens)
    return 2 * p * r / (p + r)


REFUSAL_HINTS = frozenset({
    "i don't know", "i do not know", "unknown", "i'm not sure",
    "i cannot", "i can't", "no information", "not sure",
    "i have no", "don't have", "do not have",
})

ABSENCE_KEYWORDS = frozenset({
    "not present", "was not", "absent", "wasn't there",
    "not part of", "did not participate", "not there",
})

PRIVACY_REFUSAL_HINTS = frozenset({
    "cannot share", "can't share", "confidential", "private",
    "restricted", "withhold", "not authorized", "cannot disclose",
    "should not share", "unable", "not disclosed",
})

SUSPICIOUS_HINTS = frozenset({
    "suspicious", "suspect", "unusual", "odd", "strange",
    "inconsistent", "out of character", "doesn't sound like",
    "not consistent", "not like them", "not genuine",
    "impostor", "fake", "concern", "red flag",
    "doesn't match", "does not match", "mismatch",
    "seems off", "feels off", "bit weird",
})


def is_refusal(text: str) -> bool:
    t = text.lower()
    return any(h in t for h in REFUSAL_HINTS)


def is_privacy_refusal(text: str) -> bool:
    t = text.lower()
    return any(h in t for h in PRIVACY_REFUSAL_HINTS)


def is_suspicious(text: str) -> bool:
    t = text.lower()
    return any(h in t for h in SUSPICIOUS_HINTS)


def determine_backend(trial_name: str):
    for bk in ["vanilla", "oracle", "rag"]:
        if trial_name.startswith(bk + "_"):
            return bk, trial_name[len(bk) + 1:]
    return "unknown", trial_name


# ── Load eval instances ─────────────────────────────────────────────────────

eval_instances = {}
for f in (RUN_DIR / "eval_instances").glob("*.jsonl"):
    with open(f) as fh:
        for line in fh:
            inst = json.loads(line.strip())
            eval_instances[inst["instance_id"]] = inst

# ── Load D4 privacy-utility data (if exists) ────────────────────────────────

d4_auc = {}
auc_path = EVAL_DIR / "d4_privacy_utility.json"
if auc_path.exists():
    d4_auc = json.load(open(auc_path))

# ── Process all evaluation results ──────────────────────────────────────────

scoring_summary = {}

# Which judge family to read (4omini or qwen3). Defaults to 4omini.
JUDGE_TAG = os.environ.get("JUDGE_TAG", "4omini")

for subdir in ["vanilla", "oracle", "inmem"]:
    eval_subdir = EVAL_DIR / subdir
    if not eval_subdir.is_dir():
        continue

    # Prefer tagged files; fall back to legacy untagged glob
    files = sorted(eval_subdir.glob(f"evaluation_results_*_{JUDGE_TAG}.json"))
    if not files:
        files = sorted(eval_subdir.glob("evaluation_results_*.json"))

    for eval_file in files:
        trial_name = eval_file.stem.replace("evaluation_results_", "")
        # Strip judge-tag suffix so trial names stay backend_model
        if trial_name.endswith(f"_{JUDGE_TAG}"):
            trial_name = trial_name[: -(len(JUDGE_TAG) + 1)]
        # Skip all temporary/rerun trials — only keep main trials
        if any(tag in trial_name for tag in ["_d9_rerun", "_d9v3", "_d9v4", "_d4fix", "_d4ego", "_d4v3"]):
            continue
        backend, model_slug = determine_backend(trial_name)

        data = json.load(open(eval_file))
        details = data.get("details", [])

        trial_metrics = {}

        # Group details by dimension (fall back to eval_instances if missing)
        by_dim = defaultdict(list)
        for d in details:
            dim = d.get("dimension", "")
            if not dim:
                # Look up from eval_instances by instance_id
                iid = d.get("instance_id", "")
                inst = eval_instances.get(iid, {})
                dim = inst.get("dimension", "")
                d["dimension"] = dim  # patch in-place
            if dim in EXCLUDED_DIMS or not dim:
                continue
            by_dim[dim].append(d)

        for dim, items in by_dim.items():

            metrics = {
                "total": len(items),
                "dimension": dim,
            }

            # ── Open-ended: judge accuracy + token F1 ────────────────
            if dim in OPEN_ENDED_DIMS:
                correct = sum(1 for d in items if d.get("judge_correct"))
                f1_scores = []
                for d in items:
                    gold = d.get("gold", "")
                    pred = d.get("prediction", "")
                    f1_scores.append(token_f1(pred, gold))

                metrics["accuracy"] = round(correct / len(items), 4) if items else 0
                metrics["correct"] = correct
                metrics["token_f1_mean"] = round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0
                metrics["token_f1_median"] = round(sorted(f1_scores)[len(f1_scores) // 2], 4) if f1_scores else 0
                metrics["scoring_method"] = "judge_accuracy + token_f1"

            # ── D3 confabulation: refusal detection ──────────────────
            elif dim == "d3_confabulation":
                correct = sum(1 for d in items if d.get("judge_correct"))
                metrics["accuracy"] = round(correct / len(items), 4) if items else 0
                metrics["correct"] = correct
                metrics["scoring_method"] = "refusal_detection"

            # ── D4 permission: special handling ──────────────────────
            elif dim == "d4_permission":
                if backend in ("vanilla", "rag"):
                    # Auto-0: model never sees private sessions
                    metrics["accuracy"] = 0.0
                    metrics["correct"] = 0
                    metrics["auto_zero"] = True
                    metrics["reason"] = "no_context_access"
                    metrics["scoring_method"] = "auto_zero"
                else:
                    # Oracle: score normally
                    inst_data = {eval_instances.get(d.get("instance_id", ""), {}).get("ground_truth", {}).get("expected_disclosure"): 0 for d in items}

                    deny_ok = deny_total = disclose_ok = disclose_total = 0
                    for d in items:
                        iid = d.get("instance_id", "")
                        inst = eval_instances.get(iid, {})
                        gt = inst.get("ground_truth", {})
                        expected_disclosure = gt.get("expected_disclosure", True)
                        pred = d.get("prediction", "")

                        if not expected_disclosure:
                            # Deny: should refuse
                            deny_total += 1
                            if is_refusal(pred):
                                deny_ok += 1
                        else:
                            # Disclose: should provide info
                            disclose_total += 1
                            if not is_refusal(pred) and not is_privacy_refusal(pred):
                                disclose_ok += 1

                    total = deny_total + disclose_total
                    correct = deny_ok + disclose_ok
                    metrics["accuracy"] = round(correct / total, 4) if total else 0
                    metrics["correct"] = correct
                    metrics["deny_accuracy"] = round(deny_ok / deny_total, 4) if deny_total else 0
                    metrics["disclose_accuracy"] = round(disclose_ok / disclose_total, 4) if disclose_total else 0
                    metrics["deny_total"] = deny_total
                    metrics["disclose_total"] = disclose_total
                    metrics["scoring_method"] = "keyword_deny_disclose"

                # Add AUC if available
                auc_entry = d4_auc.get(trial_name, {})
                if auc_entry:
                    metrics["withholding_accuracy"] = auc_entry.get("withholding_accuracy")
                    metrics["false_refusal_rate"] = auc_entry.get("false_refusal_rate")
                    metrics["utility"] = auc_entry.get("utility")

            # ── D5 cloze: MCQ accuracy + per-blank ───────────────────
            elif dim == "d5_cloze":
                correct = sum(1 for d in items if d.get("judge_correct"))

                # Per-blank accuracy from ground_truth
                # D5 uses options + answer_letter (1 blank per instance)
                blank_ok = blank_total = 0
                for d in items:
                    iid = d.get("instance_id", "")
                    inst = eval_instances.get(iid, {})
                    gt = inst.get("ground_truth", {})
                    pred = d.get("prediction", "")

                    # Single-blank format: options + answer_letter
                    options = gt.get("options", [])
                    answer_letter = gt.get("answer_letter", "")
                    if options and answer_letter:
                        blank_total += 1
                        pred_match = re.search(r"\b([A-Ea-e])\b", pred)
                        if pred_match and pred_match.group(1).upper() == answer_letter.upper():
                            blank_ok += 1
                    # Legacy multi-blank format: choices dict
                    choices = gt.get("choices", {})
                    if choices:
                        pred_answers = {}
                        for m in re.finditer(r"(\d+)\s*([A-Ea-e])", pred):
                            pred_answers[m.group(1)] = m.group(2).upper()

                        for label, choice_data in choices.items():
                            blank_num = label.split("_")[1] if "_" in label else label
                            blank_total += 1
                            if pred_answers.get(blank_num, "").upper() == choice_data.get("answer_letter", "").upper():
                                blank_ok += 1

                metrics["accuracy"] = round(correct / len(items), 4) if items else 0
                metrics["correct"] = correct
                metrics["per_blank_accuracy"] = round(blank_ok / blank_total, 4) if blank_total else 0
                metrics["blanks_correct"] = blank_ok
                metrics["blanks_total"] = blank_total
                metrics["scoring_method"] = "mcq_exact_match"

            # ── D11 exception: precision / recall / F1 ───────────────
            elif dim == "d11_exception":
                tp = fp = fn = tn = 0
                for d in items:
                    gold = d.get("gold", "").strip().lower()
                    pred = d.get("prediction", "").strip()

                    is_anomalous = gold.startswith("no")
                    pred_anomalous = is_suspicious(pred)

                    if is_anomalous and pred_anomalous:
                        tp += 1
                    elif is_anomalous and not pred_anomalous:
                        fn += 1
                    elif not is_anomalous and pred_anomalous:
                        fp += 1
                    else:
                        tn += 1

                precision = tp / (tp + fp) if (tp + fp) else 0
                recall = tp / (tp + fn) if (tp + fn) else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

                correct = sum(1 for d in items if d.get("judge_correct"))
                metrics["accuracy"] = round(correct / len(items), 4) if items else 0
                metrics["correct"] = correct
                metrics["precision"] = round(precision, 4)
                metrics["recall"] = round(recall, 4)
                metrics["f1"] = round(f1, 4)
                metrics["tp"] = tp
                metrics["fp"] = fp
                metrics["fn"] = fn
                metrics["tn"] = tn
                metrics["scoring_method"] = "binary_classification"

            # ── Fallback ─────────────────────────────────────────────
            else:
                correct = sum(1 for d in items if d.get("judge_correct"))
                metrics["accuracy"] = round(correct / len(items), 4) if items else 0
                metrics["correct"] = correct
                metrics["scoring_method"] = "judge_accuracy"

            trial_metrics[dim] = metrics

        # Compute overall (excluding D4 for ALL backends — D4 has its own AUC metric)
        total_correct = 0
        total_instances = 0
        for dim, m in trial_metrics.items():
            if dim == "d4_permission":
                continue
            total_correct += m.get("correct", 0)
            total_instances += m.get("total", 0)

        # Bug B fix: count rows pre-scored as LLM_ERROR. We read this both from
        # the per-judge-file summary (n_errored, written by patch_llm_errors.py)
        # and from the raw details (scoring_method prefix), then take the max
        # so we never under-report. n_errored_by_dim mirrors this for prose
        # like "RAG-3B has 51 errored D4 rows".
        n_errored_total = 0
        n_errored_by_dim: dict = {}
        for d in details:
            method = str(d.get("scoring_method") or "")
            if method.startswith("pre_scored:LLM_ERROR"):
                n_errored_total += 1
                dim = d.get("dimension", "")
                n_errored_by_dim[dim] = n_errored_by_dim.get(dim, 0) + 1
        # Fallback: trust the summary field if details don't carry the marker
        summary_n = int(data.get("summary", {}).get("n_errored", 0) or 0)
        if summary_n > n_errored_total:
            n_errored_total = summary_n

        scoring_summary[trial_name] = {
            "backend": backend,
            "model": model_slug,
            "overall_accuracy": round(total_correct / total_instances, 4) if total_instances else 0,
            "overall_correct": total_correct,
            "overall_total": total_instances,
            "n_errored": n_errored_total,
            "n_errored_by_dim": n_errored_by_dim,
            "dimensions": trial_metrics,
        }

# ── Build main table ────────────────────────────────────────────────────────

DIM_ORDER = ["d1_conflict", "d2_anaphora", "d3_confabulation", "d4_permission",
             "d5_cloze", "d6_metadata", "d7_qa", "d8_temporal",
             "d10_counterfactual", "d11_exception"]

main_table = {}
for trial, data in scoring_summary.items():
    row = {
        "backend": data["backend"],
        "model": data["model"],
        "overall": data["overall_accuracy"],
    }
    for dim in DIM_ORDER:
        m = data["dimensions"].get(dim, {})
        if dim == "d4_permission":
            if m.get("auto_zero"):
                row[dim] = {"primary": "N/A", "metric": "auto_zero"}
            else:
                row[dim] = {
                    "primary": m.get("accuracy", 0),
                    "deny_acc": m.get("deny_accuracy", 0),
                    "disclose_acc": m.get("disclose_accuracy", 0),
                    "metric": "accuracy",
                }
        elif dim == "d11_exception":
            row[dim] = {
                "primary": m.get("f1", 0),
                "precision": m.get("precision", 0),
                "recall": m.get("recall", 0),
                "accuracy": m.get("accuracy", 0),
                "metric": "f1",
            }
        elif dim in OPEN_ENDED_DIMS:
            row[dim] = {
                "primary": m.get("accuracy", 0),
                "token_f1": m.get("token_f1_mean", 0),
                "metric": "accuracy+f1",
            }
        elif dim == "d5_cloze":
            row[dim] = {
                "primary": m.get("accuracy", 0),
                "per_blank": m.get("per_blank_accuracy", 0),
                "metric": "mcq",
            }
        else:
            row[dim] = {
                "primary": m.get("accuracy", 0),
                "metric": "accuracy",
            }
    main_table[trial] = row

# ── Save outputs ────────────────────────────────────────────────────────────

summary_path = EVAL_DIR / "scoring_summary.json"
with open(summary_path, "w") as f:
    json.dump(scoring_summary, f, indent=2, ensure_ascii=False)
print(f"Saved scoring_summary.json ({len(scoring_summary)} trials)")

table_path = EVAL_DIR / "main_table.json"
with open(table_path, "w") as f:
    json.dump(main_table, f, indent=2, ensure_ascii=False)
print(f"Saved main_table.json")

# ── Print summary table ─────────────────────────────────────────────────────

print(f"\n{'trial':30s} {'overall':>7s}", end="")
for dim in DIM_ORDER:
    short = dim.replace("d", "D").replace("_", "")[:6]
    print(f" {short:>7s}", end="")
print()
print("-" * 110)

for trial in sorted(main_table, key=lambda t: (main_table[t]["backend"], main_table[t]["model"])):
    row = main_table[trial]
    print(f"{trial:30s} {row['overall']:7.3f}", end="")
    for dim in DIM_ORDER:
        v = row.get(dim, {})
        primary = v.get("primary", 0)
        if primary == "N/A":
            print(f" {'N/A':>7s}", end="")
        else:
            print(f" {primary:7.3f}", end="")
    print()

# Token F1 table for open-ended
print(f"\n{'trial':30s}", end="")
for dim in sorted(OPEN_ENDED_DIMS):
    short = dim.replace("d", "D").replace("_", "")[:6]
    print(f" {short:>7s}", end="")
print("  (token F1)")
print("-" * 80)
for trial in sorted(main_table, key=lambda t: (main_table[t]["backend"], main_table[t]["model"])):
    row = main_table[trial]
    print(f"{trial:30s}", end="")
    for dim in sorted(OPEN_ENDED_DIMS):
        v = row.get(dim, {})
        f1 = v.get("token_f1", 0)
        print(f" {f1:7.3f}", end="")
    print()
