#!/usr/bin/env python3
"""Apply Arm-B rejudge results to existing 5x5x3 evaluation_results JSONs.

For every (backend, reader, seed) cell in experiments_index.csv whose
matching out/d6_armB_2026-05-03/per_item_armB_<seed>/<backend>_<reader>.jsonl
exists, rewrite the eval JSON in place:

  - For each D4 record, replace ``policy_compliant`` + ``policy_reason``
    with ``policy_category`` (REFUSAL/NONE/COMPLY/PARSE_ERROR) and
    ``leaked_fact_in_output``; recompute ``correct``/``score``/``reason``
    via the Arm-B mapping in eval.src.scoring._armB_correctness.
  - Strip every record's legacy ``policy_compliant``/``policy_reason`` fields
    (now retired across all dimensions).
  - Rewrite the ``summary`` block: drop deny/redact/ask/leakage rates and
    policy_by_expected_accuracy; recompute policy_accuracy as the share of
    D4 records whose Arm-B verdict is correct.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval.src.scoring import _armB_correctness  # noqa: E402

INDEX_CSV = REPO_ROOT / "experiments_index.csv"
ARMB_ROOT = REPO_ROOT / "out" / "d6_armB_2026-05-03"

# Backend tag in CSV -> backend tag the rejudge cell key uses.
# rejudge_d6_armB.py keys cells as ``<backend>_<reader>`` directly from the
# CSV row, so this is a no-op map.
LEGACY_SUMMARY_KEYS = {
    "policy_by_expected_accuracy",
    "deny_compliance_rate",
    "redaction_compliance_rate",
    "ask_compliance_rate",
    "privacy_leakage_rate",
    "calibrated_refusal_rate",
}

LEGACY_RECORD_KEYS = {"policy_compliant", "policy_reason"}


def _load_armB_lookup(seed: str, backend: str, reader: str) -> dict[str, dict] | None:
    p = ARMB_ROOT / f"per_item_armB_{seed}" / f"{backend}_{reader}.jsonl"
    if not p.exists():
        return None
    out: dict[str, dict] = {}
    with p.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[r["instance_id"]] = r
    return out


def _rewrite_record(record: dict, armB: dict | None) -> tuple[dict, int]:
    """Return ``(new_record, updated_d4_correct_delta)``.

    ``updated_d4_correct_delta`` is +1 / 0 / -1 depending on whether
    ``correct`` flipped during the rewrite (used for sanity logging).
    """
    qid = record.get("question_id", "")
    is_d4 = qid.startswith("d4_perm")

    # Strip legacy fields from every record.
    new = {k: v for k, v in record.items() if k not in LEGACY_RECORD_KEYS}

    if is_d4 and armB is not None and qid in armB:
        a = armB[qid]
        category = str(a.get("category", "PARSE_ERROR"))
        leaked = bool(a.get("leaked_fact_in_output", False))
        expected_mode = str(record.get("expected_answer_mode") or "")
        ok, reason = _armB_correctness(category, expected_mode)
        before = bool(new.get("correct", False))
        new["correct"] = ok
        new["score"] = 1.0 if ok else 0.0
        new["reason"] = reason
        new["policy_category"] = category
        new["leaked_fact_in_output"] = leaked
        delta = (1 if ok else 0) - (1 if before else 0)
        return new, delta

    return new, 0


def _recompute_summary(details: list[dict], summary: dict) -> dict:
    new_summary = {k: v for k, v in summary.items() if k not in LEGACY_SUMMARY_KEYS}

    # Recount answer_scored_total/correct from details (cheap; keeps drift out).
    scored_total = sum(1 for r in details if r.get("answer_scored"))
    scored_correct = sum(1 for r in details if r.get("answer_scored") and r.get("correct"))
    new_summary["answer_scored_total"] = scored_total
    new_summary["answer_scored_correct"] = scored_correct
    new_summary["total"] = scored_total
    new_summary["correct"] = scored_correct
    new_summary["accuracy"] = (scored_correct / scored_total) if scored_total else 0.0

    # mean_score from details (recompute since 'correct' changed).
    score_sum = sum(float(r.get("score") or 0.0) for r in details if r.get("answer_scored"))
    new_summary["mean_score"] = (score_sum / scored_total) if scored_total else 0.0

    # by_type_accuracy: rebuild from details.
    by_type_total: dict[str, int] = {}
    by_type_correct: dict[str, int] = {}
    for r in details:
        if not r.get("answer_scored"):
            continue
        qt = str(r.get("question_type") or "unknown")
        by_type_total[qt] = by_type_total.get(qt, 0) + 1
        if r.get("correct"):
            by_type_correct[qt] = by_type_correct.get(qt, 0) + 1
    new_summary["by_type_accuracy"] = {
        k: (by_type_correct.get(k, 0) / n if n else 0.0)
        for k, n in sorted(by_type_total.items())
    }

    # policy_accuracy = D4 records with correct Arm-B verdict.
    d4_total = 0
    d4_correct = 0
    for r in details:
        if not r.get("answer_scored"):
            continue
        if str(r.get("question_id") or "").startswith("d4_perm"):
            d4_total += 1
            if r.get("correct"):
                d4_correct += 1
    new_summary["policy_total"] = d4_total
    new_summary["policy_accuracy"] = (d4_correct / d4_total) if d4_total else 0.0

    return new_summary


def _process_cell(json_rel: str, backend: str, reader: str, seed: str) -> tuple[bool, int, int]:
    json_path = REPO_ROOT / json_rel
    if not json_path.exists():
        return False, 0, 0
    armB = _load_armB_lookup(seed, backend, reader)
    if armB is None:
        return False, 0, 0

    payload = json.loads(json_path.read_text())
    details = payload.get("details", [])

    new_details: list[dict] = []
    flips = 0
    rewritten = 0
    for r in details:
        new_r, delta = _rewrite_record(r, armB)
        new_details.append(new_r)
        if delta:
            flips += 1
        if str(new_r.get("question_id") or "").startswith("d4_perm"):
            rewritten += 1

    payload["details"] = new_details
    payload["summary"] = _recompute_summary(new_details, payload.get("summary", {}))
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return True, rewritten, flips


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="do not write files")
    ap.add_argument("--seed", action="append", default=None,
                    help="restrict to seed(s); repeatable")
    args = ap.parse_args()

    rows: list[tuple[str, str, str, str]] = []
    with INDEX_CSV.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("table") != "main_5x5x3":
                continue
            rel = (row.get("json_path") or "").strip()
            if not rel or rel == "none":
                continue
            seed = row["trial"]
            if args.seed and seed not in set(args.seed):
                continue
            rows.append((rel, row["backend"], row["model_tag"], seed))

    print(f"[backfill] {len(rows)} cells to process", flush=True)
    n_done = 0
    n_skipped = 0
    total_d4 = 0
    total_flips = 0
    for rel, backend, reader, seed in rows:
        if args.dry_run:
            armB = _load_armB_lookup(seed, backend, reader)
            ok = armB is not None
            print(f"[dry] {seed} {backend}_{reader}: {'has' if ok else 'missing'} armB jsonl ({rel})")
            continue
        ok, n_d4, flips = _process_cell(rel, backend, reader, seed)
        if ok:
            n_done += 1
            total_d4 += n_d4
            total_flips += flips
            print(f"[ok] {seed} {backend}_{reader}: {n_d4} D4 records, {flips} 'correct' flips", flush=True)
        else:
            n_skipped += 1
            print(f"[skip] {seed} {backend}_{reader}: missing armB or json", flush=True)

    if not args.dry_run:
        print(f"[done] rewrote {n_done} JSONs, skipped {n_skipped}; "
              f"{total_d4} D4 records touched, {total_flips} correctness flips")


if __name__ == "__main__":
    main()
