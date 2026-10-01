#!/usr/bin/env python3
"""Classify D6 compliant-refusal predictions by WHY the model refused.

For Vanilla and RAG backends the corpus is only partially present in the
prompt: Vanilla truncates to the last 8K tokens of the ego user's history,
RAG only surfaces the BM25 top-k passages. When the evidence session for a
refuse-expected D6 query is missing from the prompt, the model cannot have
"withheld" the answer — it simply didn't see the content. A refusal in that
case looks like policy compliance but is really a retrieval miss; without
this check the reader would over-credit Vanilla / RAG on D6 privacy.

This script reads every evaluation_results JSON in scope, enumerates rows
where policy_expected == DENY_NO_ACCESS AND policy_compliant is True, and
classifies the refusal text into one of three labels:

* ``info_absent``  — the prediction explicitly says the content is not in
  the conversation history / the model does not remember it / the speaker
  never mentioned it. This is the degenerate retrieval-miss refusal.
* ``privacy``      — the prediction explicitly invokes privacy, confidentiality,
  consent, or policy as the reason for not sharing. This is the refusal
  pattern D6 is trying to elicit.
* ``other``        — any refusal that neither matches the info-absent
  vocabulary nor the privacy vocabulary. Worth eyeballing manually if the
  count is large.

The classifier is intentionally conservative keyword-based: we prioritise
surfacing the info_absent bucket (to surface retrieval-masked wins) over
squeezing borderline examples into ``privacy``. Each bucket prints an
example prediction so the pattern is auditable.

Usage
-----
    python analyze_refusal_provenance.py [--no-examples]

Prints a grouped table across the three stochastic seeds in
:data:`paper.paper_data.SEEDS`; does not write any paper artefact. Intended
as a diagnostic patch, not a regen pipeline.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper_data import policy_compliant_from_armB  # noqa: E402
from paper_data import (  # noqa: E402
    BACKEND_TEX, MODEL_ORDER, MODEL_TEX,
    SEEDS, DEFAULT_RUN,
    load_all_cells,
)
from d6_refusal_classifier import classify  # noqa: E402


# --- Main loop ---------------------------------------------------------------

BACKENDS = ["vanilla", "rag"]


def _scan_cell(json_path: Path):
    """Yield (qid, prediction, bucket) for every compliant-refusal row."""
    try:
        data = json.loads(json_path.read_text())
    except (OSError, json.JSONDecodeError):
        return
    for r in data.get("details", []):
        qid = r.get("question_id") or ""
        if not qid.startswith("d4_"):
            continue
        if r.get("policy_expected") != "DENY_NO_ACCESS":
            continue
        if not policy_compliant_from_armB(r):
            continue
        yield qid, r.get("prediction", ""), classify(r.get("prediction", ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-examples", action="store_true",
                        help="suppress printing a sample prediction per bucket")
    args = parser.parse_args()

    grid = load_all_cells(DEFAULT_RUN)

    # totals[backend][model][bucket] = aggregate count across SEEDS
    totals: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    examples: dict[str, str] = {}

    for backend in BACKENDS:
        for model in MODEL_ORDER:
            for seed in SEEDS:
                cell = grid.get((seed, backend, model))
                if cell is None:
                    continue
                for qid, pred, bucket in _scan_cell(cell.source_path):
                    totals[backend][model][bucket] += 1
                    if bucket not in examples and pred.strip():
                        examples[bucket] = f"{backend}/{MODEL_TEX[model]}/{seed} [{qid}]: {pred.strip()[:240]}"

    # Print table.
    print(f"{'backend':9s} {'model':14s} {'#refusals':>10s}  "
          f"{'info_absent':>12s}  {'privacy':>9s}  {'other':>6s}  "
          f"{'info_absent %':>14s}")
    print("-" * 85)
    for backend in BACKENDS:
        for model in MODEL_ORDER:
            counts = totals[backend][model]
            total = counts["info_absent"] + counts["privacy"] + counts["other"]
            if total == 0:
                print(f"{backend:9s} {MODEL_TEX[model]:14s} {'0':>10s}  "
                      f"{'--':>12s}  {'--':>9s}  {'--':>6s}  {'--':>14s}")
                continue
            pct = 100.0 * counts["info_absent"] / total
            print(f"{backend:9s} {MODEL_TEX[model]:14s} {total:>10d}  "
                  f"{counts['info_absent']:>12d}  {counts['privacy']:>9d}  "
                  f"{counts['other']:>6d}  {pct:>13.1f}%")
        print("-" * 85)

    if not args.no_examples:
        print("\nExamples (one per bucket):")
        for bucket in ("info_absent", "privacy", "other"):
            if bucket in examples:
                print(f"  [{bucket}] {examples[bucket]}")


if __name__ == "__main__":
    main()
