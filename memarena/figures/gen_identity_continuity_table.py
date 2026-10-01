"""Generate the body of the identity-continuity ablation table.

Reads ``$MEMARENA_RESULTS_DIR/out/identity_continuity/identity_results_bm25.json``
(produced by ``experiments/camera_ready/identity_continuity/e6_identity.py``)
and reports, for the renaming artefact (raw - sham) and the effect
(break - sham), the ego-clustered paired-bootstrap delta and 95% CI of the
normalised first-gold rank and GoldRecall@10.

    python -m memarena.figures.gen_identity_continuity_table
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

from memarena.figures.paper_data import PROJECT_ROOT, RESULTS_ROOT
from memarena.figures.paths import table_path

sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "camera_ready" / "identity_continuity"))
from e2_analyze import contrast, paired_by_item  # noqa: E402

RESULTS = RESULTS_ROOT / "out" / "identity_continuity" / "identity_results_bm25.json"
PRIMARY = "nr_first"
ROWS = (("raw", r"\textit{raw} $-$ \textit{sham} (renaming artefact)"),
        ("break", r"\textbf{\textit{break} $-$ \textit{sham} (the effect)}"))


def contrasts(rows: list) -> dict:
    _, egos = paired_by_item(rows, PRIMARY)
    out = {}
    for arm, _ in ROWS:
        out[arm] = {m: contrast(paired_by_item(rows, m)[0], egos, arm, ["sham"], "") for m in (PRIMARY, "gold_recall10")}
    return out


def render(res: dict) -> str:
    lines = []
    for arm, label in ROWS:
        cells = [f"${c['delta']:+.4f}$ $[{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}]$" for c in (res[arm][PRIMARY], res[arm]["gold_recall10"])]
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines) + "\n"


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate the identity-continuity table body.")
    ap.add_argument("--data", type=Path, default=RESULTS)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    blob = json.loads(args.data.read_text())
    res = contrasts(blob["rows"])
    out = args.out or table_path("identity_continuity_body.tex")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(res), encoding="utf-8")
    n_items = len({r["iid"] for r in blob["rows"]})
    print(f"[gen_identity_continuity_table] items={n_items} seeds={blob['seeds']} "
          + "; ".join(f"{a}: NR {res[a][PRIMARY]['delta']:+.4f} R@10 {res[a]['gold_recall10']['delta']:+.4f}" for a, _ in ROWS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
