"""Summarise the 3-bucket ablation labels (NO_ACCESS / DONT_KNOW / OTHER) on
the *non-leak* DENY records produced by ``judge_d6_ablation_3label.py``.

For each (backend, reader) cell pooled over seeds, report:
  N_deny       — total DENY records
  Leak%        — fact-leak rate (deterministic; same as F1_PU's P-side)
  Among non-leak DENY:
    NoAccess%  — explicit access-control / privacy refusal
    DontKnow%  — epistemic absence
    Other%     — other / off-topic / parse error
  Refusal_share = NoAccess / (NoAccess + DontKnow + Other)
                — share of the *non-leaks* that are policy-grounded refusals
                  rather than amnesia. High = real gating; low = amnesia.

Outputs:
  console table (25 rows + global)
  out/d6_ablation/buckets_per_cell.csv
  out/d6_ablation/buckets_global.csv

Run:
    python3 scripts/analyze_d6_ablation_buckets.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

ABLATION_LABELS = ["NO_ACCESS", "DONT_KNOW", "OTHER"]
BACKENDS_ORDER = ["vanilla", "rag", "oracle", "memobase", "memsearch"]
READERS_ORDER = ["0_6b", "llama3b", "7b", "8b", "32b"]


def aggregate(grid):
    """Return {(backend, reader): stats} pooled across seeds."""
    out = defaultdict(lambda: {
        "deny_total": 0, "deny_leak": 0,
        "deny_nonleak_total": 0,
        "NO_ACCESS": 0, "DONT_KNOW": 0, "OTHER": 0, "UNLABELED": 0,
    })
    for (seed, backend, reader), cell in grid.items():
        if backend not in BACKENDS_ORDER or reader not in READERS_ORDER:
            continue
        try:
            data = json.loads(Path(cell.source_path).read_text())
        except Exception:
            continue
        s = out[(backend, reader)]
        for d in data.get("details", []):
            qid = str(d.get("question_id", ""))
            if not qid.startswith("d4_perm"):
                continue
            mode = str(d.get("expected_answer_mode") or "").lower()
            if mode not in ("deny", "abstain"):
                continue
            s["deny_total"] += 1
            leaked = bool(d.get("leaked_fact_in_output", False))
            if leaked:
                s["deny_leak"] += 1
                continue
            s["deny_nonleak_total"] += 1
            cat = str(d.get("access_category_v3") or "").upper()
            if cat in ABLATION_LABELS:
                s[cat] += 1
            else:
                s["UNLABELED"] += 1
    return out


def _frac(num, den):
    return num / den if den > 0 else 0.0


def print_per_cell(stats):
    print()
    print("=" * 100)
    print("Per-cell D6 ablation buckets (DENY pool, pooled over seeds)")
    print("=" * 100)
    print(f"  {'cell':<22s}  {'N':>4s}  {'Leak%':>6s}  "
          f"{'NoAcc%':>7s}  {'DK%':>6s}  {'Oth%':>6s}  {'Unlbl':>6s}  "
          f"{'Refusal_share':>14s}")
    for b in BACKENDS_ORDER:
        for r in READERS_ORDER:
            s = stats.get((b, r))
            if s is None or s["deny_total"] == 0:
                continue
            n = s["deny_total"]
            nonleak = s["deny_nonleak_total"]
            leak_pct = s["deny_leak"] / n * 100
            na_pct = s["NO_ACCESS"] / n * 100
            dk_pct = s["DONT_KNOW"] / n * 100
            ot_pct = s["OTHER"] / n * 100
            ul_pct = s["UNLABELED"] / n * 100
            refusal_share = (
                s["NO_ACCESS"] / nonleak * 100 if nonleak > 0 else 0.0
            )
            cell_lbl = f"{b}/{r}"
            print(f"  {cell_lbl:<22s}  {n:>4d}  {leak_pct:>5.1f}   "
                  f"{na_pct:>6.1f}  {dk_pct:>5.1f}  {ot_pct:>5.1f}  {ul_pct:>5.1f}   "
                  f"{refusal_share:>13.1f}")


def print_global(stats):
    g = {k: 0 for k in (
        "deny_total", "deny_leak", "deny_nonleak_total",
        "NO_ACCESS", "DONT_KNOW", "OTHER", "UNLABELED",
    )}
    for s in stats.values():
        for k in g:
            g[k] += s.get(k, 0)
    print()
    print("=" * 60)
    print("Global D6 ablation totals (DENY pool, all 25 cells x 3 seeds)")
    print("=" * 60)
    n = g["deny_total"]
    if n == 0:
        print("(no records)")
        return
    nonleak = g["deny_nonleak_total"]
    print(f"  N_deny           = {n}")
    print(f"  Leaked           = {g['deny_leak']}    ({g['deny_leak']/n*100:.1f}%)")
    print(f"  Non-leak total   = {nonleak}    ({nonleak/n*100:.1f}%)")
    if nonleak > 0:
        print(f"  Within non-leak (% of N_deny / % of non-leak):")
        for k in ABLATION_LABELS:
            v = g[k]
            print(f"    {k:<10s}: {v:5d}    "
                  f"({v/n*100:5.1f}% / {v/nonleak*100:5.1f}%)")
        if g["UNLABELED"] > 0:
            print(f"    {'UNLABELED':<10s}: {g['UNLABELED']:5d}    "
                  f"({g['UNLABELED']/n*100:5.1f}% / {g['UNLABELED']/nonleak*100:5.1f}%)")
        print(f"  Refusal_share    = NO_ACCESS / non-leak = "
              f"{g['NO_ACCESS']/nonleak*100:.1f}%")


def write_csvs(stats, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [["backend", "reader", "n_deny", "n_leak", "n_nonleak",
             "n_no_access", "n_dont_know", "n_other", "n_unlabeled",
             "leak_pct", "no_access_pct_of_n", "dont_know_pct_of_n",
             "other_pct_of_n", "refusal_share_of_nonleak_pct"]]
    for b in BACKENDS_ORDER:
        for r in READERS_ORDER:
            s = stats.get((b, r))
            if s is None or s["deny_total"] == 0:
                continue
            n = s["deny_total"]
            nonleak = s["deny_nonleak_total"]
            rows.append([
                b, r, n, s["deny_leak"], nonleak,
                s["NO_ACCESS"], s["DONT_KNOW"], s["OTHER"], s["UNLABELED"],
                f"{s['deny_leak']/n*100:.4f}",
                f"{s['NO_ACCESS']/n*100:.4f}",
                f"{s['DONT_KNOW']/n*100:.4f}",
                f"{s['OTHER']/n*100:.4f}",
                f"{(s['NO_ACCESS']/nonleak*100) if nonleak else 0.0:.4f}",
            ])
    (out_dir / "buckets_per_cell.csv").write_text("")
    with (out_dir / "buckets_per_cell.csv").open("w", newline="") as f:
        csv.writer(f).writerows(rows)

    g = {k: 0 for k in ("deny_total", "deny_leak", "deny_nonleak_total",
                        "NO_ACCESS", "DONT_KNOW", "OTHER", "UNLABELED")}
    for s in stats.values():
        for k in g:
            g[k] += s.get(k, 0)
    g_rows = [["metric", "value"],
              ["deny_total", g["deny_total"]],
              ["deny_leak", g["deny_leak"]],
              ["deny_nonleak_total", g["deny_nonleak_total"]],
              ["NO_ACCESS", g["NO_ACCESS"]],
              ["DONT_KNOW", g["DONT_KNOW"]],
              ["OTHER", g["OTHER"]],
              ["UNLABELED", g["UNLABELED"]]]
    with (out_dir / "buckets_global.csv").open("w", newline="") as f:
        csv.writer(f).writerows(g_rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path,
                        default=REPO_ROOT / "out" / "d6_ablation")
    args = parser.parse_args()

    from memarena.figures.paper_data import load_all_cells
    grid = load_all_cells(include_ablation=False)
    print(f"Loaded {len(grid)} cells")

    stats = aggregate(grid)
    print_per_cell(stats)
    print_global(stats)
    write_csvs(stats, args.out_dir)
    print()
    print(f"Wrote {args.out_dir}/buckets_per_cell.csv")
    print(f"Wrote {args.out_dir}/buckets_global.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
