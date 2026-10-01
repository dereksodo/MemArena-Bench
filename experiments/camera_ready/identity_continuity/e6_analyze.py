"""E6 analysis: the renaming artefact first, then the identity-continuity effect.

Two contrasts, read in this order:

  raw - sham    the renaming artefact. Not the effect. A two-token real name becomes a
                one-token pseudonym, so pool token counts differ and BM25's IDF shifts.
                This bounds how much of any difference renaming alone can buy.
  break - sham  the effect. Both arms are renamed identically, so the artefact above
                cancels; the only surviving difference is whether an agent keeps one
                label across sessions.

The pool is byte-fixed per item (arm='c0star', one pool seed), so unlike E2/E3 there is
no matching to audit -- the construction check is that sham and break agree on pool token
count exactly, which is asserted below rather than eyeballed.

Ego-clustered paired bootstrap over (mapping_seed, iid) cells; R x N is never treated as
N*R independent observations.

  python3 e6_analyze.py --data identity_results_bm25.json
"""

from __future__ import annotations

import argparse, json, statistics as st, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from e2_analyze import paired_by_item, cluster_bootstrap, contrast   # noqa: E402

METRICS = ('nr_first', 'rr_first', 'gold_recall10', 'all_gold10')
PRIMARY = 'nr_first'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    args = ap.parse_args()
    blob = json.load(open(args.data))
    rows = blob['rows']

    print(f"=== E6, budget B={blob['budget']}, mapping seeds={blob['seeds']}, "
          f"pool seed={blob.get('pool_seed', 0)} (held fixed), rows={len(rows)} ===")

    # ---- construction check: the contrast's premise, not a diagnostic ----
    tok = defaultdict(list)
    for r in rows:
        tok[r['arm']].append(r['pool_tokens'])
    print(f"\n[0] pool token counts by arm (sham and break MUST be identical)")
    for a in ('raw', 'sham', 'break'):
        print(f"    {a:6s} mean {st.mean(tok[a]):>9.1f}   min {min(tok[a])}  max {max(tok[a])}")
    same = tok['sham'] == tok['break']
    print(f"    sham == break, item by item: {'YES' if same else 'NO -- contrast invalid'}")

    n_items = len({r['iid'] for r in rows})
    n_cells = len({(r['mapping_seed'], r['iid']) for r in rows})
    print(f"    {n_items} items x {blob['seeds']} mapping seeds = {n_cells} paired cells")

    # ---- arm means ----
    by = defaultdict(list)
    for r in rows:
        by[r['arm']].append(r)
    print(f"\n[1] arm means")
    print(f"    {'arm':7s}{'NR_first':>10}{'RR_first':>10}{'Recall@10':>11}{'all-gold@10':>13}")
    for a in ('raw', 'sham', 'break'):
        v = by[a]
        print(f"    {a:7s}{st.mean(r['nr_first'] for r in v):>10.4f}"
              f"{st.mean(r['rr_first'] for r in v):>10.4f}"
              f"{st.mean(r['gold_recall10'] for r in v):>11.4f}"
              f"{st.mean(r['all_gold10'] for r in v):>13.4f}")

    # ---- the two contrasts ----
    _, egos = paired_by_item(rows, PRIMARY)
    for a, label in (('raw', 'raw − sham (renaming artefact)'),
                     ('break', 'break − sham (the effect)')):
        print(f"\n[2] {label}")
        for m in METRICS:
            cell, _ = paired_by_item(rows, m)
            c = contrast(cell, egos, a, ['sham'], '')
            tag = ' PRIMARY' if m == PRIMARY else ''
            print(f"    {m:15s}{c['delta']:+.4f}  95% CI "
                  f"[{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}]  n={c['n']}{tag}")

    # ---- per-dimension, primary metric: is it one dimension or all of them? ----
    cell, _ = paired_by_item(rows, PRIMARY)
    dim_of = {(r['mapping_seed'], r['iid']): r['dim'] for r in rows}
    per_dim = defaultdict(list)
    for k, v in cell.items():
        if 'break' in v and 'sham' in v:
            per_dim[dim_of[k]].append(v['break'] - v['sham'])
    print(f"\n[3] break − sham on {PRIMARY}, by dimension")
    for d in sorted(per_dim, key=lambda x: st.mean(per_dim[x])):
        v = per_dim[d]
        print(f"    {d:22s}{st.mean(v):+.4f}   n={len(v)}")
    neg = sum(1 for d in per_dim if st.mean(per_dim[d]) < 0)
    print(f"    {neg} of {len(per_dim)} dimensions negative (gold easier to find)")

    # ---- per-seed spread: is the effect stable across mapping draws? ----
    per_seed = []
    for s in range(blob['seeds']):
        sub = {k: v for k, v in cell.items() if k[0] == s}
        c = contrast(sub, egos, 'break', ['sham'], '')
        if c:
            per_seed.append(c['delta'])
    print(f"\n[4] per-mapping-seed break − sham ({PRIMARY}): "
          f"{[round(x, 4) for x in per_seed]}")
    print(f"    mean {st.mean(per_seed):+.4f}  sd {st.pstdev(per_seed):.4f}")


if __name__ == '__main__':
    main()
