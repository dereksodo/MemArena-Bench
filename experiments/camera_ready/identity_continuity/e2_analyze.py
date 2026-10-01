"""E2 analysis: placebo gate on normalized rank, then the foreign effect.

Reports in the order the spec fixes: C0* absolute difficulty first (is the budget
informative at all?), then Delta_placebo against the equivalence margin, and only if that
passes, Delta_foreign. Ego-clustered paired bootstrap; R x N is never treated as N*R
independent observations.
"""

from __future__ import annotations

import argparse, json, random, statistics as st
from collections import defaultdict

EPS_NR = 0.015          # pre-registered, calibrated against the LoCoMo contrast's 0.108
PRIMARY = 'nr_first'


def paired_by_item(rows, metric):
    """{(seed, iid): {arm: value}} -> per-seed paired arrays, keyed by ego for clustering."""
    cell = defaultdict(dict)
    ego = {}
    for r in rows:
        cell[(r['seed'], r['iid'])][r['arm']] = r[metric]
        ego[(r['seed'], r['iid'])] = r['ego']
    return cell, ego


def cluster_bootstrap(pairs, egos, n=2000, seed=0):
    """pairs: list of (ego, delta). Resample egos with replacement."""
    by_ego = defaultdict(list)
    for e, d in pairs:
        by_ego[e].append(d)
    keys = list(by_ego)
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        picked = [by_ego[keys[rng.randrange(len(keys))]] for _ in keys]
        flat = [x for grp in picked for x in grp]
        if flat:
            means.append(sum(flat) / len(flat))
    means.sort()
    lo = means[int(0.025 * (len(means) - 1))]
    hi = means[int(0.975 * (len(means) - 1))]
    return lo, hi


def contrast(cell, egos, a, b_arms, label):
    """delta = a - mean(b_arms) per item."""
    pairs = []
    for k, v in cell.items():
        if a in v and all(b in v for b in b_arms):
            pairs.append((egos[k], v[a] - st.mean(v[b] for b in b_arms)))
    if not pairs:
        return None
    d = st.mean(p[1] for p in pairs)
    lo, hi = cluster_bootstrap(pairs, egos)
    return {'label': label, 'delta': d, 'ci': (lo, hi), 'n': len(pairs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    args = ap.parse_args()
    blob = json.load(open(args.data))
    rows, B = blob['rows'], blob['budget']

    print(f"=== E2, budget B={B}, seeds={blob['seeds']}, rows={len(rows)} ===")
    if blob['diags']:
        print(f"!! {len(blob['diags'])} infeasible pool builds -- inspect before reading on")

    # ---- step 1: is the budget informative? C0* absolute difficulty only ----
    c0 = [r for r in rows if r['arm'] == 'c0star']
    print(f"\n[1] C0* absolute difficulty (budget sanity, arms not compared yet)")
    print(f"    NR_first  {st.mean(r['nr_first'] for r in c0):.4f}"
          f"   RR_first {st.mean(r['rr_first'] for r in c0):.4f}"
          f"   Recall@10 {st.mean(r['gold_recall10'] for r in c0):.4f}"
          f"   gold-in-tie {st.mean(r['gold_tied'] for r in c0):.3f}")
    bydim = defaultdict(list)
    for r in c0:
        bydim[r['dim']].append(r)
    print(f"    {'dimension':22s}{'NR_first':>10}{'Recall@10':>11}{'outrank>':>10}")
    for d in sorted(bydim):
        v = bydim[d]
        print(f"    {d:22s}{st.mean(x['nr_first'] for x in v):>10.4f}"
              f"{st.mean(x['gold_recall10'] for x in v):>11.4f}"
              f"{sum(1 for x in v if x['outrank']=='>')/len(v):>10.4f}")

    # ---- step 2: placebo gate, primary metric only ----
    cell, egos = paired_by_item(rows, PRIMARY)
    plac = contrast(cell, egos, 'c0prime', ['c0star'], "Δ_placebo = C0' − C0*")
    print(f"\n[2] Placebo gate on {PRIMARY} (ε = ±{EPS_NR})")
    print(f"    {plac['label']}: {plac['delta']:+.4f}  95% CI [{plac['ci'][0]:+.4f}, {plac['ci'][1]:+.4f}]  n={plac['n']}")
    passed = plac['ci'][0] >= -EPS_NR and plac['ci'][1] <= EPS_NR
    print(f"    gate: {'PASS' if passed else 'FAIL'} — CI {'inside' if passed else 'NOT inside'} [-{EPS_NR}, {EPS_NR}]")

    # ---- step 3: foreign effect ----
    for metric in ('nr_first', 'rr_first', 'gold_recall10'):
        cell_m, _ = paired_by_item(rows, metric)
        f = contrast(cell_m, egos, 'c3', ['c0star', 'c0prime'], f"Δ_foreign ({metric})")
        tag = 'PRIMARY' if metric == PRIMARY else 'secondary'
        print(f"\n[3] {f['label']}  [{tag}]")
        print(f"    {f['delta']:+.4f}  95% CI [{f['ci'][0]:+.4f}, {f['ci'][1]:+.4f}]  n={f['n']}")

    # per-seed spread on the primary metric
    perseed = []
    for s in range(blob['seeds']):
        sub = {k: v for k, v in cell.items() if k[0] == s}
        f = contrast(sub, egos, 'c3', ['c0star', 'c0prime'], '')
        if f:
            perseed.append(f['delta'])
    if perseed:
        print(f"\n    per-seed Δ_foreign ({PRIMARY}): "
              f"{[round(x, 4) for x in perseed]}")
        print(f"    across seeds: mean {st.mean(perseed):+.4f}"
              f"  sd {st.pstdev(perseed):.4f}")

    # matching diagnostics
    c3 = [r for r in rows if r['arm'] == 'c3']
    print(f"\n[4] construction diagnostics")
    print(f"    C3 sessions from the target's own map: {sum(r['c3_overlap'] or 0 for r in c3)}")
    print(f"    caliper failures/item: C0' p95 "
          f"{sorted(r['caliper_fail'] for r in rows if r['arm']=='c0prime')[int(.95*len([1 for r in rows if r['arm']=='c0prime']))]}"
          f"   C3 p95 {sorted(r['caliper_fail'] for r in c3)[int(.95*len(c3))]}")
    print(f"    C3 pool-token rel. diff: p50 {st.median(r['token_rel_diff'] for r in c3):.5f}"
          f"  max {max(r['token_rel_diff'] for r in c3):.5f}")


if __name__ == '__main__':
    main()
