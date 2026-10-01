"""E2 — matched-reachability social-coherence ablation, offline BM25 ranking.

Retriever is B.15's: rank_bm25.BM25Okapi at Okapi defaults, regex [A-Za-z0-9_]+ then
lower-case, one document per session serialized as build_prompt renders it, per-pool IDF.
Ties get fractional (average) ranks; input order never breaks them.

Usage:  python3 e2_structural.py --budget 100 --seeds 10 --out out_b100.json
"""

from __future__ import annotations

import argparse, glob, gzip, json, os, re, statistics as st, sys, time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pools import build_pool                                    # noqa: E402
from rank_bm25 import BM25Okapi                                 # noqa: E402

BASE = Path(os.environ.get('MEMARENA_CORPUS') or os.environ.get('MEMARENA_DATASET_DIR')
            or Path(__file__).resolve().parents[3] / 'data' / 'benchmark')
CACHE = Path(os.environ.get('MEMARENA_TOKEN_CACHE', '/tmp/session_tokens.json'))
TOKEN_RE = re.compile(r'[A-Za-z0-9_]+')
SIM_BASE = datetime(2026, 4, 1)


def tokenize(s: str):
    return [w.lower() for w in TOKEN_RE.findall(s or '')]


def serialize(rec: dict, variant: str = 'full') -> str:
    body = '\n'.join(f"  {t.get('speaker_id')}: {t.get('text','')}" for t in rec['turns'])
    if variant == 'noheader':
        # Drop the bracketed header: date, modality, location and the participant list.
        # Same-ego sessions name the ego there; foreign ones cannot. If the C3 effect is a
        # header artefact it should collapse here.
        return body
    if variant == 'textonly':
        # Also drop the speaker prefix, so no name reaches BM25 from structure at all.
        return '\n'.join((t.get('text', '') or '') for t in rec['turns'])
    ts = (SIM_BASE + timedelta(days=float(rec['turns'][0].get('timestamp', 0)))).strftime('%B %d, %Y')
    head = (f"[{ts} | {rec.get('modality','')} @ {rec.get('location_id','')} "
            f"| [{', '.join(rec.get('participants',[]))}]]")
    return head + '\n' + body


def fractional_ranks(scores):
    """Average rank within each tie group. Returns rank per index, 1-based."""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks, order


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget', type=int, default=100)
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0, help='debug: cap cohort size')
    ap.add_argument('--doc-variant', choices=('full', 'noheader', 'textonly'), default='full',
                    help="what each BM25 document contains. 'noheader' drops the date/"
                         "modality/location/participant header; 'textonly' also drops the "
                         "per-turn speaker prefix, so no name reaches BM25 from structure.")
    ap.add_argument('--mask-query-ego', action='store_true',
                    help='strip the target ego\'s own name tokens from the query before '
                         'scoring, so BM25 cannot match on them.')
    ap.add_argument('--length-metric', choices=('qwen', 'bm25'), default='qwen',
                    help="unit for nearest-neighbour length matching. 'qwen' is the Qwen3-8B "
                         "serialized token count E2 used; 'bm25' is the regex token count "
                         "e3_endtoend.py used, so --length-metric bm25 reproduces E3's pools.")
    args = ap.parse_args()

    ego_map = json.load(open(BASE / 'ego_session_map.json'))
    sessions, doc = {}, {}
    with gzip.open(BASE / 'corpus_sessions.jsonl.gz', 'rt') as f:
        for line in f:
            r = json.loads(line)
            sessions[r['session_id']] = r
            doc[r['session_id']] = tokenize(serialize(r, args.doc_variant))
    all_sids = sorted(sessions)
    if args.length_metric == 'bm25':
        tok_len = {sid: len(d) for sid, d in doc.items()}
    else:
        tok_len = json.load(open(CACHE))
    print(f'corpus: {len(all_sids)} sessions, length metric = {args.length_metric}, '
          f'doc variant = {args.doc_variant}, mask-query-ego = {args.mask_query_ego}', flush=True)

    cohort = []
    for p in sorted(glob.glob(str(BASE / 'eval_instances' / '*.jsonl'))):
        for line in open(p):
            r = json.loads(line)
            e = r.get('ego_agent_id')
            ev = [s for s in (r.get('evidence_session_ids') or []) if s in sessions]
            if not e or not ev:
                continue
            mem = [s for s in (ego_map.get(e) or []) if s in sessions]
            if mem and all(g in set(mem) for g in ev):
                cohort.append((r['instance_id'], r['dimension'], e, ev, mem, r.get('query', '')))
    if args.limit:
        cohort = cohort[:args.limit]
    print(f'cohort: {len(cohort)} items', flush=True)

    rows, diags, t0 = [], [], time.time()
    for si in range(args.seeds):
        for n, (iid, dim, ego, gold, mem, query) in enumerate(cohort):
            full_map = set(ego_map.get(ego) or [])
            q = tokenize(query)
            if args.mask_query_ego:
                drop = {w.lower() for w in ego.split('_')}
                q = [t for t in q if t not in drop]
            for arm in ('c0star', 'c0prime', 'c3'):
                pool, d = build_pool(
                    arm=arm, instance_id=iid, ego_id=ego, gold=gold,
                    ego_full_map=full_map, ego_pool=mem, all_sessions=all_sids,
                    tok=tok_len, budget=args.budget, seed=si)
                if not pool or d.get('infeasible'):
                    diags.append({'iid': iid, 'arm': arm, 'seed': si, **d})
                    continue
                bm = BM25Okapi([doc[s] for s in pool])
                sc = list(bm.get_scores(q))
                ranks, order = fractional_ranks(sc)
                gi = [i for i, s in enumerate(pool) if s in set(gold)]
                di = [i for i in range(len(pool)) if i not in set(gi)]
                r_min = min(ranks[i] for i in gi)
                top10 = {pool[i] for i in order[:10]}
                gmax, dmax = max(sc[i] for i in gi), max(sc[i] for i in di)
                rows.append(dict(
                    seed=si, arm=arm, iid=iid, dim=dim, ego=ego, n_gold=len(gold),
                    nr_first=(r_min - 1) / max(1, len(pool) - len(gold)),
                    rr_first=1.0 / r_min,
                    gold_recall10=len(set(gold) & top10) / len(gold),
                    all_gold10=float(set(gold) <= top10),
                    outrank=('>' if dmax > gmax else ('=' if dmax == gmax else '<')),
                    n_above=sum(1 for i in di if ranks[i] < r_min),
                    gold_tied=float(sum(1 for i in gi if sum(1 for k in range(len(sc))
                                                             if sc[k] == sc[i]) > 1) > 0),
                    caliper_fail=d.get('caliper_fail', 0),
                    token_rel_diff=d.get('token_rel_diff', 0.0),
                    c3_overlap=d.get('overlap_with_ego_map'),
                ))
            if n and n % 400 == 0:
                print(f'  seed {si} item {n}/{len(cohort)}  {time.time()-t0:.0f}s', flush=True)

    json.dump({'rows': rows, 'diags': diags, 'budget': args.budget, 'seeds': args.seeds},
              open(args.out, 'w'))
    print(f'wrote {args.out}: {len(rows)} rows, {len(diags)} infeasible, '
          f'{time.time()-t0:.0f}s', flush=True)

    by = defaultdict(list)
    for r in rows:
        by[r['arm']].append(r)
    print(f"\n{'arm':9s}{'n':>7}{'NR_first':>10}{'RR_first':>10}{'Recall@10':>11}"
          f"{'outrank>':>10}{'c3 overlap':>12}")
    for arm in ('c0star', 'c0prime', 'c3'):
        v = by[arm]
        if not v:
            continue
        ov = [r['c3_overlap'] for r in v if r['c3_overlap'] is not None]
        print(f"{arm:9s}{len(v):>7}{st.mean(r['nr_first'] for r in v):>10.4f}"
              f"{st.mean(r['rr_first'] for r in v):>10.4f}"
              f"{st.mean(r['gold_recall10'] for r in v):>11.4f}"
              f"{sum(1 for r in v if r['outrank']=='>')/len(v):>10.4f}"
              f"{(sum(ov) if ov else 0):>12}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
