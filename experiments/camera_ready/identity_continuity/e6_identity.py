"""E19/E6 — cross-session agent-identity continuity, isolated.

The C3 ablation swapped same-ego distractors for foreign ones, which moved persona, topic,
relationships and entity recurrence at once. This holds *everything* fixed and changes only
whether an agent's identity is the same label across sessions.

Three arms over one identical pool per item:

  raw     no renaming. A sanity reference, not the baseline: it differs from the other two
          in token count, because a two-token real name becomes a one-token pseudonym.
  sham    one global mapping. Agent A is PERSON_07 in the query, in gold and in every
          distractor. Cross-session identity is preserved.
  break   query and gold use sham's mapping; each non-gold session remaps agents
          independently, so A is PERSON_07 in gold but PERSON_12 / PERSON_31 / ... across
          distractors. Cross-session identity is destroyed and nothing else is.

sham and break perform the same number of substitutions on the same surface forms with
one-token pseudonyms, so renaming artefacts cancel in the contrast. The claim this licenses
is about cross-session *agent-identity* continuity only — the simulator reliably identifies
agent IDs, speakers and their name forms; generic person names and locations are not
covered and are not attempted.

Sanity: raw ~ sham. Effect: break - sham.

  python3 e6_identity.py --budget 100 --seeds 5 --out e6_bm25.json
  python3 e6_identity.py --budget 100 --seeds 5 --dense --out e6_dense.json
"""

from __future__ import annotations

import argparse, glob, gzip, hashlib, json, os, re, sys, time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pools import build_pool                                                # noqa: E402
from e2_structural import BASE, CACHE, serialize, tokenize, fractional_ranks  # noqa: E402
from rank_bm25 import BM25Okapi                                             # noqa: E402

ARMS = ('raw', 'sham', 'break')


def surface_forms(slug: str):
    """Longest first, so 'Amina Suleiman' is replaced before 'Amina'."""
    parts = slug.split('_')
    first, last = parts[0].title(), parts[-1].title()
    forms = {slug, f'{first} {last}', first, last}
    return sorted((f for f in forms if len(f) > 2), key=len, reverse=True)


def pseudonym(idx: int) -> str:
    # one BM25 token, fixed width, so every agent costs the same regardless of real name
    return f'PERSON_{idx:02d}'


def mapping(agents, seed, salt=''):
    """Deterministic agent -> pseudonym index, permuted by (seed, salt)."""
    order = sorted(agents, key=lambda a: hashlib.sha256(
        f'{seed}|{salt}|{a}'.encode()).hexdigest())
    return {a: pseudonym(i) for i, a in enumerate(order)}


def build_matcher(forms_by_agent):
    """One alternation over every surface form, longest first, plus form -> agent.

    Looping re.sub per form meant ~400 full scans of every document; a single compiled
    alternation is one scan. On 13,343 sessions that is the difference between hours and
    seconds.
    """
    owner = {}
    for agent, forms in forms_by_agent:
        for f in forms:
            owner.setdefault(f, agent)
    pat = re.compile(r'\b(' + '|'.join(re.escape(f) for f in
                     sorted(owner, key=len, reverse=True)) + r')\b')
    return pat, owner


def apply_map(text: str, matcher, amap) -> str:
    pat, owner = matcher
    return pat.sub(lambda m: amap.get(owner[m.group(0)], m.group(0)), text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget', type=int, default=100)
    ap.add_argument('--seeds', type=int, default=5, help='mapping seeds')
    ap.add_argument('--pool-seed', type=int, default=0, help='pool draw, held fixed')
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda:6')  # reserved; dense path is e6_dense.py
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    ego_map = json.load(open(BASE / 'ego_session_map.json'))
    sessions = {}
    with gzip.open(BASE / 'corpus_sessions.jsonl.gz', 'rt') as f:
        for line in f:
            r = json.loads(line)
            sessions[r['session_id']] = r
    all_sids = sorted(sessions)
    tok_len = json.load(open(CACHE))
    agents = sorted({a for r in sessions.values() for a in (r.get('participants') or [])}
                    | {t.get('speaker_id') for r in sessions.values() for t in r['turns']}
                    - {None})
    matcher = build_matcher([(a, surface_forms(a)) for a in agents])
    print(f'{len(agents)} agents, {len(sessions)} sessions', flush=True)

    raw_text = {s: serialize(r) for s, r in sessions.items()}

    cohort = []
    for p in sorted(glob.glob(str(BASE / 'eval_instances' / '*.jsonl'))):
        for line in open(p):
            r = json.loads(line)
            e = r.get('ego_agent_id')
            ev = [s for s in (r.get('evidence_session_ids') or []) if s in sessions]
            mem = [s for s in (ego_map.get(e) or []) if s in sessions]
            if e and ev and mem and all(g in set(mem) for g in ev):
                cohort.append((r['instance_id'], r['dimension'], e, ev, mem, r.get('query', '')))
    if args.limit:
        cohort = cohort[:args.limit]
    print(f'cohort {len(cohort)}', flush=True)

    rows, t0 = [], time.time()
    for ms in range(args.seeds):
        gmap = mapping(agents, ms)                      # the one global mapping
        # Both mappings depend only on (session, mapping_seed), never on the item, so the
        # whole corpus is rewritten twice per seed instead of once per item per arm. That is
        # the difference between ten minutes and seven hours.
        print(f'  mapping-seed {ms}: rewriting corpus...', flush=True)
        sham_tok = {s: tokenize(apply_map(raw_text[s], matcher, gmap))
                    for s in raw_text}
        break_tok = {s: tokenize(apply_map(raw_text[s], matcher,
                                           mapping(agents, ms, salt=s)))
                     for s in raw_text}
        raw_tok = {s: tokenize(raw_text[s]) for s in raw_text}
        print(f'    done {time.time()-t0:.0f}s', flush=True)
        for n, (iid, dim, ego, gold, mem, query) in enumerate(cohort):
            pool, d = build_pool(arm='c0star', instance_id=iid, ego_id=ego, gold=gold,
                                 ego_full_map=set(ego_map.get(ego) or []), ego_pool=mem,
                                 all_sessions=all_sids, tok=tok_len,
                                 budget=args.budget, seed=args.pool_seed)
            if not pool or d.get('infeasible'):
                continue
            goldset = set(gold)
            q_raw = tokenize(query)
            q_map = tokenize(apply_map(query, matcher, gmap))
            for arm in ARMS:
                if arm == 'raw':
                    dt = [raw_tok[s] for s in pool]; qt = q_raw
                elif arm == 'sham':
                    dt = [sham_tok[s] for s in pool]; qt = q_map
                else:
                    dt = [sham_tok[s] if s in goldset else break_tok[s] for s in pool]
                    qt = q_map
                bm = BM25Okapi(dt)
                sc = list(bm.get_scores(qt))
                ranks, order = fractional_ranks(sc)
                gi = [i for i, s in enumerate(pool) if s in goldset]
                r_min = min(ranks[i] for i in gi)
                top10 = {pool[i] for i in order[:10]}
                rows.append(dict(
                    mapping_seed=ms, seed=ms, arm=arm, iid=iid, dim=dim, ego=ego,
                    n_gold=len(gold),
                    nr_first=(r_min - 1) / max(1, len(pool) - len(gold)),
                    rr_first=1.0 / r_min,
                    gold_recall10=len(goldset & top10) / len(goldset),
                    all_gold10=float(goldset <= top10),
                    pool_tokens=sum(len(x) for x in dt),
                ))
            if n and n % 300 == 0:
                print(f'  mapping-seed {ms} item {n}/{len(cohort)}  {time.time()-t0:.0f}s',
                      flush=True)

    json.dump({'rows': rows, 'budget': args.budget, 'seeds': args.seeds,
               'pool_seed': args.pool_seed}, open(args.out, 'w'))
    print(f'wrote {args.out}: {len(rows)} rows, {time.time()-t0:.0f}s')

    import statistics as st
    by = defaultdict(list)
    for r in rows:
        by[r['arm']].append(r)
    print(f"\n{'arm':7s}{'NR_first':>10}{'RR_first':>10}{'Recall@10':>11}{'pool tokens':>13}")
    for a in ARMS:
        v = by[a]
        if v:
            print(f"{a:7s}{st.mean(r['nr_first'] for r in v):>10.4f}"
                  f"{st.mean(r['rr_first'] for r in v):>10.4f}"
                  f"{st.mean(r['gold_recall10'] for r in v):>11.4f}"
                  f"{st.mean(r['pool_tokens'] for r in v):>13.0f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
