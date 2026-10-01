"""Deterministic pool construction for the matched-reachability coherence ablation.

Shared by E2 (offline BM25 ranking) and E3 (the eval-pipeline adapter), so both see
byte-identical pools. See docs/superpowers/specs/2026-07-27-w2-coherence-ablation-design.md

Arms, all at budget B with all-gold-in-pool = 100%:
  c0star  gold + (B-|G|) same-ego non-gold, drawn at the arm seed
  c0prime gold + (B-|G|) same-ego non-gold, independent draw, length-matched to c0star
  c3      gold + (B-|G|) foreign sessions, length-matched to c0star

Foreign means "not in the target ego's FULL map" -- 38.5% of sessions sit in more than one
ego map, so excluding only the sampled pool would leak the target's own sessions back in.
"""

from __future__ import annotations

import bisect
import hashlib
import random
from typing import Dict, List, Sequence, Set, Tuple

CALIPER = 0.10          # nearest-neighbour length caliper, fraction of the target's tokens
CALIPER_WIDE = 0.20     # first fallback, per the spec's matching-failure policy


def _rng(*parts: object) -> random.Random:
    """Seed deterministically from the item identity, so a pool never depends on
    iteration order or on how many items were processed before it."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def _match_by_length(
    targets: Sequence[str],
    donors: Sequence[str],
    tok: Dict[str, int],
    rng: random.Random,
) -> Tuple[List[str], int, int]:
    """Nearest-neighbour match each target to an unused donor of similar token length.

    Returns (picked, n_caliper_fail, n_widened). Bisect over a sorted donor list keeps this
    O(len(targets) * log(len(donors))) rather than the quadratic scan the feasibility
    sample used.
    """
    order = sorted(donors, key=lambda s: tok[s])
    lens = [tok[s] for s in order]
    used: Set[int] = set()
    picked: List[str] = []
    fail = widened = 0

    for t in sorted(targets, key=lambda s: -tok[s]):   # longest first: hardest to match
        want = tok[t]
        i = bisect.bisect_left(lens, want)
        best_i, best_d = None, None
        # walk outward from the insertion point until the nearest unused donor is found
        lo, hi = i - 1, i
        while lo >= 0 or hi < len(order):
            for j in (hi, lo):
                if j is None or j < 0 or j >= len(order) or j in used:
                    continue
                d = abs(lens[j] - want)
                if best_d is None or d < best_d:
                    best_i, best_d = j, d
            if best_d is not None:
                # anything further out is at least this far away on the closer side
                near = min(
                    abs(lens[hi] - want) if hi < len(order) else 10**9,
                    abs(lens[lo] - want) if lo >= 0 else 10**9,
                )
                if near >= best_d:
                    break
            lo -= 1
            hi += 1

        if best_i is None:
            fail += 1
            continue
        if best_d > CALIPER * want:
            if best_d <= CALIPER_WIDE * want:
                widened += 1
            else:
                fail += 1
        used.add(best_i)
        picked.append(order[best_i])

    rng.shuffle(picked)   # pool order must not encode the matching order
    return picked, fail, widened


def build_pool(
    *,
    arm: str,
    instance_id: str,
    ego_id: str,
    gold: Sequence[str],
    ego_full_map: Set[str],
    ego_pool: Sequence[str],
    all_sessions: Sequence[str],
    tok: Dict[str, int],
    budget: int,
    seed: int,
) -> Tuple[List[str], Dict[str, object]]:
    """Return (pool_session_ids, diagnostics). Gold is always present."""
    gold = list(dict.fromkeys(gold))
    need = budget - len(gold)
    if need <= 0:
        return list(gold), {"need": need, "degenerate": True}

    same_donors = [s for s in ego_pool if s not in set(gold)]
    rng = _rng(arm, instance_id, seed)

    # c0star is the reference draw every other arm is matched against.
    ref_rng = _rng("c0star", instance_id, seed)
    if len(same_donors) < need:
        return [], {"infeasible": "same-ego donors < need", "available": len(same_donors)}
    c0star = ref_rng.sample(same_donors, need)

    diag: Dict[str, object] = {"need": need, "n_gold": len(gold)}

    if arm == "c0star":
        picked, fail, widened = c0star, 0, 0
    elif arm == "c0prime":
        rest = [s for s in same_donors if s not in set(c0star)]
        picked, fail, widened = _match_by_length(c0star, rest, tok, rng)
    elif arm == "c3":
        foreign = [s for s in all_sessions if s not in ego_full_map and s not in set(gold)]
        diag["n_foreign_donors"] = len(foreign)
        picked, fail, widened = _match_by_length(c0star, foreign, tok, rng)
    else:
        raise ValueError(f"unknown arm {arm!r}")

    pool = list(gold) + picked
    ref_tokens = sum(tok[s] for s in c0star) + sum(tok[s] for s in gold)
    pool_tokens = sum(tok[s] for s in pool)
    diag.update(
        caliper_fail=fail,
        caliper_widened=widened,
        pool_tokens=pool_tokens,
        token_abs_diff=abs(pool_tokens - ref_tokens),
        token_rel_diff=abs(pool_tokens - ref_tokens) / max(1, ref_tokens),
        size=len(pool),
        all_gold_in_pool=all(g in pool for g in gold),
        overlap_with_ego_map=(
            len([s for s in picked if s in ego_full_map]) if arm == "c3" else None
        ),
    )
    return pool, diag
