#!/usr/bin/env python3
"""Generate the paired-bootstrap p-value table → ``paper/tables/appendix_pvalues.tex``.

For each structural claim in the paper (e.g. "Memobase × Qwen3-32B beats
Oracle × Qwen3-32B on Reasoning"), we compute:

* $\\Delta$ (pp): macro-mean accuracy difference A − B, averaged over
  stochastic seeds.
* 95 % bootstrap CI: percentile CI from resampling the paired per-instance
  correctness arrays with replacement.
* Two-sided $p$-value: fraction of bootstrap resamples where $\\Delta$
  crosses zero, doubled (min-tail convention). Stars: ``**`` for $p<0.001$,
  ``*`` for $p<0.01$.

Pairing is at the ``question_id`` granularity so both cells in a comparison
are scored on exactly the same instances; cross-seed pairing averages the
per-seed paired differences.

Runtime note: bootstrap with ``B=2000`` resamples and ~200 instances × 4
dimensions × 20 comparisons is ~10s on CPU; no GPU needed.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper_data import (  # noqa: E402
    BACKEND_TEX, MODEL_ORDER, MODEL_TEX,
    NEW_FROM_OLD, SEEDS, DEFAULT_RUN,
    load_all_cells,
)
from paths import tables_dir  # noqa: E402

TABLES_DIR = tables_dir()
BOOTSTRAP_B = 2000
RNG_SEED = 1002

# Metrics and their sub-dim memberships.
METRIC_SUBDIMS = {
    "Rec":  ["d5_cloze", "d6_metadata"],                 # D1 + D2
    "Rea":  ["d7_qa", "d8_temporal", "d10_counterfactual", "d1_conflict", "d2_anaphora"],  # D3 + D4
    "Tr":   ["d3_confabulation", "d9_negation", "d4_permission"],  # D5 + D6
}

# Claims to test: A (treated) vs B (reference) on each metric.
# Each entry is (A_backend, A_model_label, B_backend, B_model_label, model_key).
CLAIMS = []
# Structured vs Oracle for every model available in both.
for model in ["0_6b", "llama3b", "8b", "32b"]:
    for backend in ["memobase", "memos"]:
        CLAIMS.append((backend, model, "oracle", model))
# Mem0 partial comparison: Mem0 is s1-only so it will render as n=1 (no std),
# but we still emit the row for completeness of the table.
CLAIMS.append(("mem0", "0_6b", "oracle", "0_6b"))


# ---------------------------------------------------------------------------
# Load per-instance correctness so pairing works.
# ---------------------------------------------------------------------------

def _load_instance_correct(path: Path) -> dict[str, tuple[str, bool]]:
    """Return {question_id: (sub_dim, correct_bool)}.

    Sub-dim is derived from the qid prefix when the JSON lacks a ``dimension``
    field (structured memcache dumps).
    """
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    # Build qid-prefix -> sub_dim_id map from NEW_FROM_OLD:
    qid_to_sub: dict[str, str] = {}
    for paper_dim, subs in NEW_FROM_OLD.items():
        for sub in subs:
            prefix = sub.split("_", 1)[0]
            qid_to_sub[prefix] = sub
    out: dict[str, tuple[str, bool]] = {}
    for r in data.get("details", []):
        qid = r.get("question_id") or ""
        if not qid:
            continue
        if r.get("answer_scored") is False:
            continue
        raw = r.get("raw_response") or ""
        if isinstance(raw, str) and (raw.startswith("LLM_ERROR") or raw.startswith("TEXT_SESSIONS_ERROR")):
            continue
        sub = r.get("dimension")
        if not sub:
            sub = qid_to_sub.get(qid.split("_", 1)[0])
        if not sub:
            continue
        if r.get("scoring_method") == "judge":
            val = r.get("judge_correct")
            if val is None:
                val = r.get("correct", False)
        else:
            val = r.get("correct", False)
        out[qid] = (sub, bool(val))
    return out


def _paired_metric_delta(corrA, corrB, metric: str, qid_sample) -> float:
    """Δ on the given metric using a specific qid list (bootstrap sample)."""
    subs = METRIC_SUBDIMS[metric]
    a_c = a_n = b_c = b_n = 0
    # Per-dim then average (macro) to match main-table "Rec/Rea/Tr" definition.
    per_sub_a = defaultdict(lambda: [0, 0])
    per_sub_b = defaultdict(lambda: [0, 0])
    for qid in qid_sample:
        if qid in corrA and qid in corrB:
            sA, vA = corrA[qid]
            sB, vB = corrB[qid]
            if sA in subs and sB == sA:
                per_sub_a[sA][1] += 1
                per_sub_a[sA][0] += int(vA)
                per_sub_b[sB][1] += 1
                per_sub_b[sB][0] += int(vB)

    def _macro(per_sub):
        accs = []
        for k, (c, n) in per_sub.items():
            if n:
                accs.append(c / n)
        return sum(accs) / len(accs) if accs else None

    a = _macro(per_sub_a)
    b = _macro(per_sub_b)
    if a is None or b is None:
        return float("nan")
    return a - b


def _agent_from_meta(meta: dict) -> str | None:
    return (
        meta.get("ego_agent_id")
        or meta.get("instance_query_agent")
        or meta.get("answerer_agent_id")
        or meta.get("target_agent")
        or meta.get("ego")
    )


def _load_qid_to_agent_from_qars(path: Path) -> dict[str, str]:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[str, str] = {}
    for q in data.get("qars", []):
        if not isinstance(q, dict):
            continue
        qid = q.get("id") or q.get("question_id") or q.get("instance_id")
        meta = q.get("meta") or q.get("metadata") or {}
        agent = _agent_from_meta(meta) if isinstance(meta, dict) else None
        if qid and agent:
            out[str(qid)] = str(agent)
    return out


def _load_qid_to_agent_from_eval_instances(run_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    inst_dir = run_dir / "eval_instances"
    if not inst_dir.exists():
        return out
    for path in sorted(inst_dir.glob("*.jsonl")):
        try:
            lines = path.read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            qid = row.get("instance_id") or row.get("id") or row.get("question_id")
            meta = row.get("metadata") or {}
            agent = (
                row.get("ego_agent_id")
                or row.get("answerer_agent_id")
                or row.get("asker_agent_id")
                or row.get("ego")
                or (_agent_from_meta(meta) if isinstance(meta, dict) else None)
            )
            if qid and agent:
                out[str(qid)] = str(agent)
    return out


def _load_qid_to_agent(run_dir: Path = DEFAULT_RUN) -> dict[str, str]:
    """Return ``{qid: ego_agent_id}`` from staged QA files or eval instances.

    Older paper runs wrote one canonical
    ``eval_results/oracle/masim_qa_oracle_*.json`` file. The public staging
    helper writes seed-scoped ``eval_results_s*/.../masim_qa_*.json`` files
    and also links ``eval_instances/``. Try all of those layouts before
    falling back to instance-level bootstrap.
    """
    candidates: list[Path] = []
    candidates.extend(sorted((run_dir / "eval_results" / "oracle").glob("masim_qa_oracle_*.json")))
    candidates.extend(sorted(run_dir.glob("eval_results_s*/oracle/masim_qa_oracle_*.json")))
    candidates.extend(sorted(run_dir.glob("eval_results_s*/*/masim_qa_*.json")))
    for src in candidates:
        out = _load_qid_to_agent_from_qars(src)
        if out:
            print(f"[gen_pvalues] qid→agent source: {src}")
            return out

    out = _load_qid_to_agent_from_eval_instances(run_dir)
    if out:
        print(f"[gen_pvalues] qid→agent source: {run_dir / 'eval_instances'}")
        return out

    print(
        f"[gen_pvalues] WARNING: no masim_qa_*.json or eval_instances metadata under {run_dir}; "
        "falling back to instance-level bootstrap."
    )
    return {}


def _bootstrap(corrA_seeds, corrB_seeds, metric: str,
               qid_to_agent: dict[str, str] | None = None,
               B: int = BOOTSTRAP_B):
    """Paired CLUSTER bootstrap over agents, averaged across the seeds we have.

    Strategy:
      1. Per-instance correctness is averaged across seeds first
         (avoids treating the same qid under different seeds as i.i.d.).
      2. Instances are grouped by ego_agent_id (~50 clusters total).
      3. Each bootstrap iteration resamples AGENTS with replacement,
         then takes ALL qids of the resampled agents. This avoids
         pseudoreplication from intra-agent / intra-session correlation
         (1,922 instances span 50 agents, so multiple instances per
         agent are not independent).
      (Revised 2026-04-25 per 0426 reviewer feedback; previous instance-
      level bootstrap underestimated CI width.)

    If ``qid_to_agent`` is empty / None / contains no overlap with the
    available qids, falls back to the prior instance-level bootstrap so
    the script never silently produces wrong stats.
    """
    rng = random.Random(RNG_SEED)
    # Aggregate per-instance correctness across seeds to avoid
    # pseudoreplication (reviewer #7 feedback).
    inst_agg = {}  # qid -> (sub, [vA_per_seed], [vB_per_seed])
    for idx, (a, b) in enumerate(zip(corrA_seeds, corrB_seeds)):
        for qid, (sub, vA) in a.items():
            if qid in b:
                _, vB = b[qid]
                if vA is None or vB is None:
                    continue
                if qid not in inst_agg:
                    inst_agg[qid] = (sub, [], [])
                inst_agg[qid][1].append(int(vA))
                inst_agg[qid][2].append(int(vB))
    if not inst_agg:
        return None
    # Collapse to per-instance mean correctness
    pool = {}
    for qid, (sub, vas, vbs) in inst_agg.items():
        pool[qid] = (sub, sum(vas) / len(vas), sum(vbs) / len(vbs))
    keys = list(pool.keys())

    def _macro_delta(sample_keys):
        per_sub_a = defaultdict(lambda: [0, 0])
        per_sub_b = defaultdict(lambda: [0, 0])
        subs = METRIC_SUBDIMS[metric]
        for k in sample_keys:
            sub, vA, vB = pool[k]
            if sub not in subs:
                continue
            per_sub_a[sub][1] += 1
            per_sub_a[sub][0] += vA
            per_sub_b[sub][1] += 1
            per_sub_b[sub][0] += vB

        def _macro(per_sub):
            accs = []
            for k, (c, n) in per_sub.items():
                if n:
                    accs.append(c / n)
            return sum(accs) / len(accs) if accs else None

        a = _macro(per_sub_a)
        b = _macro(per_sub_b)
        if a is None or b is None:
            return None
        return a - b

    point = _macro_delta(keys)
    if point is None:
        return None
    deltas = []
    # Cluster bootstrap by ego_agent_id when mapping available; else fall
    # back to instance-level (preserves backward compat in degraded envs).
    agents_to_qids: dict[str, list[str]] = defaultdict(list)
    if qid_to_agent:
        for k in keys:
            a = qid_to_agent.get(k)
            if a is not None:
                agents_to_qids[a].append(k)
    agent_list = list(agents_to_qids.keys())
    if agent_list:
        n_agents = len(agent_list)
        for _ in range(B):
            sample: list[str] = []
            for _ in range(n_agents):
                a = agent_list[rng.randrange(n_agents)]
                sample.extend(agents_to_qids[a])
            v = _macro_delta(sample)
            if v is not None:
                deltas.append(v)
    else:
        # Fallback: no agent metadata → revert to instance-level bootstrap.
        n = len(keys)
        for _ in range(B):
            sample = [keys[rng.randrange(n)] for _ in range(n)]
            v = _macro_delta(sample)
            if v is not None:
                deltas.append(v)
    if not deltas:
        return None
    deltas.sort()
    lo = deltas[int(0.025 * len(deltas))]
    hi = deltas[int(0.975 * len(deltas))]
    # Two-sided p: min-tail mass doubled.
    p_left = sum(1 for d in deltas if d <= 0) / len(deltas)
    p_right = sum(1 for d in deltas if d >= 0) / len(deltas)
    p = 2 * min(p_left, p_right)
    p = min(p, 1.0)
    return (point, lo, hi, p)


def _stars(p: float) -> str:
    if p < 0.001:
        return "$^{**}$"
    if p < 0.01:
        return "$^{*}$"
    return ""


def _format_p(p: float, B: int = BOOTSTRAP_B) -> str:
    """Format p-values without implying an exact zero.

    With ``B`` bootstrap iterations, a zero empirical tail count means the
    p-value is below the bootstrap resolution, not that the true p-value is
    literally zero.
    """
    min_p = 2.0 / B
    if p < min_p - 1e-12:
        return "$<\\!0.001^{**}$"
    return f"{p:.4f}{_stars(p)}"


def main() -> None:
    grid = load_all_cells(DEFAULT_RUN)
    qid_to_agent = _load_qid_to_agent(DEFAULT_RUN)
    n_agents = len(set(qid_to_agent.values()))
    if qid_to_agent:
        print(f"[gen_pvalues] loaded qid→agent for {len(qid_to_agent)} qids "
              f"across {n_agents} agents (cluster bootstrap target)")

    rows: list[str] = []
    for A_backend, A_model, B_backend, B_model in CLAIMS:
        # Collect per-seed instance maps; skip if neither cell exists at all.
        corrA = [
            _load_instance_correct(grid[(s, A_backend, A_model)].source_path)
            for s in SEEDS
            if (s, A_backend, A_model) in grid
        ]
        corrB = [
            _load_instance_correct(grid[(s, B_backend, B_model)].source_path)
            for s in SEEDS
            if (s, B_backend, B_model) in grid
        ]
        if not corrA or not corrB:
            continue
        # Trim to the number of seeds that both sides share.
        n = min(len(corrA), len(corrB))
        corrA = corrA[:n]
        corrB = corrB[:n]

        label_A = f"{BACKEND_TEX[A_backend]} $\\times$ {MODEL_TEX[A_model]}"
        label_B = f"{BACKEND_TEX[B_backend]} $\\times$ {MODEL_TEX[B_model]}"
        for metric in ("Rec", "Rea", "Tr"):
            result = _bootstrap(corrA, corrB, metric, qid_to_agent=qid_to_agent)
            if result is None:
                rows.append(f"{label_A} & {label_B} & {metric} & -- & -- & -- & -- \\\\")
                continue
            delta, lo, hi, p = result
            rows.append(
                f"{label_A} & {label_B} & {metric} "
                f"& {delta * 100:+.2f} & {lo * 100:+.2f} & {hi * 100:+.2f} "
                f"& {_format_p(p)} \\\\"
            )

    tex = (
        "\\begin{table}[h]\n\\centering\n"
        "\\caption{Paired \\emph{cluster} bootstrap significance tests "
        "(two-sided, $B=2000$) for structural claims on \\benchL{}. "
        "\\emph{A} and \\emph{B} identify the two cells being compared on "
        "paired per-instance outcomes. $\\Delta$ is the macro-mean accuracy "
        "difference ($A-B$) in percentage points; 95\\% CI is percentile "
        "bootstrap; $p$ is two-sided (min-tail doubled), reported as "
        "$<\\!0.001$ when the empirical tail mass is below the $B=2000$ "
        "resolution limit. Stars: $p<0.001$ ($^{**}$), $p<0.01$ ($^{*}$). Stochastic seeds: s2/s3/s4; per-"
        "instance correctness is averaged across seeds first, then "
        "resampled at the agent level (cluster bootstrap over the 50 ego "
        "agents) so that intra-agent / intra-session correlation does not "
        "inflate the precision of the CI.}\n"
        "\\label{tab:pvalues}\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{@{}ll l rrr r@{}}\n\\toprule\n"
        "\\textbf{A} & \\textbf{B} & Metric & $\\Delta$ (pp) & 95\\% CI lo "
        "& 95\\% CI hi & $p$ \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLES_DIR / "appendix_pvalues.tex"
    out.write_text(tex)
    print(f"[gen_pvalues] wrote {out}")


if __name__ == "__main__":
    main()
