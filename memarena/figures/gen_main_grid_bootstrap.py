#!/usr/bin/env python3
"""Generate a current-main-grid bootstrap table from ``out/``.

This replaces the retired stale-run p-value table for the paper claims that
actually depend on the current MemArena-L main grid.  It reads the committed
``out/accuracy_memarena_l_*`` and ``out/accuracy_memarena_l_baselines_*``
cells, pairs each backend against Oracle by question id and seed, and reports
an exploratory paired ego-cluster bootstrap.  Since all agents belong to one
shared social world, the ego clusters are not independent exchangeable units;
the diagnostic p-values should not be read as inferential significance tests.

Default output is intentionally narrow: MemSearch vs Oracle over Rec, Rea, D4,
D6-F1PU, and Trust.  Use ``--backends`` and ``--metrics`` to widen it for audit
runs without changing the paper-facing table.

D6 follows ``eval.src.permission_metrics``: by default only the 144 items that
carry a protected fact, and a leak is a DENY item labelled DISCLOSE_CORRECT.
``--population all --leak legacy`` reproduces the May-2026 table (all 200
items; leak = fact flag or DISCLOSE_CORRECT).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from eval.src.permission_metrics import (  # noqa: E402
    LEAK_LABEL,
    POPULATION_ALL,
    POPULATION_FACT,
    load_instances,
    population_ids,
)
from memarena.figures.main_grid import default_instances_path  # noqa: E402
from memarena.figures.paper_data import RESULTS_ROOT  # noqa: E402
from memarena.figures.paths import table_path  # noqa: E402

OUT_ROOT = RESULTS_ROOT / "out"
LEAK_LEGACY = "legacy"

MODEL_ORDER = ("0_6b", "llama3b", "7b", "8b", "32b")
MODEL_TEX = {
    "0_6b": "Qwen3-0.6B",
    "llama3b": "Llama-3.2-3B",
    "7b": "Mistral-7B",
    "8b": "Qwen3-8B",
    "32b": "Qwen3-32B",
}

BACKEND_TEX = {
    "vanilla": "Vanilla",
    "rag": "RAG",
    "memobase": "Memobase",
    "memsearch": "MemSearch",
    "oracle": "Oracle",
}

SEEDS = ("s2", "s3", "s4")
DEFAULT_BACKENDS = ("memsearch",)
# D6-F1PU and Trust are excluded: the fact-bearing D6 items sit in too few ego
# clusters (ALLOW: 2) for an ego-cluster bootstrap. Pass them via --metrics
# (with --population all --leak legacy) to reproduce the May-2026 table.
DEFAULT_METRICS = ("Rec", "Rea", "D4")

PREFIX_TO_SUBDIM = {
    "d1": "d1_conflict",
    "d2": "d2_anaphora",
    "d3": "d3_confabulation",
    "d4": "d4_permission",
    "d5": "d5_cloze",
    "d6": "d6_metadata",
    "d7": "d7_qa",
    "d8": "d8_temporal",
    "d10": "d10_counterfactual",
}

METRIC_SUBDIMS = {
    "Rec": {"d5_cloze", "d6_metadata"},
    "Rea": {"d7_qa", "d8_temporal", "d10_counterfactual", "d1_conflict", "d2_anaphora"},
    "D4": {"d1_conflict", "d2_anaphora"},
    "D5": {"d3_confabulation"},
}


@dataclass
class QidPair:
    subdim: str
    a_correct: list[float] = field(default_factory=list)
    b_correct: list[float] = field(default_factory=list)
    is_allow: bool = False
    is_deny: bool = False
    a_leak: list[float] = field(default_factory=list)
    b_leak: list[float] = field(default_factory=list)
    a_disclose_correct: list[float] = field(default_factory=list)
    b_disclose_correct: list[float] = field(default_factory=list)

    def a_acc(self) -> float:
        return _mean(self.a_correct)

    def b_acc(self) -> float:
        return _mean(self.b_correct)

    def a_leak_rate(self) -> float:
        return _mean(self.a_leak)

    def b_leak_rate(self) -> float:
        return _mean(self.b_leak)

    def a_utility(self) -> float:
        return _mean(self.a_disclose_correct)

    def b_utility(self) -> float:
        return _mean(self.b_disclose_correct)


@dataclass(frozen=True)
class BootstrapRow:
    backend: str
    model: str
    metric: str
    delta: float
    lo: float
    hi: float
    p_value: float
    n_qids: int
    n_agents: int
    n_boot: int


def _mean(xs: Iterable[float]) -> float:
    values = list(xs)
    if not values:
        return float("nan")
    return sum(values) / len(values)


def _qid_prefix(qid: str) -> str:
    match = re.match(r"^(d\d+)", qid)
    if not match:
        raise ValueError(f"cannot derive sub-dimension from qid {qid!r}")
    return match.group(1)


def _subdim(qid: str) -> str:
    prefix = _qid_prefix(qid)
    try:
        return PREFIX_TO_SUBDIM[prefix]
    except KeyError as exc:
        raise ValueError(f"unsupported qid prefix {prefix!r} for {qid!r}") from exc


def _load_details(path: Path) -> dict[str, dict]:
    with open(path) as fh:
        data = json.load(fh)
    if isinstance(data, dict) and isinstance(data.get("details"), list):
        rows = data["details"]
    elif isinstance(data, dict) and isinstance(data.get("records"), list):
        rows = data["records"]
    elif isinstance(data, list):
        rows = data
    else:
        raise ValueError(f"{path} is not an evaluation-results JSON with details/records")

    out: dict[str, dict] = {}
    for row in rows:
        qid = row.get("question_id") or row.get("id") or row.get("qid")
        if not qid:
            raise ValueError(f"{path} has a row without question_id/id/qid")
        out[str(qid)] = row
    return out


def _eval_path(out_root: Path, backend: str, model: str, seed: str) -> Path:
    if backend in {"oracle", "vanilla", "rag"}:
        disk_backend = "inmem" if backend == "rag" else backend
        base = out_root / f"accuracy_memarena_l_baselines_{model}"
        return (
            base
            / f"eval_results_{seed}"
            / disk_backend
            / f"evaluation_results_{disk_backend}_{model}_{seed}_judge_remote.json"
        )

    if backend in {"memobase", "memsearch"}:
        disk_model = "3b" if model == "llama3b" else model
        base = out_root / f"accuracy_memarena_l_{disk_model}"
        return (
            base
            / f"eval_results_{seed}"
            / "memory_cache"
            / f"evaluation_results_{backend}_{model}_{seed}_judge_remote.json"
        )

    raise ValueError(f"unknown backend {backend!r}")


def _cell_record(record: dict, leak_mode: str = LEAK_LABEL) -> tuple[float, bool, bool, float, float]:
    correct = 1.0 if bool(record.get("correct")) else 0.0
    expected = str(
        record.get("policy_expected")
        or record.get("expected_answer_mode")
        or ""
    ).upper()
    mode = str(record.get("expected_answer_mode") or "").lower()
    is_allow = expected.startswith("ALLOW") or mode == "disclose"
    is_deny = expected.startswith("DENY") or mode in {"deny", "abstain"}

    cat = str(record.get("policy_category") or record.get("category") or "").upper()
    if leak_mode == LEAK_LEGACY:
        leak = bool(record.get("leaked_fact_in_output") or record.get("fact_in_output"))
        if cat == "DISCLOSE_CORRECT":
            leak = True
        disclose_correct = cat == "DISCLOSE_CORRECT" or leak
    else:
        leak = cat == "DISCLOSE_CORRECT"
        disclose_correct = leak
    return correct, is_allow, is_deny, float(leak), float(disclose_correct)


def _paired_qids(
    out_root: Path, backend: str, model: str, oracle: str = "oracle", leak_mode: str = LEAK_LABEL
) -> dict[str, QidPair]:
    pairs: dict[str, QidPair] = {}
    for seed in SEEDS:
        a_path = _eval_path(out_root, backend, model, seed)
        b_path = _eval_path(out_root, oracle, model, seed)
        if not a_path.exists():
            raise FileNotFoundError(a_path)
        if not b_path.exists():
            raise FileNotFoundError(b_path)

        a_rows = _load_details(a_path)
        b_rows = _load_details(b_path)
        common = sorted(set(a_rows) & set(b_rows))
        if len(common) < 0.95 * max(len(a_rows), len(b_rows)):
            raise RuntimeError(
                f"{backend}/{model}/{seed} only has {len(common)} paired qids "
                f"({len(a_rows)} vs {len(b_rows)} rows)"
            )
        for qid in common:
            a_correct, a_allow, a_deny, a_leak, a_disc = _cell_record(a_rows[qid], leak_mode)
            b_correct, b_allow, b_deny, b_leak, b_disc = _cell_record(b_rows[qid], leak_mode)
            item = pairs.setdefault(qid, QidPair(subdim=_subdim(qid)))
            item.a_correct.append(a_correct)
            item.b_correct.append(b_correct)
            item.is_allow = item.is_allow or a_allow or b_allow
            item.is_deny = item.is_deny or a_deny or b_deny
            if item.subdim == "d4_permission":
                item.a_leak.append(a_leak)
                item.b_leak.append(b_leak)
                item.a_disclose_correct.append(a_disc)
                item.b_disclose_correct.append(b_disc)

    incomplete = [qid for qid, item in pairs.items() if len(item.a_correct) != len(SEEDS)]
    if incomplete:
        raise RuntimeError(f"{backend}/{model} has incomplete seed pairing for {len(incomplete)} qids")
    return pairs


def _extract_agent(row: dict) -> str | None:
    for key in ("ego_agent_id", "instance_query_agent", "answerer_agent_id", "agent_id"):
        value = row.get(key)
        if value:
            return str(value)
    meta = row.get("meta")
    if isinstance(meta, dict):
        for key in ("ego_agent_id", "instance_query_agent", "answerer_agent_id", "agent_id"):
            value = meta.get(key)
            if value:
                return str(value)
    return None


def _qid(row: dict) -> str | None:
    for key in ("id", "question_id", "qid"):
        value = row.get(key)
        if value:
            return str(value)
    return None


def _load_qid_to_agent(path: Path) -> dict[str, str]:
    with open(path) as fh:
        data = json.load(fh)
    if isinstance(data, dict) and isinstance(data.get("qars"), list):
        rows = data["qars"]
    elif isinstance(data, dict) and isinstance(data.get("items"), list):
        rows = data["items"]
    elif isinstance(data, dict) and isinstance(data.get("records"), list):
        rows = data["records"]
    elif isinstance(data, list):
        rows = data
    else:
        raise ValueError(f"{path} is not a QA JSON with qars/items/records")

    mapping: dict[str, str] = {}
    for row in rows:
        qid = _qid(row)
        agent = _extract_agent(row)
        if qid and agent:
            mapping[qid] = agent
    if not mapping:
        raise ValueError(f"could not extract qid->agent mapping from {path}")
    return mapping


def _load_qid_to_agent_from_dataset(instances_dir: Path) -> dict[str, str]:
    """qid -> ego agent straight from the released eval instances."""
    mapping: dict[str, str] = {}
    for path in sorted(instances_dir.glob("*.jsonl")):
        with open(path) as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    if row.get("ego_agent_id"):
                        mapping[str(row["instance_id"])] = str(row["ego_agent_id"])
    if not mapping:
        raise FileNotFoundError(f"no eval instances with ego_agent_id under {instances_dir}")
    return mapping


def _default_qa_map(out_root: Path) -> Path:
    preferred = (
        out_root
        / "b5_topk_sweep_v2"
        / "0_6b"
        / "k8"
        / "eval_results_s2"
        / "inmem"
        / "masim_qa_inmem_0_6b_s2.json"
    )
    if preferred.exists():
        return preferred
    matches = sorted(out_root.glob("**/masim_qa_*.json"))
    if not matches:
        raise FileNotFoundError("no masim_qa_*.json found under out/ for qid->agent clusters")
    return matches[0]


def _score_accuracy(pairs: dict[str, QidPair], qids: list[str], metric: str, side: str) -> float:
    wanted = METRIC_SUBDIMS[metric]
    values: list[float] = []
    for qid in qids:
        item = pairs[qid]
        if item.subdim not in wanted:
            continue
        values.append(item.a_acc() if side == "a" else item.b_acc())
    return _mean(values)


def _score_f1pu(pairs: dict[str, QidPair], qids: list[str], side: str) -> float:
    privacy_values: list[float] = []
    utility_values: list[float] = []
    for qid in qids:
        item = pairs[qid]
        if item.subdim != "d4_permission":
            continue
        if item.is_deny:
            leak = item.a_leak_rate() if side == "a" else item.b_leak_rate()
            privacy_values.append(1.0 - leak)
        elif item.is_allow:
            utility = item.a_utility() if side == "a" else item.b_utility()
            utility_values.append(utility)
    privacy = _mean(privacy_values)
    utility = _mean(utility_values)
    if math.isnan(privacy) or math.isnan(utility) or privacy + utility <= 0:
        return float("nan")
    return 2.0 * privacy * utility / (privacy + utility)


def _score(pairs: dict[str, QidPair], qids: list[str], metric: str, side: str) -> float:
    if metric in METRIC_SUBDIMS:
        return _score_accuracy(pairs, qids, metric, side)
    if metric == "D6-F1PU":
        return _score_f1pu(pairs, qids, side)
    if metric == "Trust":
        d5 = _score_accuracy(pairs, qids, "D5", side)
        d6 = _score_f1pu(pairs, qids, side)
        if math.isnan(d5) or math.isnan(d6):
            return float("nan")
        return 0.5 * (d5 + d6)
    raise ValueError(f"unknown metric {metric!r}")


def _delta(pairs: dict[str, QidPair], qids: list[str], metric: str) -> float:
    return _score(pairs, qids, metric, "a") - _score(pairs, qids, metric, "b")


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] * (hi - pos) + sorted_values[hi] * (pos - lo)


def _bootstrap(
    pairs: dict[str, QidPair],
    qid_to_agent: dict[str, str],
    metric: str,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float, float, int, int]:
    paired_qids = sorted(pairs)
    missing = [qid for qid in paired_qids if qid not in qid_to_agent]
    if missing:
        raise RuntimeError(f"qid->agent mapping is missing {len(missing)} paired qids")

    agent_to_qids: dict[str, list[str]] = defaultdict(list)
    for qid in paired_qids:
        agent_to_qids[qid_to_agent[qid]].append(qid)
    agents = sorted(agent_to_qids)
    point = _delta(pairs, paired_qids, metric)

    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(n_boot):
        sample_qids: list[str] = []
        for agent in rng.choices(agents, k=len(agents)):
            sample_qids.extend(agent_to_qids[agent])
        draws.append(_delta(pairs, sample_qids, metric))

    draws.sort()
    lo = _percentile(draws, 0.025)
    hi = _percentile(draws, 0.975)
    if point < 0:
        tail = sum(1 for value in draws if value >= 0)
    else:
        tail = sum(1 for value in draws if value <= 0)
    p_value = min(1.0, 2.0 * tail / n_boot)
    return point, lo, hi, p_value, len(paired_qids), len(agents)


def _format_pp(value: float) -> str:
    return f"{100.0 * value:+.1f}"


def _format_p(value: float, n_boot: int) -> str:
    floor = 2.0 / n_boot
    if value < floor:
        return "$<0.001$" if floor <= 0.001 else f"$<{floor:.3f}$"
    return f"{value:.3f}"


def _tex_escape(value: str) -> str:
    return value.replace("_", r"\_")


def _write_csv(path: Path, rows: list[BootstrapRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "backend",
                "model",
                "metric",
                "delta_pp",
                "ci_lo_pp",
                "ci_hi_pp",
                "p_value",
                "n_qids",
                "n_agents",
                "n_boot",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "backend": row.backend,
                    "model": row.model,
                    "metric": row.metric,
                    "delta_pp": f"{100.0 * row.delta:.6f}",
                    "ci_lo_pp": f"{100.0 * row.lo:.6f}",
                    "ci_hi_pp": f"{100.0 * row.hi:.6f}",
                    "p_value": f"{row.p_value:.8f}",
                    "n_qids": row.n_qids,
                    "n_agents": row.n_agents,
                    "n_boot": row.n_boot,
                }
            )


def _write_tex(path: Path, rows: list[BootstrapRow], oracle: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body: list[str] = []
    last_backend: str | None = None
    last_model: str | None = None
    for row in rows:
        if last_backend is not None and row.backend != last_backend:
            body.append(r"\midrule")
            last_model = None
        backend = BACKEND_TEX[row.backend] if row.backend != last_backend else ""
        model = MODEL_TEX[row.model] if row.model != last_model or backend else ""
        body.append(
            f"{backend:<10} & {model:<15} & {_tex_escape(row.metric):<8} "
            f"& {_format_pp(row.delta):>6} "
            f"& {_format_pp(row.lo):>6} "
            f"& {_format_pp(row.hi):>6} "
            f"& {_format_p(row.p_value, row.n_boot):>8} \\\\"
        )
        last_backend = row.backend
        last_model = row.model

    tex = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Current \\texttt{out/} main-grid exploratory paired ego-cluster bootstrap against "
        f"{BACKEND_TEX[oracle]}. $\\Delta$ is backend $-$ {BACKEND_TEX[oracle]} in "
        "percentage points, using s2/s3/s4 paired by question id and resampling ego "
        "agents with replacement. Rec and Rea are instance-weighted category "
        "accuracies; D4 is cross-session reasoning accuracy; D6-F1PU uses the "
        "permission utility score $2PU/(P+U)$, with $P=1-$ leak rate on DENY and "
        "$U=$ DISCLOSE\\_CORRECT rate on ALLOW; Trust is the mean of D5 accuracy and "
        "D6-F1PU. Because all agents inhabit one shared social world with overlapping "
        "ties, events, and conversation histories, ego agents are not independent "
        "exchangeable clusters; intervals and diagnostic $p$ values may therefore be "
        "anti-conservative. We report them only as fixed-world exploratory diagnostics, "
        "not inferential significance tests or cross-world generalization claims.}\n"
        "\\label{tab:main-grid-bootstrap}\n"
        "\\scriptsize\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{@{}lllrrrr@{}}\n"
        "\\toprule\n"
        "Backend & Model & Metric & $\\Delta$ & 95\\% lo & 95\\% hi & diag. $p$ \\\\\n"
        "\\midrule\n"
        + "\n".join(body)
        + "\n\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    path.write_text(tex)


def _parse_csv_list(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _validate_choices(name: str, values: tuple[str, ...], allowed: Iterable[str]) -> None:
    allowed_set = set(allowed)
    bad = [value for value in values if value not in allowed_set]
    if bad:
        raise SystemExit(f"unknown {name}: {', '.join(bad)}")


def build_rows(args: argparse.Namespace) -> tuple[list[BootstrapRow], Path]:
    out_root = Path(args.out_root)
    instances = Path(args.instances) if args.instances else default_instances_path()
    if args.qa_map:
        qa_map_path = Path(args.qa_map)
        qid_to_agent = _load_qid_to_agent(qa_map_path)
    else:
        qa_map_path = instances.parent
        qid_to_agent = _load_qid_to_agent_from_dataset(qa_map_path)
    d6_keep = population_ids(load_instances(instances), args.population)

    rows: list[BootstrapRow] = []
    for backend in args.backends:
        for model_index, model in enumerate(args.models):
            pairs = _paired_qids(out_root, backend, model, args.oracle, args.leak)
            pairs = {q: p for q, p in pairs.items() if p.subdim != "d4_permission" or q in d6_keep}
            for metric_index, metric in enumerate(args.metrics):
                seed = args.random_seed + 1000 * model_index + 37 * metric_index
                point, lo, hi, p_value, n_qids, n_agents = _bootstrap(
                    pairs,
                    qid_to_agent,
                    metric,
                    args.boot,
                    seed,
                )
                rows.append(
                    BootstrapRow(
                        backend=backend,
                        model=model,
                        metric=metric,
                        delta=point,
                        lo=lo,
                        hi=hi,
                        p_value=p_value,
                        n_qids=n_qids,
                        n_agents=n_agents,
                        n_boot=args.boot,
                    )
                )
    return rows, qa_map_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", default=str(OUT_ROOT), help="root containing accuracy_memarena_l_*")
    parser.add_argument("--qa-map", default=None, help="QA JSON carrying qid->ego-agent metadata (default: dataset eval instances)")
    parser.add_argument("--instances", default=None, help="d4_permission.jsonl (default: $MEMARENA_DATASET_DIR)")
    parser.add_argument("--population", default=POPULATION_FACT, choices=(POPULATION_FACT, POPULATION_ALL))
    parser.add_argument("--leak", default=LEAK_LABEL, choices=(LEAK_LABEL, LEAK_LEGACY))
    parser.add_argument("--oracle", default="oracle", choices=("oracle",))
    parser.add_argument("--backends", default=",".join(DEFAULT_BACKENDS))
    parser.add_argument("--models", default=",".join(MODEL_ORDER))
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--boot", type=int, default=2000, help="number of bootstrap resamples")
    parser.add_argument("--random-seed", type=int, default=1002)
    parser.add_argument(
        "--output-tex",
        default=None,
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="optional CSV audit output path",
    )
    args = parser.parse_args(argv)

    args.backends = _parse_csv_list(args.backends)
    args.models = _parse_csv_list(args.models)
    args.metrics = _parse_csv_list(args.metrics)
    _validate_choices("backend", args.backends, BACKEND_TEX)
    _validate_choices("model", args.models, MODEL_ORDER)
    _validate_choices("metric", args.metrics, set(METRIC_SUBDIMS) | {"D6-F1PU", "Trust"})
    if args.boot <= 0:
        raise SystemExit("--boot must be positive")

    rows, qa_map_path = build_rows(args)
    if args.output_tex is None:
        args.output_tex = str(table_path("appendix_main_grid_bootstrap.tex"))
    Path(args.output_tex).parent.mkdir(parents=True, exist_ok=True)
    _write_tex(Path(args.output_tex), rows, args.oracle)
    if args.output_csv:
        _write_csv(Path(args.output_csv), rows)
    print(f"[gen_main_grid_bootstrap] qid->agent source: {qa_map_path}")
    print(f"[gen_main_grid_bootstrap] wrote {args.output_tex}")
    if args.output_csv:
        print(f"[gen_main_grid_bootstrap] wrote {args.output_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
