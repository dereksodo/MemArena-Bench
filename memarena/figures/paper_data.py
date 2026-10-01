#!/usr/bin/env python3
"""Shared data loader for all paper figure/table generators.

Every generator script in this repo (`paper/gen_tables.py`, root-level
`analyze_*.py`, `gen_*.py`) ultimately needs the same thing: given a run
directory, enumerate every judged cell and return per-dimension accuracies.
This module is that common layer.

## Directory layout (as of 2026-04-18)

The canonical run directory is ``MASim/runs/l_20260408_111046/``, with the
following sub-structure that this loader understands:

    eval_results/           # seed s1  (original single-seed pass)
      vanilla/evaluation_results_vanilla_{model}_4omini.json
      inmem/evaluation_results_rag_{model}_4omini.json
      oracle/evaluation_results_oracle_{model}_4omini.json
      memory_cache/memory_cache/
        evaluation_results_memcache_{backend}_A_paired_{model_raw}.json
    eval_results_s2/        # seeds s2/s3/s4 (stochastic re-runs @ T=0.3)
    eval_results_s3/
    eval_results_s4/
        # same subdirs; note baselines cover only {0_6b, 7b, 8b, 32b}
        # (Llama-3.2-3B was not re-answered for stochastic baselines)

``{model}`` is one of ``0_6b | 3b | 7b | 8b | 32b | llama3b`` for baselines.
``{model_raw}`` for structured files follows Memobase/MemOS naming
(``qwen3_0_6b``, ``qwen3_32b_awq``, ``llama3_2_3b``, ``qwen3_0_6b_vanilla`` for
the Mem0 partial cache); this module normalises all of these to the baseline
``{model}`` vocabulary so downstream code can join them.

## Dimension remap

The simulator emits eval dimensions ``d1..d11``. The paper reports the six
consolidated dimensions D1..D6 (plus a legacy D7 that is no longer shown in
the main table). The mapping lives in ``NEW_FROM_OLD``.

## Why a single accuracy number per cell is not enough

Downstream generators need three different slices of the same data:

* **Category-level** (Rec / Rea / Conf): the main results table
  (``tab:results-L``) averages dimensions inside each category.
* **Per-dimension** (D1..D7): the appendix tables (``tab:results-L-full``
  and ``tab:results-L-subdim``) show each dimension individually.
* **Per-sub-dimension** (d1..d11): the appendix sub-dim table and D6 analysis
  need access to raw eval dims without aggregation.

``CellResult`` exposes all three; ``aggregate_seeds`` turns a list of
``CellResult`` (one per seed) into mean ± std.

The loader is deliberately tolerant: cells with missing files are represented
as ``None`` in the returned grid so that each downstream script can decide
whether to print a dash, fall back to s1, or skip the row.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Constants shared across all paper generators
# ---------------------------------------------------------------------------

# Adjusted for MemArena layout: file lives at memarena/figures/paper_data.py,
# so repo root is three parents up (vs two in the origin MemArena/paper/ layout).
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Root that the ``json_path`` column of ``experiments_index.csv`` is relative
# to. Result files are not shipped with the repo; point ``MEMARENA_RESULTS_DIR``
# at the directory that holds the ``out/`` tree to regenerate paper artifacts.
RESULTS_ROOT = Path(os.getenv("MEMARENA_RESULTS_DIR") or PROJECT_ROOT)

# Default run directory. Override via the ``MEMARENA_RUN_DIR`` env var so
# external users can render figures against their own MASim runs without
# editing this file.
def _resolve_default_run() -> Path:
    override = os.getenv("MEMARENA_RUN_DIR")
    if override:
        p = Path(override)
        return p if p.is_absolute() else (PROJECT_ROOT / p)
    return PROJECT_ROOT / "MASim" / "runs" / "l_20260408_111046"

DEFAULT_RUN = _resolve_default_run()

# Paper dimensions D1..D6 and the simulator eval dimensions they aggregate.
# A legacy D7 exception bucket is kept only so older helper scripts can still
# parse archived outputs, but it is not part of the active paper mapping.
NEW_FROM_OLD: dict[str, list[str]] = {
    "d1_cloze":         ["d5_cloze"],                                 # Recall: cloze fidelity
    "d2_metadata":      ["d6_metadata"],                              # Recall: metadata completeness
    "d3_factual_qa":    ["d7_qa", "d8_temporal", "d10_counterfactual"],  # Reasoning: factual QA (3 sub-types)
    "d4_cross_session": ["d1_conflict", "d2_anaphora"],               # Reasoning: cross-session reasoning
    "d5_abstention":    ["d3_confabulation"],                         # Trust: balanced abstention / true-claim foil scoring
    "d6_permission":    ["d4_permission"],                            # Trust: permission-aware access
    "d7_exception":     ["d11_exception"],                            # Trust: legacy, not in main table
}
DIM_KEYS_PAPER: list[str] = list(NEW_FROM_OLD.keys())  # D1..D7 order
DIM_SHORT: list[str] = ["D1", "D2", "D3", "D4", "D5", "D6", "D7"]

# Category indices into DIM_KEYS_PAPER.
CAT_RECALL    = [0, 1]      # D1 Cloze + D2 Metadata
CAT_REASONING = [2, 3]      # D3 Factual QA + D4 Cross-Session
CAT_TRUST     = [4]         # D5 Abstention only (D6 analysed separately in §core-findings)

# Model vocabulary. Keys are the canonical baseline identifier; values are the
# LaTeX string used in table headers.
MODEL_TEX: dict[str, str] = {
    "0_6b":    "Qwen3-0.6B",
    "llama3b": "Llama-3.2-3B",
    "7b":      "Mistral-7B",
    "8b":      "Qwen3-8B",
    "32b":     "Qwen3-32B",
    "3b":      "Phi-3.5-mini",  # legacy; kept for parsing old s1 filenames
}
# Canonical ordering when rendering model columns in the main table.
MODEL_ORDER: list[str] = ["0_6b", "llama3b", "7b", "8b", "32b"]

# Backend vocabulary. Directory-level keys on the left, paper-facing labels on
# the right.  ``memory_cache`` is the simulator's umbrella directory; we flatten
# it into the three structured-memory backends (memobase / memos / mem0).
BACKEND_TEX: dict[str, str] = {
    # Both Vanilla cells get the same prompt: the TEXT_SESSIONS path ignores the
    # adapter's vanilla.max_tokens (512 in pipeline_vanilla512.yaml) and caps
    # the context at 8,192 x 3 chars (AnswerEngine._build_text_prompt).
    "vanilla":      "Vanilla",       # main table (pipeline_vanilla512.yaml; 512 is a no-op)
    "vanilla_full": "Vanilla (full)",  # appendix (pipeline.yaml); prompts identical to "vanilla"
    "oracle":       "Oracle",
    "rag":          "RAG",
    "memobase":     "Memobase",
    "memsearch":    "MemSearch",      # main-table replacement for MemOS
    "memos":        "MemOS",          # appendix ablation
    "mem0":         "Mem0",
    "hipporag":     "HippoRAG",
}
BACKEND_ORDER: list[str] = [
    "vanilla", "oracle", "rag", "memobase", "memsearch",
    "vanilla_full", "memos", "mem0", "hipporag",
]

# Seeds we expect to find on disk.  s1 came first (single-seed pass) but was
# run with an older answering-pipeline snapshot (pre-Mistral chat-template fix,
# possibly other prompt drift), so s1 baseline scores diverge systematically
# from s2/s3/s4 by ~20pp on some cells.  The paper therefore reports mean±std
# over the three *stochastic* seeds only; s1 is retained on disk for the
# cross-judge appendix (where it supplies both _qwen3 and _4omini files) but
# does NOT enter any main-text aggregate.
#
# ``SEEDS_ALL`` lists every seed the loader can find; ``SEEDS`` is the default
# set that downstream aggregators pull from.  Pass an explicit ``seeds=``
# argument to ``aggregate_seeds`` to override.
#
# External users who generated their own trials (say "s1", "s2", "s3") can
# override the default for figure aggregation via the ``MEMARENA_SEEDS`` env
# var, e.g. ``MEMARENA_SEEDS=s1,s2,s3 python scripts/reproduce_figures.py --all``.
# The paper default stays ``s2,s3,s4`` so unmodified reproductions still match
# the published numbers (s1 was an earlier single-seed pass with different
# prompt templates and is excluded from the stochastic aggregate).
def _parse_seed_env(env_val: str | None, default: list[str]) -> list[str]:
    if not env_val:
        return default
    parts = [s.strip() for s in env_val.split(",") if s.strip()]
    return parts or default

SEEDS_ALL: list[str] = _parse_seed_env(
    os.getenv("MEMARENA_SEEDS_ALL"), ["s1", "s2", "s3", "s4"]
)
SEEDS: list[str] = _parse_seed_env(os.getenv("MEMARENA_SEEDS"), ["s2", "s3", "s4"])


# Directory names a trial may live under. For s1 this returns BOTH
# ``eval_results_s1`` (preferred, matches external-user layout) and the
# legacy MemArena ``eval_results`` (bare, no suffix). Loader tries each in
# order and stops at the first existing directory.
def _seed_dir_candidates(seed: str) -> list[str]:
    if seed == "s1":
        return [f"eval_results_{seed}", "eval_results"]
    return [f"eval_results_{seed}"]


# ---------------------------------------------------------------------------
# Filename parsers
# ---------------------------------------------------------------------------

# Canonicalises the model fragment that appears inside a structured memcache
# filename (``qwen3_0_6b``, ``qwen3_8b_awq``, ...) into the short baseline
# identifier used in MODEL_ORDER. Returns ``None`` if the fragment is not a
# model we track (e.g. ``gpt41mini`` under the Config-B closed-API path).
def _normalise_memcache_model(raw: str) -> str | None:
    # Order matters: strip the longest variant suffix first.
    mapping = [
        ("qwen3_0_6b_vanilla", "0_6b"),  # Mem0 partial-cache flavour
        ("qwen3_0_6b",         "0_6b"),
        ("qwen3_8b_awq",       "8b"),
        ("qwen3_8b",           "8b"),
        ("qwen3_32b_awq",      "32b"),
        ("qwen3_32b",          "32b"),
        ("llama3_2_3b",        "llama3b"),
        ("mistral_7b",         "7b"),
    ]
    for needle, canonical in mapping:
        if raw == needle or raw.startswith(needle):
            return canonical
    return None


# Regex for baseline judge files. Two filename conventions are accepted:
#   1. paper canonical: ``evaluation_results_{backend}_{model}_4omini.json``
#   2. external / smoke: ``evaluation_results_{backend}_{model}.json`` (no
#      judge-tag suffix, since fresh users don't pin to gpt-4o-mini up front)
# The model fragment may carry an ``_awq`` suffix on quantized weights
# (``8b_awq``, ``32b_awq``, ``7b_awq``); ``_canonicalise_model`` strips it.
_BASELINE_RE = re.compile(
    r"^evaluation_results_(?P<backend>vanilla|rag|oracle)_"
    r"(?P<model>0_6b|32b(?:_awq)?|8b(?:_awq)?|7b(?:_awq)?|3b|llama3b)"
    r"(?:_4omini)?\.json$"
)


def _canonicalise_model(raw: str) -> str:
    """Strip the ``_awq`` quantization suffix so the AWQ slot maps to the
    same canonical key (e.g. ``8b_awq`` → ``8b``) as the bf16/fp16 slot."""
    return raw[:-len("_awq")] if raw.endswith("_awq") else raw

# Regex for structured memcache judge files. Note that these do **not** carry a
# ``_4omini`` suffix in the current snapshot; judge-vs-self is encoded in the
# ``summary`` field inside the JSON.
_MEMCACHE_RE = re.compile(
    r"^evaluation_results_memcache_(?P<backend>memobase|memos|mem0|hipporag)_"
    r"(?P<config>A_paired|B_remote)_(?P<model_raw>.+?)\.json$"
)


def policy_compliant_from_armB(record: dict) -> bool:
    """Derive a boolean ``policy_compliant`` from a D4 record's Arm-B fields.

    The legacy ``policy_compliant`` boolean was retired when D4 moved to the
    Arm-B 3-way LLM rubric. Figure scripts that used ``r["policy_compliant"]``
    can call this shim instead so they don't need to learn the new mapping.

    Mapping (mirrors ``eval/src/scoring._armB_correctness``):
      expected_answer_mode == "disclose"           → COMPLY  ⇒ True
      expected_answer_mode in {"deny","abstain"}  → REFUSAL ⇒ True
      anything else                                ⇒ False
    """
    if "policy_category" not in record:
        # Defensive: treat older non-D4 records as non-applicable.
        return False
    cat = str(record.get("policy_category") or "")
    mode = str(record.get("expected_answer_mode") or "").lower()
    if mode == "disclose":
        return cat == "COMPLY"
    if mode in ("deny", "abstain"):
        return cat == "REFUSAL"
    return False


def _seed_from_dir(seed_dir: str) -> str:
    # ``eval_results`` → s1 (historical); ``eval_results_s2`` → s2 etc.
    return "s1" if seed_dir == "eval_results" else seed_dir.split("_")[-1]


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------

@dataclass
class CellResult:
    """One (seed, backend, model) evaluation.

    Attributes
    ----------
    seed, backend, model
        Identifiers; ``backend`` uses the ``BACKEND_TEX`` keyset, not the
        on-disk directory name (``rag`` rather than ``inmem``).
    per_sub_dim
        ``{sub_dim_id: (n_correct, n_total)}`` at the finest granularity the
        simulator reports (d1..d11 + any new schema variants that show up in
        the JSON).  Downstream aggregations (`accuracy_by_paper_dim`, etc.)
        derive from this single dict to avoid double-counting.
    source_path
        Absolute path to the loaded JSON. Useful for debugging a cell that
        looks off.
    """

    seed: str
    backend: str
    model: str
    per_sub_dim: dict[str, tuple[int, int]]
    source_path: Path

    # --- derived quantities -------------------------------------------------

    def total(self) -> tuple[int, int]:
        c = sum(x[0] for x in self.per_sub_dim.values())
        n = sum(x[1] for x in self.per_sub_dim.values())
        return c, n

    def accuracy(self) -> float | None:
        c, n = self.total()
        return c / n if n else None

    def accuracy_by_paper_dim(self) -> dict[str, float | None]:
        """Return ``{'d1_cloze': acc, ..., 'd7_exception': acc}``.

        Uses a **weighted** average across sub-dims (sum-of-correct over
        sum-of-total), which matches how the paper's Table 3 columns are
        computed: this treats every question instance equally, rather than
        giving equal weight to each sub-dim regardless of sample size.
        """
        out: dict[str, float | None] = {}
        for paper_dim, sub_dims in NEW_FROM_OLD.items():
            correct = sum(self.per_sub_dim.get(sd, (0, 0))[0] for sd in sub_dims)
            total   = sum(self.per_sub_dim.get(sd, (0, 0))[1] for sd in sub_dims)
            out[paper_dim] = correct / total if total else None
        return out

    def accuracy_by_sub_dim(self) -> dict[str, float | None]:
        return {sd: (c / n if n else None) for sd, (c, n) in self.per_sub_dim.items()}


# ---------------------------------------------------------------------------
# Low-level JSON reader
# ---------------------------------------------------------------------------

# Map from simulator ``question_id`` prefix (``d1``, ``d10``) to the canonical
# sub-dimension identifier used by NEW_FROM_OLD.  Structured-memcache JSON
# dumps do NOT set a ``dimension`` field, so we must derive it from the qid.
# Keep this in lock-step with ``NEW_FROM_OLD``.
_QID_PREFIX_TO_SUB_DIM: dict[str, str] = {
    "d1":  "d1_conflict",
    "d2":  "d2_anaphora",
    "d3":  "d3_confabulation",
    "d4":  "d4_permission",
    "d5":  "d5_cloze",
    "d6":  "d6_metadata",
    "d7":  "d7_qa",
    "d8":  "d8_temporal",
    "d9":  "d9_negation",
    "d10": "d10_counterfactual",
    "d11": "d11_exception",
}


def _sub_dim_from_qid(qid: str) -> str | None:
    """``'d10_a1b2c3' → 'd10_counterfactual'``. Returns None on malformed input."""
    if not qid:
        return None
    prefix = qid.split("_", 1)[0]
    return _QID_PREFIX_TO_SUB_DIM.get(prefix)


def _load_cell_json(path: Path) -> dict[str, tuple[int, int]]:
    """Parse a single evaluation_results JSON into {sub_dim: (correct, total)}.

    The scoring pipeline uses two different correctness fields depending on the
    schema version:

    * Early s1 dumps store ``judge_correct`` (bool) when ``scoring_method ==
      "judge"`` and ``correct`` otherwise (deterministic D6).
    * s2/s3/s4 dumps just use ``correct`` uniformly; ``judge_correct`` is
      absent.

    We honour both. For rows the scorer flagged as
    ``answer_scoring_skipped`` (LLM_ERROR, TEXT_SESSIONS_ERROR, etc.) we drop
    the row entirely so an infrastructure failure does not look like a wrong
    answer.  The sub-dimension identifier is pulled from ``dimension`` if
    present (baseline files), else derived from the ``question_id`` prefix
    (structured-memcache files); see :func:`_sub_dim_from_qid`.
    """
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    details = data.get("details", [])
    out: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for d in details:
        # Filter rows the judge could not score. ``answer_scored == False`` is
        # the canonical flag; legacy dumps mark ``raw_response`` with
        # ``LLM_ERROR`` or ``TEXT_SESSIONS_ERROR`` strings instead.
        if d.get("answer_scored") is False:
            continue
        raw = d.get("raw_response") or ""
        if isinstance(raw, str) and (raw.startswith("LLM_ERROR") or raw.startswith("TEXT_SESSIONS_ERROR")):
            continue

        # Extract sub-dim identifier. Baseline dumps carry a canonical
        # ``dimension`` field; structured-memcache dumps omit it and we have
        # to reconstruct from the qid prefix (``d10_...`` →
        # ``d10_counterfactual``, etc.).
        sub = d.get("dimension")
        if not sub:
            sub = _sub_dim_from_qid(d.get("question_id", ""))
        if not sub:
            continue

        # Resolve correctness across both schema versions.
        if d.get("scoring_method") == "judge":
            val = d.get("judge_correct")
            if val is None:
                val = d.get("correct", False)
        else:
            val = d.get("correct", False)

        bucket = out[sub]
        bucket[1] += 1
        bucket[0] += int(bool(val))
    return {k: (v[0], v[1]) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# experiments_index.csv loader
# ---------------------------------------------------------------------------

# CSV "backend" tokens map to paper_data backend keys. The CSV uses "inmem"
# (the on-disk system label) where the paper labels the same cell as "rag".
_CSV_BACKEND_TO_PAPER: dict[str, str] = {
    "oracle":               "oracle",
    "vanilla":              "vanilla",          # main table; vanilla.max_tokens=512 is a no-op
    "vanilla_full":         "vanilla_full",     # appendix; same prompts as "vanilla"
    "inmem":                "rag",
    "memobase":             "memobase",
    "memsearch":            "memsearch",        # main-table replacement for memos
    "memos":                "memos",            # appendix ablation
    # ablation backends (A4/A7/A8 rebuttal)
    "oracle_gated":         "oracle_gated",     # A7: retrieval-time gating
    "memobase_writer32":    "memobase_writer32", # A4: writer >> reader
    "temporal_window_1d":   "temporal_1d",      # A8: temporal sweep
    "temporal_window_3d":   "temporal_3d",
    "temporal_window_7d":   "temporal_7d",
    "temporal_window_15d":  "temporal_15d",
    "temporal_window_alld": "temporal_all",
}

DEFAULT_INDEX_CSV: Path = PROJECT_ROOT / "experiments_index.csv"


# ---------------------------------------------------------------------------
# Grid-runner layout (scripts/test_cell.sh + judge_cell.sh)
# ---------------------------------------------------------------------------
#
#   <runs>/<backend>/<model>/<seed>/<system>/evaluation_results_<ns>_judge_<file_tag>.json
#
# selected with ``MEMARENA_RUNS_DIR=<runs>`` (or ``runs_dir=``); the judge with
# ``MEMARENA_JUDGE=<tag>`` (or ``judge=``), default the primary judge
# (memarena.judges.PRIMARY_JUDGE, gpt-4o-mini). Without MEMARENA_RUNS_DIR the
# experiments_index.csv layout of the submitted paper is read, as before.

# runner backend -> (paper backend key, system directory, namespace prefix)
RUNS_BACKENDS: dict[str, tuple[str, str, str]] = {
    "vanilla":   ("vanilla", "vanilla", "vanilla"),
    "rag":       ("rag", "inmem", "inmem"),
    "memobase":  ("memobase", "memory_cache", "memobase"),
    "memsearch": ("memsearch", "memory_cache", "memsearch"),
    "oracle":    ("oracle", "oracle", "oracle"),
}


def runs_dir_from_env() -> Path | None:
    value = os.getenv("MEMARENA_RUNS_DIR")
    if not value:
        return None
    p = Path(value).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def selected_judge(judge: str | None = None) -> str:
    from memarena.judges import PRIMARY_JUDGE, judge_spec

    tag = judge or os.getenv("MEMARENA_JUDGE") or PRIMARY_JUDGE
    judge_spec(tag)  # validates
    return tag


def runs_eval_path(runs_dir: Path, backend: str, model: str, seed: str, judge: str) -> Path:
    """Judged-cell file of the runner layout for a runner backend (``rag``, not ``inmem``)."""
    from memarena.judges import eval_filename

    _paper, system, prefix = RUNS_BACKENDS[backend]
    return Path(runs_dir) / backend / model / seed / system / eval_filename(f"{prefix}_{model}_{seed}", judge)


def ablation_runs_dir() -> Path:
    """Grid-runner tree holding the ablation cells (``<backend>-<tag>/`` next to
    the main grid): ``MEMARENA_ABLATION_RUNS_DIR``, else ``MEMARENA_RUNS_DIR``,
    else ``$MEMARENA_RESULTS_DIR/out/runs_final``."""
    value = os.getenv("MEMARENA_ABLATION_RUNS_DIR")
    if value:
        p = Path(value).expanduser()
        return p if p.is_absolute() else (PROJECT_ROOT / p)
    return runs_dir_from_env() or (RESULTS_ROOT / "out" / "runs_final")


def ablation_eval_path(cell: str, system: str, prefix: str, model: str, seed: str, judge: str) -> Path:
    """Judged-cell file of an ablation cell, e.g. ``cell="oracle-access"``,
    ``system="oracle"``, ``prefix="oracle_access"``."""
    from memarena.judges import eval_filename

    return ablation_runs_dir() / cell / model / seed / system / eval_filename(f"{prefix}_{model}_{seed}", judge)


def _runs_cell_done(cell_dir: Path, judge: str) -> bool:
    """False while this judge is still working on the cell (progress.json says so)."""
    from memarena.judges import judge_state

    progress = cell_dir / "progress.json"
    if not progress.exists():
        return True  # copied result trees may drop progress.json
    try:
        state = judge_state(json.loads(progress.read_text()), judge)
    except (OSError, json.JSONDecodeError):
        return True
    return state.get("status") in (None, "done")


def load_runs_cells(
    runs_dir: Path,
    judge: str | None = None,
) -> dict[tuple[str, str, str], CellResult]:
    """Every judged main-grid cell of a runner tree, keyed by ``(seed, backend, model)``.

    Cells whose judge has not finished (``progress.json`` judges.<tag>.status
    other than ``done``) are left out like missing ones.
    """
    tag = selected_judge(judge)
    grid: dict[tuple[str, str, str], CellResult] = {}
    for backend, (paper_backend, _system, _prefix) in RUNS_BACKENDS.items():
        for model in MODEL_ORDER:
            for seed in SEEDS_ALL:
                path = runs_eval_path(runs_dir, backend, model, seed, tag)
                if not path.exists() or not _runs_cell_done(path.parent.parent, tag):
                    continue
                per_sub_dim = _load_cell_json(path)
                if not per_sub_dim:
                    continue
                grid[(seed, paper_backend, model)] = CellResult(
                    seed=seed, backend=paper_backend, model=model,
                    per_sub_dim=per_sub_dim, source_path=path,
                )
    return grid


def results_source_label() -> str:
    """One line naming where load_all_cells() reads from (for logs)."""
    runs = runs_dir_from_env()
    if runs is not None:
        return f"runner layout {runs} (judge {selected_judge()})"
    return f"experiments_index.csv layout {os.getenv('MEMARENA_INDEX_CSV') or DEFAULT_INDEX_CSV}"


def load_all_cells(
    run_dir: Path = DEFAULT_RUN,
    index_csv: Path | None = None,
    include_ablation: bool = True,
    runs_dir: Path | None = None,
    judge: str | None = None,
) -> dict[tuple[str, str, str], CellResult]:
    """Return every judged cell, keyed by ``(seed, backend, model)``.

    With ``runs_dir=`` or ``MEMARENA_RUNS_DIR`` set, the cells come from the
    grid-runner tree for judge ``judge`` / ``MEMARENA_JUDGE`` (default the
    primary judge, gpt-4o-mini); see :func:`load_runs_cells`. That tree holds the
    main grid only, so ``include_ablation`` and ``index_csv`` do not apply.
    Otherwise:

    Path resolution comes from ``experiments_index.csv`` (override via
    ``index_csv=`` or the ``MEMARENA_INDEX_CSV`` env var). Both ``main_5x5x3``
    and ``ablation`` rows are consumed (the ablation block carries
    ``vanilla_full`` and ``memos``, which the main 2026-05 paper revision
    moved out of the headline table); pass ``include_ablation=False`` to
    restrict the grid to main-table backends only. ``latency`` rows are
    always skipped — those are answer-only timing dumps, not judge eval
    results. Rows whose ``json_path`` is ``"none"`` or missing on disk are
    dropped silently so in-flight cells just don't appear in the grid.

    The legacy ``run_dir`` parameter is kept for backwards compatibility but
    is no longer consulted: the CSV records absolute layout.
    """
    del run_dir  # legacy arg — layout now comes from the index CSV
    runs = runs_dir if runs_dir is not None else runs_dir_from_env()
    if runs is not None:
        return load_runs_cells(Path(runs), judge)

    csv_path = index_csv or Path(os.getenv("MEMARENA_INDEX_CSV") or DEFAULT_INDEX_CSV)
    if not csv_path.exists():
        return {}

    import csv as _csv
    seeds_allowed = set(SEEDS_ALL)
    accept_tables = {"main_5x5x3", "ablation"} if include_ablation else {"main_5x5x3"}
    grid: dict[tuple[str, str, str], CellResult] = {}
    with csv_path.open(newline="") as fh:
        for row in _csv.DictReader(fh):
            if row.get("table") not in accept_tables:
                continue
            json_rel = (row.get("json_path") or "").strip()
            if not json_rel or json_rel == "none":
                continue
            seed = (row.get("trial") or "").strip()
            if seed not in seeds_allowed:
                continue
            backend_csv = (row.get("backend") or "").strip()
            backend = _CSV_BACKEND_TO_PAPER.get(backend_csv)
            if backend is None:
                continue
            model = _canonicalise_model((row.get("model_tag") or "").strip())
            if model not in MODEL_TEX:
                continue

            json_path = RESULTS_ROOT / json_rel
            if not json_path.exists():
                continue
            per_sub_dim = _load_cell_json(json_path)
            if not per_sub_dim:
                continue
            grid[(seed, backend, model)] = CellResult(
                seed=seed, backend=backend, model=model,
                per_sub_dim=per_sub_dim, source_path=json_path,
            )
    return grid


# ---------------------------------------------------------------------------
# Seed aggregation
# ---------------------------------------------------------------------------

@dataclass
class Stat:
    """Mean ± (sample) standard deviation across seeds, plus book-keeping.

    ``n`` is the number of seeds that actually contributed. Tables should
    render a dash when ``n == 0`` and fall back to point estimate (no ±) when
    ``n == 1``.
    """

    mean: float
    std: float
    n: int

    @classmethod
    def from_values(cls, values: Iterable[float | None]) -> "Stat":
        xs = [float(v) for v in values if v is not None and not math.isnan(float(v))]
        if not xs:
            return cls(mean=float("nan"), std=float("nan"), n=0)
        mean = sum(xs) / len(xs)
        if len(xs) < 2:
            return cls(mean=mean, std=0.0, n=len(xs))
        var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
        return cls(mean=mean, std=math.sqrt(var), n=len(xs))


def aggregate_seeds(
    grid: dict[tuple[str, str, str], CellResult],
    backend: str, model: str,
    dim_selector,  # callable(CellResult) -> float | None
    seeds: Iterable[str] = SEEDS,
) -> Stat:
    """Aggregate one scalar across the requested seeds.

    Parameters
    ----------
    dim_selector : function
        Given a ``CellResult``, return the scalar to aggregate (accuracy for
        one paper dim, a category average, or anything else). Return
        ``None`` to say "this cell does not have this quantity".

    Example::

        rec_stat = aggregate_seeds(
            grid, "memobase", "0_6b",
            lambda c: _category_acc(c, CAT_RECALL),
        )
    """
    values: list[float | None] = []
    for seed in seeds:
        cell = grid.get((seed, backend, model))
        if cell is None:
            continue
        values.append(dim_selector(cell))
    return Stat.from_values(values)


def category_accuracy(cell: CellResult, indices: Iterable[int]) -> float | None:
    """Category accuracy = weighted mean over the D-indices (e.g. Recall=[0,1]).

    Returns ``None`` when every contributing sub-dim is absent.
    """
    total_c = total_n = 0
    for i in indices:
        paper_dim = DIM_KEYS_PAPER[i]
        for sub in NEW_FROM_OLD[paper_dim]:
            c, n = cell.per_sub_dim.get(sub, (0, 0))
            total_c += c
            total_n += n
    return total_c / total_n if total_n else None


# ---------------------------------------------------------------------------
# LaTeX formatting helpers
# ---------------------------------------------------------------------------

def fmt_pct(val: float | None, bold: bool = False, decimals: int = 1) -> str:
    """Render a fraction as a percentage.  ``None`` becomes ``--``."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "--"
    s = f"{val * 100:.{decimals}f}"
    return f"\\textbf{{{s}}}" if bold else s


def fmt_stat(stat: Stat, bold: bool = False, decimals: int = 1) -> str:
    """Render a ``Stat`` as ``mean±std`` (percentage points).

    * ``n == 0``   → ``--``
    * ``n == 1``   → point estimate only, no ``±`` (honest: one sample can't
      have an std deviation); print a footnote hint via caller if needed.
    * ``n >= 2``   → ``mean ± std``.

    Bolding applies to the mean only.
    """
    if stat.n == 0:
        return "--"
    m = f"{stat.mean * 100:.{decimals}f}"
    if stat.n == 1:
        return f"\\textbf{{{m}}}" if bold else m
    s = f"{stat.std * 100:.{decimals}f}"
    body = f"{m}{{\\scriptsize$\\pm${s}}}"
    return f"\\textbf{{{body}}}" if bold else body


# ---------------------------------------------------------------------------
# Diagnostic helper: quick coverage report
# ---------------------------------------------------------------------------

def print_coverage(grid: dict[tuple[str, str, str], CellResult]) -> None:
    """Print one line per (backend, model) listing which seeds are present.

    Handy when debugging a missing cell before running a generator that will
    silently render the cell as a dash.
    """
    by_bm: dict[tuple[str, str], list[str]] = defaultdict(list)
    for (seed, backend, model), _ in grid.items():
        by_bm[(backend, model)].append(seed)
    for (backend, model) in sorted(by_bm):
        seeds = sorted(by_bm[(backend, model)])
        print(f"{backend:10s} {model:10s} seeds={seeds}")


if __name__ == "__main__":
    # Invoke directly to sanity-check the loader on the canonical run.
    grid = load_all_cells()
    print(f"Loaded {len(grid)} cells from {results_source_label()}")
    print_coverage(grid)
