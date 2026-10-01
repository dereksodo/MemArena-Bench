#!/usr/bin/env python3
"""Emit experiments_index.csv for the MemArena-L paper experiments.

Two blocks:
- main_5x5x3 : Table 3 (tab:results-L) — 5 backends x 5 readers x 3 seeds.
                Production data lives in:
                  vanilla/oracle/inmem (baselines):
                    out/accuracy_memarena_l_baselines_{reader}/eval_results_{trial}/{backend}/
                      evaluation_results_{backend}_{reader}_{trial}_judge_remote.json
                  memobase/memos (memory_cache):
                    out/accuracy_memarena_l_{reader_slug}/eval_results_{trial}/memory_cache/
                      evaluation_results_{backend}_{reader}_{trial}_judge_remote.json
                Where reader_slug == "3b" for reader=="llama3b", else == reader.
- latency    : Efficiency table (tab:efficiency) — 5 readers x 5 backends, s2 only.
                Production data lives in out/latency_spark_{model_tag}_s2/...

Smoke (test_mode=1) outputs are intentionally NOT consulted: those are
sentinel "I don't know" results and aren't experiment data. A cell with no
real local file becomes "none".
"""
from __future__ import annotations

import csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "out"

# Main-table backends (post-2026-05 paper revision):
#   - vanilla here means vanilla-512: max_tokens=512 generation budget,
#     scored from out/accuracy_memarena_l_vanilla512_<reader>/...
#   - memsearch (markdown + milvus-lite) is the new structured-memory
#     candidate; replaces memos in main-table-position. Memos stays as
#     an ablation backend so appendix tables can still render it.
#   - vanilla_full (the previous default-budget vanilla) and memos are
#     ablation backends; their data is at the legacy paths.
BACKENDS = ["oracle", "vanilla", "inmem", "memobase", "memsearch"]
ABLATION_BACKENDS = ["vanilla_full", "memos"]
MODELS = [
    ("0_6b",    "Qwen/Qwen3-0.6B"),
    ("llama3b", "meta-llama/Llama-3.2-3B-Instruct"),
    ("7b",      "mistralai/Mistral-7B-Instruct-v0.3"),
    ("8b",      "Qwen/Qwen3-8B"),
    ("32b",     "Qwen/Qwen3-32B-AWQ"),
]
TRIALS = [("s2", 1002), ("s3", 1003), ("s4", 1004)]

# DGX-Spark latency layout: memobase results land under memory_cache; the
# RAG cell ("inmem") is run as the BM25 baseline, written under inmem.
LATENCY_DIR = {
    "oracle":    "oracle",
    "vanilla":   "vanilla",
    "inmem":     "inmem",
    "memobase":  "memory_cache",
    "memsearch": "memory_cache",
    "memos":     "memory_cache",
}


def rel(p: Path) -> str:
    return str(p.relative_to(REPO))


def find_main(backend: str, model_tag: str, trial: str) -> str:
    """Locate the per-cell evaluation_results JSON for a main-table cell.

    Layouts per chain runner:
      oracle/inmem  -> out/accuracy_memarena_l_baselines_<reader>/eval_results_<trial>/<backend>/
                       evaluation_results_<backend>_<reader>_<trial>_judge_remote.json
      vanilla       -> out/accuracy_memarena_l_vanilla512_<reader>/eval_results_<trial>/vanilla/
                       evaluation_results_vanilla_<reader>_<trial>_judge_remote.json   (max_tokens=512)
      memobase/memsearch/memos -> out/accuracy_memarena_l_<reader_slug>/eval_results_<trial>/memory_cache/
                                   evaluation_results_<backend>_<reader>_<trial>_judge_remote.json

    The directory slug strips the "llama" prefix for Llama-3.2-3B (writes to
    accuracy_memarena_l_3b/ even though file names use the "llama3b" tag).

    Ablation aliases: backend="vanilla_full" reads the legacy
    accuracy_memarena_l_baselines_<reader>/.../vanilla/... path so we can keep
    the ctx ablation appendix.
    """
    reader_slug = "3b" if model_tag == "llama3b" else model_tag

    if backend == "vanilla":
        cand = (
            OUT
            / f"accuracy_memarena_l_vanilla512_{model_tag}"
            / f"eval_results_{trial}"
            / "vanilla"
            / f"evaluation_results_vanilla_{model_tag}_{trial}_judge_remote.json"
        )
        if cand.exists():
            return rel(cand)
        return "none"

    if backend == "vanilla_full":
        cand = (
            OUT
            / f"accuracy_memarena_l_baselines_{model_tag}"
            / f"eval_results_{trial}"
            / "vanilla"
            / f"evaluation_results_vanilla_{model_tag}_{trial}_judge_remote.json"
        )
        if cand.exists():
            return rel(cand)
        return "none"

    if backend in ("memobase", "memos", "memsearch"):
        cand = (
            OUT
            / f"accuracy_memarena_l_{reader_slug}"
            / f"eval_results_{trial}"
            / "memory_cache"
            / f"evaluation_results_{backend}_{model_tag}_{trial}_judge_remote.json"
        )
        if cand.exists():
            return rel(cand)
        return "none"

    # oracle / inmem (BM25 RAG)
    cand = (
        OUT
        / f"accuracy_memarena_l_baselines_{model_tag}"
        / f"eval_results_{trial}"
        / backend
        / f"evaluation_results_{backend}_{model_tag}_{trial}_judge_remote.json"
    )
    if cand.exists():
        return rel(cand)
    return "none"


def find_latency(backend: str, model_tag: str) -> str:
    """Locate the per-cell answer_results JSON used for latency analysis.

    DGX-Spark runs land in out/latency_spark_{model_tag}_s2/eval_results_s2/{sub}/
    with file name keyed on the public backend label (Memobase under memory_cache,
    RAG under inmem with 'baseline_simplerag' prefix).
    """
    base = OUT / f"latency_spark_{model_tag}_s2" / "eval_results_s2"
    sub = LATENCY_DIR[backend]
    label = "baseline_simplerag" if backend == "inmem" else backend
    cand = base / sub / f"answer_results_{label}_{model_tag}_s2.json"
    if cand.exists():
        return rel(cand)
    return "none"


def main() -> None:
    rows = []
    for backend in BACKENDS:
        for model_tag, model_name in MODELS:
            for trial, seed in TRIALS:
                rows.append({
                    "table": "main_5x5x3",
                    "backend": backend,
                    "model_tag": model_tag,
                    "model_name": model_name,
                    "trial": trial,
                    "seed": seed,
                    "json_path": find_main(backend, model_tag, trial),
                })

    # Ablation block: vanilla_full (default-budget vanilla) and memos
    # (replaced by memsearch in the main table). Same shape as main_5x5x3
    # so the figure code can join these in for appendix tables.
    for backend in ABLATION_BACKENDS:
        for model_tag, model_name in MODELS:
            for trial, seed in TRIALS:
                rows.append({
                    "table": "ablation",
                    "backend": backend,
                    "model_tag": model_tag,
                    "model_name": model_name,
                    "trial": trial,
                    "seed": seed,
                    "json_path": find_main(backend, model_tag, trial),
                })

    for model_tag, model_name in MODELS:
        for backend in BACKENDS:
            rows.append({
                "table": "latency",
                "backend": backend,
                "model_tag": model_tag,
                "model_name": model_name,
                "trial": "s2",
                "seed": 1002,
                "json_path": find_latency(backend, model_tag),
            })

    out_csv = REPO / "experiments_index.csv"
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "table", "backend", "model_tag", "model_name", "trial", "seed", "json_path",
        ])
        writer.writeheader()
        writer.writerows(rows)

    found = sum(1 for r in rows if r["json_path"] != "none")
    print(f"wrote {out_csv} ({len(rows)} rows; {found} with local JSON, {len(rows)-found} none)")


if __name__ == "__main__":
    main()
