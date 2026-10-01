"""Cache-off latency model shared by every latency number in the paper.

Every latency we report is for a cold prefill (no prefix-cache reuse). With the
cache off, a dense reader's TTFT depends only on the prompt length N_p, so one
curve per reader is calibrated on the Spark node (``scripts/calibrate_ttft.py``
under ``scripts/run_ttft_calibration.sh``) and each backend x reader cell is
composed from that curve and the backend's measured per-question prompt
lengths and search times:

    TTFT(q)    = T_search(q) + f_r(N_p(q)),             f_r(N) = a + b N + c N^2, c >= 0
    T_total(q) = TTFT(q) + (N_c(q) - 1) * d_r(N_p(q)),  d_r(N) = d0 + d1 N

A cell reports the median over its questions.

Inputs, under ``$MEMARENA_RESULTS_DIR`` (results are not shipped):
  out/ttft_calib/calib_<reader>.jsonl       calibration requests (phase=timed)
  out/latency2_answer/<b>_<r>_s2/...         per-question N_p, N_c, search_time_ms;
      Vanilla / Oracle / RAG for every reader, Memobase / MemSearch from the
      Qwen3-0.6B cache replay (their retrieved context does not depend on the reader)
  out/latency2_ingest/..., out/latency_spark_0_6b_s2/memory_cache/...
      Memobase / MemSearch caches: per-question retrieve_latency_ms, query_timestamp
  out/latency_selection/queries_s2.json     ego of each timed question (cross-check only)

Memobase / MemSearch questions whose cache row has no query_timestamp are
dropped from both: their Memobase profile was built at day 0 and is empty,
which would shorten the prompt.
"""
from __future__ import annotations

import ast
import json
import statistics
from functools import lru_cache
from pathlib import Path

import numpy as np

from memarena.figures.paper_data import RESULTS_ROOT

READERS = ["0_6b", "llama3b", "8b", "32b"]
BACKENDS = ["vanilla", "oracle", "inmem", "memobase", "memsearch"]
REPLAY = ("memobase", "memsearch")


def _paths(root: Path) -> dict:
    out = root / "out"
    return {
        "calib": out / "ttft_calib",
        "answers": out / "latency2_answer",
        "selection": out / "latency_selection" / "queries_s2.json",
        "memsearch": out / "latency2_ingest" / "memsearch_0_6b_s2" / "memcache_memsearch_A_paired_0_6b_s2.jsonl",
        "memobase": out / "latency_spark_0_6b_s2" / "memory_cache" / "memobase" / "0_6b" / "s2"
                    / "memcache_memobase_A_paired_0_6b_s2.jsonl",
    }


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def fit_curve(rows: list[dict]) -> dict:
    n = np.array([r["prompt_tokens"] for r in rows], float) / 1000.0
    y = np.array([r["ttft_ms"] for r in rows], float)
    X = np.vander(n, 3, increasing=True)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    if coef[2] < 0:  # attention cost cannot be negative
        c1, *_ = np.linalg.lstsq(X[:, :2], y, rcond=None)
        coef = np.array([c1[0], c1[1], 0.0])
    yhat = X @ coef
    rel = np.abs(yhat - y) / y
    return {
        "a_ms": float(coef[0]), "b_ms_per_1k": float(coef[1]), "c_ms_per_1k2": float(coef[2]),
        "R2": float(1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
        "rel_resid_p50": float(np.median(rel)), "rel_resid_p95": float(np.percentile(rel, 95)),
        "n_obs": len(rows), "n_p_range": [int(round(n.min() * 1000)), int(round(n.max() * 1000))],
    }


def fit_decode(rows: list[dict]) -> dict:
    rows = [r for r in rows if (r["completion_tokens"] or 0) > 1]
    n = np.array([r["prompt_tokens"] for r in rows], float) / 1000.0
    d = np.array([r["decode_ms"] / (r["completion_tokens"] - 1) for r in rows], float)
    coef, *_ = np.linalg.lstsq(np.vander(n, 2, increasing=True), d, rcond=None)
    return {"d0_ms_per_tok": float(coef[0]), "d1_ms_per_tok_per_1k": float(coef[1])}


def ttft_llm(curve: dict, n_p: float) -> float:
    n = n_p / 1000.0
    return curve["a_ms"] + curve["b_ms_per_1k"] * n + curve["c_ms_per_1k2"] * n * n


def decode_ms_per_tok(dec: dict, n_p: float) -> float:
    return dec["d0_ms_per_tok"] + dec["d1_ms_per_tok_per_1k"] * n_p / 1000.0


def _answer_file(answers: Path, backend: str, reader: str) -> Path | None:
    cands = [c for c in (answers / f"{backend}_{reader}_s2").rglob(f"answer_results_{backend}_{reader}_s2.json")
             if "/runs/" not in str(c)]
    return cands[0] if cands else None


def _questions(p: dict, backend: str, reader: str, caches: dict) -> list[dict]:
    """Per-question (N_p, N_c, T_search) a cell is composed from."""
    f = _answer_file(p["answers"], backend, "0_6b" if backend in REPLAY else reader)
    if f is None:
        return []
    out = []
    for r in json.loads(f.read_text()):
        if r.get("prompt_tokens") is None or (r.get("completion_tokens") or 0) < 1:
            continue
        if backend in REPLAY:
            rows = [caches[b].get(r["question_id"]) for b in REPLAY]
            if any(x is None or x.get("query_timestamp") is None for x in rows):
                continue
            t_search = float(caches[backend][r["question_id"]]["retrieve_latency_ms"])
        else:
            t_search = float(r.get("search_time_ms") or 0.0)
        out.append({"n_p": int(r["prompt_tokens"]), "n_c": int(r["completion_tokens"]), "t_search": t_search})
    return out


def _cross_check(p: dict, reader: str, curve: dict) -> dict:
    """Curve vs the cold requests of the earlier cache-on timing runs.

    Cold there: each ego's first Vanilla question, and Oracle / RAG questions
    (only the short system prompt was shareable). The first two requests of a
    cell are warm-up and dropped. Median measured / predicted per backend.
    """
    items = json.loads(p["selection"].read_text())["items"]
    if isinstance(items, str):
        items = ast.literal_eval(items)
    ego = {it["instance_id"]: it["ego_agent_id"] for it in items}
    out = {}
    for backend in ("vanilla", "oracle", "inmem"):
        f = _answer_file(p["answers"], backend, reader)
        if f is None:
            continue
        seen, ratios = set(), []
        for i, r in enumerate(json.loads(f.read_text())):
            e = ego.get(r["question_id"])
            first = e not in seen
            seen.add(e)
            if i < 2 or r.get("ttft_ms") is None or (backend == "vanilla" and not first):
                continue
            ratios.append(r["ttft_ms"] / ttft_llm(curve, r["prompt_tokens"]))
        if ratios:
            out[backend] = {"n": len(ratios), "measured_over_predicted_p50": statistics.median(ratios)}
    return out


@lru_cache(maxsize=None)
def latency_model(root: Path = RESULTS_ROOT) -> dict:
    """Curves, decode fits and composed cells (keyed ``<backend>_<reader>``)."""
    p = _paths(Path(root))
    caches = {b: {r["instance_id"]: r for r in _jsonl(p[b]) if r.get("error") is None} for b in REPLAY}
    curves, decode, cells, checks = {}, {}, {}, {}
    for reader in READERS:
        f = p["calib"] / f"calib_{reader}.jsonl"
        rows = [r for r in _jsonl(f) if r["phase"] == "timed" and r["ttft_ms"] is not None] if f.exists() else []
        if len(rows) < 10:
            continue
        cv, dv = fit_curve(rows), fit_decode(rows)
        curves[reader], decode[reader] = cv, dv
        if p["selection"].exists():
            checks[reader] = _cross_check(p, reader, cv)
        for backend in BACKENDS:
            qs = _questions(p, backend, reader, caches)
            if not qs:
                continue
            llm = [ttft_llm(cv, q["n_p"]) for q in qs]
            ttft = [q["t_search"] + t for q, t in zip(qs, llm)]
            total = [t + (q["n_c"] - 1) * decode_ms_per_tok(dv, q["n_p"]) for q, t in zip(qs, ttft)]
            cells[f"{backend}_{reader}"] = {
                "n_questions": len(qs),
                "prompt_source_reader": "0_6b" if backend in REPLAY else reader,
                "n_p_p50": statistics.median(q["n_p"] for q in qs),
                "n_c_p50": statistics.median(q["n_c"] for q in qs),
                "t_search_p50_ms": statistics.median(q["t_search"] for q in qs),
                "ttft_llm_p50_ms": statistics.median(llm),
                "ttft_p50_ms": statistics.median(ttft),
                "decode_tok_per_s_p50": statistics.median(1000.0 / decode_ms_per_tok(dv, q["n_p"]) for q in qs),
                "total_p50_ms": statistics.median(total),
            }
    return {"curves": curves, "decode": decode, "cells": cells, "cross_check_old_runs": checks}
