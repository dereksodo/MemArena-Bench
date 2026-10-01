#!/usr/bin/env python3
"""Per-cell serving-cost and ingest-overhead appendix tables.

Emits two LaTeX booktabs tables under ``paper/tables/``:

* ``appendix_serving_costs.tex`` -- per-(reader, backend) cell:
  prompt/completion tokens, TTFT, decode throughput, search time,
  per-query energy, peak temperature.

* ``appendix_ingest_overhead.tex`` -- per-(reader, structured backend)
  ingest wall-time, ingest energy, extraction-batch count, J/ego/day,
  mean power.

Sources (all under ``$MEMARENA_RESULTS_DIR/MASim/runs/l_20260408_111046/spark_results_s1``):

* Per-query timing/tokens:
    - vanilla, oracle, inmem: ``<backend>/<reader>/answer_results_run.json``
    - memobase, memos: ``paperhook/memory_cache/answer_results_paperhook_<backend>_<reader>_s1.json``
      with the special case ``paperhook/memory_cache_8agent/memory_cache/...``
      for ``memobase x qwen3_32b_awq``.
* Per-cell answer-phase energy + peak temperature:
    - vanilla, oracle, inmem: ``hw/hw_agg_<reader>_<backend>.json``
    - memobase, memos: ``hw/hw_agg_<reader>_<backend>_answerB.json``
* Ingest energy and per-day rows:
    - ``hw/hw_agg_<reader>_<backend>_ingest.json`` (sum the 15 ``day_*_ingest`` rows;
      ignore ``day_*_query`` probe rows).
* Cache build wall-time + extraction-batch count:
    - ``memory_cache/memcache_<backend>_<reader>_spark8.summary.json``

Notes on ``timing_reliable``:
    Vanilla / Oracle pipelines instrument the flag and set it to ``True`` for
    every row. The inmem / memobase / memos pipelines never set the flag
    (it stays ``None``). We treat ``None`` as "not flagged as bad" and keep
    those rows; only rows explicitly marked ``False`` are dropped. This is
    flagged in the table caption.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


from memarena.figures.paper_data import RESULTS_ROOT
from memarena.figures.paths import table_path

SPARK_S1 = RESULTS_ROOT / "MASim" / "runs" / "l_20260408_111046" / "spark_results_s1"
HW_DIR = SPARK_S1 / "hw"
CACHE_DIR = SPARK_S1 / "memory_cache"

OUT_SERVING = table_path("appendix_serving_costs.tex")
OUT_INGEST = table_path("appendix_ingest_overhead.tex")

N_DAYS = 15
N_EGOS = 8  # spark probes are 8-ego subsets for s1


# ---------------------------------------------------------------------------
# Reader / backend registry.

READERS = [
    ("qwen3_0.6b",   "Qwen3-0.6B"),
    ("llama3_3b",    "Llama-3.2-3B"),
    ("qwen3_8b",     "Qwen3-8B"),
    ("qwen3_32b_awq", "Qwen3-32B-AWQ"),
]

BACKENDS_SERVING = [
    ("vanilla",  "Vanilla",  r"\texttt{vanilla}"),
    ("inmem",    "RAG",      r"\texttt{inmem}"),
    ("oracle",   "Oracle",   r"\texttt{oracle}"),
    ("memobase", "Memobase", r"\texttt{memobase}"),
]

STRUCTURED = [
    ("memobase", "Memobase", r"\texttt{memobase}"),
]


# ---------------------------------------------------------------------------
# Path helpers.

def answer_results_path(backend: str, reader: str) -> Optional[Path]:
    """Return the canonical per-query answer-results JSON for a cell, or None."""
    if backend in ("vanilla", "oracle", "inmem"):
        p = SPARK_S1 / backend / reader / "answer_results_run.json"
        return p if p.exists() else None
    if backend in ("memobase", "memos"):
        primary = SPARK_S1 / "paperhook" / "memory_cache" / \
            f"answer_results_paperhook_{backend}_{reader}_s1.json"
        if primary.exists():
            return primary
        alt = SPARK_S1 / "paperhook" / "memory_cache_8agent" / "memory_cache" / \
            f"answer_results_paperhook_{backend}_{reader}_s1.json"
        if alt.exists():
            return alt
        return None
    return None


def hw_agg_answer_path(backend: str, reader: str) -> Optional[Path]:
    """Per-cell answer-phase hw_agg path. memobase/memos use the answerB file."""
    if backend in ("vanilla", "oracle", "inmem"):
        p = HW_DIR / f"hw_agg_{reader}_{backend}.json"
    elif backend in ("memobase", "memos"):
        p = HW_DIR / f"hw_agg_{reader}_{backend}_answerB.json"
    else:
        return None
    return p if p.exists() else None


def hw_agg_ingest_path(backend: str, reader: str) -> Optional[Path]:
    p = HW_DIR / f"hw_agg_{reader}_{backend}_ingest.json"
    return p if p.exists() else None


def cache_summary_path(backend: str, reader: str) -> Optional[Path]:
    p = CACHE_DIR / f"memcache_{backend}_{reader}_spark8.summary.json"
    return p if p.exists() else None


# ---------------------------------------------------------------------------
# IO + aggregation.

def _load_json(p: Path) -> Optional[object]:
    try:
        with open(p, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _median(xs: list[float]) -> Optional[float]:
    if not xs:
        return None
    return float(statistics.median(xs))


@dataclass
class ServingRow:
    reader_disp: str
    backend_disp: str
    backend_tex: str
    n: int = 0
    prompt_tok_med: Optional[float] = None
    completion_tok_med: Optional[float] = None
    ttft_ms_med: Optional[float] = None
    decode_tok_s_med: Optional[float] = None
    search_ms_med: Optional[float] = None  # may be None for vanilla/oracle (no search)
    energy_j_med: Optional[float] = None  # mean per-query energy (column header is J/q)
    peak_temp_c: Optional[float] = None
    notes: list[str] = field(default_factory=list)


def compute_serving_row(reader: str, reader_disp: str,
                        backend: str, backend_disp: str, backend_tex: str) -> ServingRow:
    row = ServingRow(reader_disp=reader_disp,
                     backend_disp=backend_disp,
                     backend_tex=backend_tex)

    # Per-query timing/tokens.
    ar_path = answer_results_path(backend, reader)
    if ar_path is None:
        row.notes.append("no answer_results")
    else:
        data = _load_json(ar_path)
        if not isinstance(data, list):
            row.notes.append("answer_results not a list")
        else:
            # Filter: drop only rows explicitly marked timing_reliable False.
            kept = [r for r in data if r.get("timing_reliable") is not False]
            row.n = len(kept)

            pts = [r["prompt_tokens"] for r in kept
                   if isinstance(r.get("prompt_tokens"), (int, float)) and r.get("prompt_tokens") is not None]
            cts = [r["completion_tokens"] for r in kept
                   if isinstance(r.get("completion_tokens"), (int, float)) and r.get("completion_tokens") is not None]
            ttfts = [r["ttft_ms"] for r in kept
                     if isinstance(r.get("ttft_ms"), (int, float)) and r.get("ttft_ms") is not None]
            row.prompt_tok_med = _median(pts)
            row.completion_tok_med = _median(cts)
            row.ttft_ms_med = _median(ttfts)

            # Decode throughput per row: completion * 1000 / (answer_time_ms - ttft_ms).
            tok_s_per_row: list[float] = []
            for r in kept:
                ct = r.get("completion_tokens")
                at = r.get("answer_time_ms")
                tt = r.get("ttft_ms")
                if (ct is None or at is None or tt is None):
                    continue
                denom = at - tt
                if denom is None or denom <= 0 or ct <= 0:
                    continue
                tok_s_per_row.append(ct * 1000.0 / denom)
            row.decode_tok_s_med = _median(tok_s_per_row)

            # Search time: only meaningful for backends that actually search.
            if backend in ("inmem", "memobase", "memos"):
                searches = [r.get("search_time_ms") for r in kept
                            if isinstance(r.get("search_time_ms"), (int, float))]
                searches_nonnull = [s for s in searches if s is not None]
                if searches_nonnull:
                    med_s = _median(searches_nonnull)
                    # If all zeros, we still record 0.0 (interpretation is "not separately timed").
                    row.search_ms_med = med_s
                # else leave None (rendered as n/a)
            # vanilla/oracle: leave None (rendered as em-dash)

    # Per-cell energy + peak temp.
    hw_path = hw_agg_answer_path(backend, reader)
    if hw_path is None:
        row.notes.append("no hw_agg")
    else:
        d = _load_json(hw_path)
        per_cell = (d or {}).get("per_cell") or {}
        # Use mean rather than median for the J/q column: per-query energy on
        # small readers (Q0.6B, Llama-3.2-3B) frequently sits below the GPU
        # power-probe noise floor on individual queries, so the median collapses
        # to 0.0 even though the mean (and total) energy is non-zero. Reporting
        # mean keeps the column comparable across reader scales and matches the
        # 0.56 J/q figure cited in the abstract.
        ej = per_cell.get("mean_energy_j_per_query")
        if isinstance(ej, (int, float)):
            row.energy_j_med = float(ej)
        pt = per_cell.get("peak_temp_c")
        if isinstance(pt, (int, float)):
            row.peak_temp_c = float(pt)

    return row


@dataclass
class IngestRow:
    reader_disp: str
    backend_disp: str
    backend_tex: str
    wall_s: Optional[float] = None       # cache summary elapsed_seconds (preferred)
    energy_kj: Optional[float] = None    # sum of energy_j_net over 15 day_*_ingest rows
    n_extractions: Optional[int] = None  # n_ingest_batches from summary
    j_per_ego_day: Optional[float] = None
    mean_power_w: Optional[float] = None
    notes: list[str] = field(default_factory=list)


def compute_ingest_row(reader: str, reader_disp: str,
                       backend: str, backend_disp: str, backend_tex: str) -> IngestRow:
    row = IngestRow(reader_disp=reader_disp,
                    backend_disp=backend_disp,
                    backend_tex=backend_tex)

    # Cache summary -> wall time + extraction count.
    sp = cache_summary_path(backend, reader)
    if sp is None:
        row.notes.append("no cache summary")
    else:
        d = _load_json(sp) or {}
        es = d.get("elapsed_seconds")
        if isinstance(es, (int, float)):
            row.wall_s = float(es)
        nb = d.get("n_ingest_batches")
        if isinstance(nb, (int, float)):
            row.n_extractions = int(nb)

    # hw_agg ingest -> energy + mean power.
    hp = hw_agg_ingest_path(backend, reader)
    if hp is None:
        row.notes.append("no hw_agg ingest")
    else:
        d = _load_json(hp) or {}
        ingest_rows = [r for r in d.get("per_query", [])
                       if "ingest" in (r.get("instance_id") or "")]
        if ingest_rows:
            # If cache summary lacked wall-time, fall back to summed elapsed_s.
            if row.wall_s is None:
                row.wall_s = sum(float(r.get("elapsed_s", 0.0) or 0.0) for r in ingest_rows)
            # Detect "hardware probe never fired": all rows have n_samples == 0
            # AND total energy is 0. In that case wall-time is real but energy is
            # not measured -- mark energy/power columns n/a rather than 0.0.
            total_samples = sum(int(r.get("n_samples", 0) or 0) for r in ingest_rows)
            total_e = sum(float(r.get("energy_j_net", 0.0) or 0.0) for r in ingest_rows)
            if total_samples == 0 and total_e == 0.0:
                row.notes.append("hw probe absent for ingest")
            else:
                row.energy_kj = total_e / 1000.0
                # Mean power: average of per-day mean_power_w, excluding zeros
                # (idle / skipped slots).
                powers = [float(r.get("mean_power_w") or 0.0) for r in ingest_rows]
                powers_nz = [p for p in powers if p > 0]
                if powers_nz:
                    row.mean_power_w = sum(powers_nz) / len(powers_nz)
                # J per ego per day.
                row.j_per_ego_day = total_e / N_DAYS / N_EGOS

    return row


# ---------------------------------------------------------------------------
# LaTeX rendering helpers.

def _fmt(v: Optional[float], spec: str = ".0f", na: str = "n/a") -> str:
    if v is None:
        return na
    try:
        if isinstance(v, float) and (v != v):  # NaN
            return na
    except Exception:
        return na
    return format(v, spec)


def render_serving_table(rows: list[ServingRow]) -> str:
    """Booktabs LaTeX. Group by reader; multirow on reader column."""
    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Per-cell serving costs on \benchL{} (s1 trial seed, Spark probes, "
                 r"$N{=}250$ unique answer queries for \texttt{vanilla}/\texttt{oracle}/\texttt{inmem} "
                 r"and $N{=}1{,}579$ replayed queries for \texttt{memobase}/\texttt{memos}). "
                 r"Token, latency, and search-time columns are per-row medians across rows whose "
                 r"\texttt{timing\_reliable} flag is not explicitly false; \texttt{vanilla}/\texttt{oracle} "
                 r"populate the flag (all rows true), the structured-memory pipelines do not. "
                 r"Decode throughput is the per-row median of $\textsc{ct}\cdot{}1000/(\textsc{at}{-}\textsc{ttft})$. "
                 r"\textbf{J/q is the cell mean rather than the median}: per-query energy on the smallest "
                 r"readers (Q$0.6$B, Llama-$3.2$-$3$B) sits below the GPU power-probe noise floor on a "
                 r"majority of individual queries, so the median collapses to $0.0$ even though the cumulative "
                 r"energy is non-zero; we report the mean to keep the column comparable across reader scales "
                 r"and to match the $0.56$~J/q figure cited in the abstract / Finding~3. "
                 r"Search time was not separately instrumented in the answer logs (em-dash for non-search backends, "
                 r"\texttt{0.0} reported when the field is present but always zero). "
                 r"Energy and peak temperature come from per-cell hardware aggregates "
                 r"(\texttt{*\_answerB} files for structured-memory cells).}")
    lines.append(r"\label{tab:appendix-serving-costs}")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{@{}llrrrrrrrr@{}}")
    lines.append(r"\toprule")
    lines.append(r"Reader & Backend & $n_q$ & prompt tok & comp.\ tok & "
                 r"TTFT (ms) & decode (tok/s) & search (ms) & J/q & peak $^\circ$C \\")
    lines.append(r"\midrule")

    # Group by reader.
    current_reader = None
    reader_to_rows: dict[str, list[ServingRow]] = {}
    order: list[str] = []
    for r in rows:
        if r.reader_disp not in reader_to_rows:
            reader_to_rows[r.reader_disp] = []
            order.append(r.reader_disp)
        reader_to_rows[r.reader_disp].append(r)

    for ridx, reader_disp in enumerate(order):
        block = reader_to_rows[reader_disp]
        for j, r in enumerate(block):
            if j == 0:
                first = r"\multirow{" + str(len(block)) + r"}{*}{" + reader_disp + r"}"
            else:
                first = ""
            search_str = (r"\textemdash" if r.backend_disp in ("Vanilla", "Oracle")
                          else _fmt(r.search_ms_med, ".1f"))
            lines.append(
                f"{first} & {r.backend_tex} & "
                f"{r.n if r.n else 'n/a'} & "
                f"{_fmt(r.prompt_tok_med, '.0f')} & "
                f"{_fmt(r.completion_tok_med, '.0f')} & "
                f"{_fmt(r.ttft_ms_med, '.1f')} & "
                f"{_fmt(r.decode_tok_s_med, '.1f')} & "
                f"{search_str} & "
                f"{_fmt(r.energy_j_med, '.1f')} & "
                f"{_fmt(r.peak_temp_c, '.0f')} \\\\"
            )
        if ridx != len(order) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def render_ingest_body(rows: list[IngestRow]) -> str:
    """Rows of tab:appendix-ingest-overhead as the paper typesets them (the
    caption and header live in paper_rev0719/tables/appendix_ingest_overhead.tex)."""
    lines = []
    for r in rows:
        lines.append(
            f"{r.reader_disp} & {r.backend_tex} & "
            f"{_fmt(r.wall_s, '.0f')} & "
            f"{_fmt(r.energy_kj, '.2f')} & "
            f"{r.n_extractions if r.n_extractions is not None else 'n/a'} & "
            f"{_fmt(r.j_per_ego_day, '.1f')} & "
            f"{_fmt(r.mean_power_w, '.2f')} \\\\"
        )
    return "\n".join(lines) + "\n"


def render_ingest_table(rows: list[IngestRow]) -> str:
    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Per-cell ingest overhead for the structured-memory backends "
                 r"(\texttt{memobase}, \texttt{memos}) on \benchL{} (s1 trial seed, "
                 r"direct Spark $8$-ego $\times$ $15$-day cache build). Wall time and extraction-batch count "
                 r"come from the cache-build summary; energy is the sum of \texttt{energy\_j\_net} over the "
                 r"$15$ \texttt{day\_*\_ingest} rows of the per-query hardware log. "
                 r"$\mathrm{J\,/\,ego\cdot{}day}=\text{total energy}/15/8$. "
                 r"Mean power averages the per-day \texttt{mean\_power\_w} excluding zero-power slots. "
                 r"Any $50$-ego equivalent is a $\times 50/8$ lower-bound normalization "
                 r"that assumes perfect, isolated database/index scaling.}")
    lines.append(r"\label{tab:appendix-ingest-overhead}")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\begin{tabular}{@{}llrrrrr@{}}")
    lines.append(r"\toprule")
    lines.append(r"Reader & Backend & wall (s) & energy (kJ) & "
                 r"$n_\text{batches}$ & J/ego$\cdot$day & mean power (W) \\")
    lines.append(r"\midrule")

    reader_to_rows: dict[str, list[IngestRow]] = {}
    order: list[str] = []
    for r in rows:
        if r.reader_disp not in reader_to_rows:
            reader_to_rows[r.reader_disp] = []
            order.append(r.reader_disp)
        reader_to_rows[r.reader_disp].append(r)

    for ridx, reader_disp in enumerate(order):
        block = reader_to_rows[reader_disp]
        for j, r in enumerate(block):
            if j == 0:
                first = r"\multirow{" + str(len(block)) + r"}{*}{" + reader_disp + r"}"
            else:
                first = ""
            lines.append(
                f"{first} & {r.backend_tex} & "
                f"{_fmt(r.wall_s, '.0f')} & "
                f"{_fmt(r.energy_kj, '.2f')} & "
                f"{r.n_extractions if r.n_extractions is not None else 'n/a'} & "
                f"{_fmt(r.j_per_ego_day, '.1f')} & "
                f"{_fmt(r.mean_power_w, '.2f')} \\\\"
            )
        if ridx != len(order) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Stdout helpers.

def print_serving_rows(rows: list[ServingRow]) -> None:
    print()
    print("=== Serving costs ===")
    header = (f"{'reader':14s} {'backend':10s} {'n':>5s} "
              f"{'pt':>6s} {'ct':>5s} {'ttft':>8s} {'tok/s':>7s} "
              f"{'srch':>7s} {'J/q':>9s} {'°C':>4s}  notes")
    print(header)
    print("-" * len(header))
    for r in rows:
        srch = ("---" if r.backend_disp in ("Vanilla", "Oracle")
                else _fmt(r.search_ms_med, ".1f"))
        print(f"{r.reader_disp:14s} {r.backend_disp:10s} {r.n:5d} "
              f"{_fmt(r.prompt_tok_med, '.0f'):>6s} "
              f"{_fmt(r.completion_tok_med, '.0f'):>5s} "
              f"{_fmt(r.ttft_ms_med, '.1f'):>8s} "
              f"{_fmt(r.decode_tok_s_med, '.1f'):>7s} "
              f"{srch:>7s} "
              f"{_fmt(r.energy_j_med, '.1f'):>9s} "
              f"{_fmt(r.peak_temp_c, '.0f'):>4s}  "
              f"{', '.join(r.notes)}")


def print_ingest_rows(rows: list[IngestRow]) -> None:
    print()
    print("=== Ingest overhead ===")
    header = (f"{'reader':14s} {'backend':10s} "
              f"{'wall_s':>8s} {'kJ':>8s} {'n_batches':>10s} "
              f"{'J/ego·d':>10s} {'mean_W':>8s}  notes")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r.reader_disp:14s} {r.backend_disp:10s} "
              f"{_fmt(r.wall_s, '.0f'):>8s} "
              f"{_fmt(r.energy_kj, '.2f'):>8s} "
              f"{(str(r.n_extractions) if r.n_extractions is not None else 'n/a'):>10s} "
              f"{_fmt(r.j_per_ego_day, '.1f'):>10s} "
              f"{_fmt(r.mean_power_w, '.2f'):>8s}  "
              f"{', '.join(r.notes)}")


# ---------------------------------------------------------------------------
# Main.

def main() -> None:
    serving_rows: list[ServingRow] = []
    for reader, reader_disp in READERS:
        for backend, be_disp, be_tex in BACKENDS_SERVING:
            row = compute_serving_row(reader, reader_disp, backend, be_disp, be_tex)
            serving_rows.append(row)

    ingest_rows: list[IngestRow] = []
    for reader, reader_disp in READERS:
        for backend, be_disp, be_tex in STRUCTURED:
            row = compute_ingest_row(reader, reader_disp, backend, be_disp, be_tex)
            ingest_rows.append(row)

    print_serving_rows(serving_rows)
    print_ingest_rows(ingest_rows)

    OUT_SERVING.parent.mkdir(parents=True, exist_ok=True)
    OUT_SERVING.write_text(render_serving_table(serving_rows))
    OUT_INGEST.write_text(render_ingest_table(ingest_rows))
    table_path("appendix_ingest_overhead_body.tex").write_text(render_ingest_body(ingest_rows))
    print()
    print(f"Wrote {OUT_SERVING}")
    print(f"Wrote {OUT_INGEST}")


if __name__ == "__main__":
    main()
