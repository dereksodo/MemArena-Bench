"""Emit the table bodies for the latency appendix and the efficiency table.

All numbers come from the cache-off latency model (memarena.figures.latency_model):

  <artifact root>/tables/appendix_latency_fit_body.tex        per-reader prefill / decode curves
  <artifact root>/tables/appendix_latency_predicted_body.tex  end-to-end T_total per cell
  <artifact root>/tables/efficiency_body.tex                  Vanilla / Oracle / RAG answering cost
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from memarena.figures.gen_main_table import ms  # noqa: E402
from memarena.figures.latency_model import latency_model  # noqa: E402
from memarena.figures.paths import table_path  # noqa: E402

READERS = ["0_6b", "llama3b", "7b", "8b", "32b"]
READER_LABEL = {
    "0_6b":    r"Qwen3-0.6B",
    "llama3b": r"Llama-3.2-3B",
    "7b":      r"Mistral-7B-Inst.\ v0.3",
    "8b":      r"Qwen3-8B",
    "32b":     r"Qwen3-32B-AWQ",
}
BACKENDS = ["vanilla", "oracle", "inmem", "memobase", "memsearch"]
EFFICIENCY_BACKENDS = [("vanilla", "Vanilla"), ("oracle", "Oracle"), ("inmem", "RAG")]


def fit_body(m: dict) -> str:
    lines = []
    for r in READERS:
        cv, dv = m["curves"].get(r), m["decode"].get(r)
        if cv is None:
            lines.append(f"{READER_LABEL[r]} & " + " & ".join(["--"] * 8) + r" \\")
            continue
        lines.append(
            f"{READER_LABEL[r]} & {cv['a_ms']:.1f} & {cv['b_ms_per_1k']:.1f} & {cv['c_ms_per_1k2']:.2f} & "
            f"{cv['R2']:.4f} & {100 * cv['rel_resid_p50']:.1f} & {100 * cv['rel_resid_p95']:.1f} & "
            f"{dv['d0_ms_per_tok']:.1f} & {dv['d1_ms_per_tok_per_1k']:.2f}" + r" \\")
    return "\n".join(lines) + "\n"


def predicted_body(m: dict) -> str:
    lines = []
    for r in READERS:
        if r not in m["curves"]:
            lines.append(f"{READER_LABEL[r]} & " + " & ".join([r"\texttt{n/a}"] * len(BACKENDS)) + r" \\")
            continue
        lines.append(f"{READER_LABEL[r]} & " + " & ".join(
            ms(m["cells"][f"{b}_{r}"]["total_p50_ms"]) for b in BACKENDS) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"$T_{\text{search}}$ (ms, p50) & " + " & ".join(
        ms(m["cells"][f"{b}_0_6b"]["t_search_p50_ms"]) for b in BACKENDS) + r" \\")
    return "\n".join(lines) + "\n"


def efficiency_body(m: dict) -> str:
    lines = []
    readers = [r for r in READERS if r in m["curves"]]
    for i, r in enumerate(readers):
        for j, (b, label) in enumerate(EFFICIENCY_BACKENDS):
            c = m["cells"][f"{b}_{r}"]
            head = READER_LABEL[r] if j == 0 else ""
            lines.append(f"{head} & {label} & {ms(c['ttft_p50_ms'])} & {c['decode_tok_per_s_p50']:.1f} & "
                         f"{c['total_p50_ms'] / 1000:.2f} & {c['n_c_p50']:.0f}" + r" \\")
        if i < len(readers) - 1:
            lines.append(r"\midrule")
    return "\n".join(lines) + "\n"


def main() -> int:
    m = latency_model()
    for name, body in (("appendix_latency_fit_body.tex", fit_body(m)),
                       ("appendix_latency_predicted_body.tex", predicted_body(m)),
                       ("efficiency_body.tex", efficiency_body(m))):
        out = table_path(name)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(body, encoding="utf-8")
        print(f"[gen_latency_appendix_tables] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
