"""Generate the body of tab:dataset-summary from the released MemArena-L files.

Reads ``corpus_sessions.jsonl.gz``, ``ego_session_map.json`` and
``eval_instances/`` from ``$MEMARENA_DATASET_DIR`` (default ``data/benchmark``).
Tokens are counted over each turn's ``text`` with the ``Qwen/Qwen3-8B``
tokenizer (no special tokens); words are whitespace-separated.

    python -m memarena.figures.gen_dataset_summary
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import statistics as st
from pathlib import Path
from typing import List

from memarena.figures.main_grid import PAPER_DIMS
from memarena.figures.paper_data import PROJECT_ROOT
from memarena.figures.paths import table_path

N_DAYS = 15
TOKENIZER = "Qwen/Qwen3-8B"
EVAL_SUBDIMS = sorted({s for subs in PAPER_DIMS.values() for s in subs} | {"d4_permission"})


def _n(x: float) -> str:
    return f"{x:,.0f}"


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate the dataset summary table body.")
    ap.add_argument("--dataset-dir", type=Path, default=Path(os.getenv("MEMARENA_DATASET_DIR") or PROJECT_ROOT / "data" / "benchmark"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    sessions = {}
    with gzip.open(args.dataset_dir / "corpus_sessions.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                s = json.loads(line)
                texts = [t.get("text") or "" for t in s.get("turns", [])]
                sessions[s["session_id"]] = {
                    "turns": len(texts),
                    "tokens": sum(len(tok.encode(t, add_special_tokens=False)) for t in texts),
                    "words": sum(len(t.split()) for t in texts),
                }
    ego_map = json.loads((args.dataset_dir / "ego_session_map.json").read_text())
    n_instances = sum(
        sum(1 for line in (args.dataset_dir / "eval_instances" / f"{d}.jsonl").open() if line.strip()) for d in EVAL_SUBDIMS
    )

    tokens = [v["tokens"] for v in sessions.values()]
    turns = [v["turns"] for v in sessions.values()]
    total_tokens, total_turns = sum(tokens), sum(turns)
    ego_sessions = [len(ids) for ids in ego_map.values()]
    ego_turns = [sum(sessions[i]["turns"] for i in ids) for ids in ego_map.values()]
    ego_tokens = [sum(sessions[i]["tokens"] for i in ids) for ids in ego_map.values()]

    def mean_range(xs, fmt):
        return f"{fmt(st.mean(xs))} [{fmt(min(xs))}, {fmt(max(xs))}]"

    rows = [
        ("Agents / simulated days", f"{len(ego_map)} / {N_DAYS}"),
        ("Sessions / turns", f"{_n(len(sessions))} / {_n(total_turns)}"),
        ("Corpus tokens / whitespace words", f"{_n(total_tokens)} / {_n(sum(v['words'] for v in sessions.values()))}"),
        ("Mean turns per session", f"{total_turns / len(sessions):.2f}"),
        ("Mean tokens per session", f"{total_tokens / len(sessions):.1f}"),
        ("Session token range", f"{_n(min(tokens))}--{_n(max(tokens))}"),
        ("Ego sessions per agent, mean [min, max]", mean_range(ego_sessions, lambda x: f"{x:,.1f}" if x != int(x) else _n(x))),
        ("Ego turns per agent, mean [min, max]", mean_range(ego_turns, lambda x: f"{x:,.1f}" if x != int(x) else _n(x))),
        ("Ego-observed tokens per agent, mean [min, max]", mean_range(ego_tokens, _n)),
        ("Ego-observed tokens per agent per day, mean [min, max]", mean_range([t / N_DAYS for t in ego_tokens], _n)),
        (r"$2\times$ corpus tokens / 50-agent rough estimate", _n(2 * total_tokens / len(ego_map))),
        ("Evaluated instances", _n(n_instances)),
    ]
    body = "\n".join(f"{k} & {v} \\\\" for k, v in rows) + "\n"
    out = args.out or table_path("dataset_summary_body.tex")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")
    print(f"[gen_dataset_summary] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
