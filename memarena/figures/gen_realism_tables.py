"""Generate the realism-audit table bodies (see ``memarena.realism``).

  realism_structure_body.tex  interaction structure, MemArena-L vs. REALTALK
  realism_lexical_body.tex    length-controlled self-BLEU / distinct-2 across five corpora

    python -m memarena.figures.gen_realism_tables
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from memarena import realism
from memarena.figures.paths import table_path


def render(structure: dict, lexical: dict) -> dict:
    ours, real = structure["MemArena-L"], structure["REALTALK"]
    s_rows = [
        ("word tokens per message", "words_per_message", "{:.1f}"),
        ("word tokens per speaker-run (bursts merged)", "words_per_speaker_run", "{:.1f}"),
        ("messages per speaker-run", "messages_per_speaker_run", "{:.2f}"),
    ]
    lines = [f"{label} & ${fmt.format(ours[k])}$ & ${fmt.format(real[k])}$ \\\\" for label, k, fmt in s_rows]
    lines.append(f"messages containing ``?'' & ${100 * ours['question_message']:.1f}\\%$ & ${100 * real['question_message']:.1f}\\%$ \\\\")
    names = [n for n, _, _ in realism.CORPORA]
    lex = [
        "self-BLEU ($\\downarrow$) & " + " & ".join(f"${lexical[n]['self_bleu']:.3f}$" for n in names) + " \\\\",
        "distinct-2 ($\\uparrow$) & " + " & ".join(f"${lexical[n]['distinct_2']:.3f}$" for n in names) + " \\\\",
    ]
    return {"realism_structure_body.tex": "\n".join(lines) + "\n", "realism_lexical_body.tex": "\n".join(lex) + "\n"}


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate the realism-audit table bodies.")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    structure = {"MemArena-L": realism.interaction_structure(realism.memarena_sessions()),
                 "REALTALK": realism.interaction_structure(realism.realtalk_sessions())}
    lexical = realism.length_controlled_lexical()
    for name, body in render(structure, lexical).items():
        out = (args.out_dir / name) if args.out_dir else table_path(name)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(body, encoding="utf-8")
    print("[gen_realism_tables] " + "; ".join(f"{n} sb={v['self_bleu']:.4f} d2={v['distinct_2']:.4f} n={v['n']}" for n, v in lexical.items()))
    print("[gen_realism_tables] structure " + str({k: {m: round(x, 4) for m, x in v.items()} for k, v in structure.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
