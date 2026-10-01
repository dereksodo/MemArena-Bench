"""Compare the numbers in two LaTeX tables, in order.

Used by ``scripts/reproduce_figures.py --check`` to prove that a regenerated
table carries exactly the values of the checked-in one. Comments, \\label{} /
\\ref{} / \\cite{} arguments and colour specs are ignored, so formatting
changes do not register as differences; any changed, added or dropped number
does.

    python -m memarena.tools.tex_numeric_diff regenerated.tex paper_rev0719/tables/main_SML.tex
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple

_COMMENT = re.compile(r"(?<!\\)%.*$", re.MULTILINE)
_SKIP_ARGS = re.compile(r"\\(?:label|ref|eqref|cite|citep|citet|textcolor|color|includegraphics|input|usepackage)\s*(?:\[[^\]]*\])?\{[^}]*\}")
_COLOUR_MIX = re.compile(r"\b[a-z]+!\d+(?:!\w+)*")
_NUMBER = re.compile(r"(?<![\w.])[-+−]?\d+(?:[.,]\d+)*(?:\.\d+)?")


_TABULAR_BODY = re.compile(r"\\begin\{tabular\*?\}.*?\\end\{tabular\*?\}", re.DOTALL)


def extract_numbers(tex: str, body_only: bool = False, skip_header: bool = False) -> List[str]:
    text = _COMMENT.sub("", tex)
    if body_only or skip_header:
        text = "\n".join(m.group(0) for m in _TABULAR_BODY.finditer(text))
    if skip_header:
        text = "\n".join(part.split("\\midrule", 1)[-1] for part in text.split("\\end{tabular"))
    text = _SKIP_ARGS.sub(" ", text)
    text = _COLOUR_MIX.sub(" ", text)
    text = text.replace("{,}", "")
    return [m.group(0).replace("−", "-").lstrip("+") for m in _NUMBER.finditer(text)]


def diff_numbers(a: List[str], b: List[str], tol: float = 0.0) -> List[Tuple[int, str, str]]:
    out: List[Tuple[int, str, str]] = []
    for i in range(max(len(a), len(b))):
        x = a[i] if i < len(a) else "<missing>"
        y = b[i] if i < len(b) else "<missing>"
        if x == y:
            continue
        try:
            if abs(float(x.replace(",", "")) - float(y.replace(",", ""))) <= tol:
                continue
        except ValueError:
            pass
        out.append((i, x, y))
    return out


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Diff the numbers of two LaTeX tables.")
    ap.add_argument("generated", type=Path)
    ap.add_argument("reference", type=Path)
    ap.add_argument("--tol", type=float, default=0.0, help="absolute tolerance per number")
    ap.add_argument("--show", type=int, default=20, help="max mismatches to print")
    ap.add_argument("--body-only", action="store_true", help="compare only tabular bodies (ignore captions)")
    ap.add_argument("--skip-header", action="store_true", help="also ignore everything up to each tabular's first \\midrule")
    args = ap.parse_args(argv)

    gen = extract_numbers(args.generated.read_text(encoding="utf-8"), args.body_only, args.skip_header)
    ref = extract_numbers(args.reference.read_text(encoding="utf-8"), args.body_only, args.skip_header)
    bad = diff_numbers(gen, ref, args.tol)
    if not bad:
        print(f"OK {args.reference.name}: {len(ref)} numbers identical")
        return 0
    print(f"DIFF {args.reference.name}: {len(bad)} of {max(len(gen), len(ref))} numbers differ")
    for i, x, y in bad[: args.show]:
        print(f"  #{i}: generated {x} vs reference {y}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
