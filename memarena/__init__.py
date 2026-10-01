"""MemArena: egocentric, permission-aware memory benchmark for LLM agents.

Top-level entry points:
    - memarena.tools    — dataset QA utilities (scan/repair/bundle/validate)
    - memarena.figures  — paper-figure reproduction scripts
    - memarena.human_calibration — judge-vs-human κ tooling

The underlying data-generation pipeline lives in the sibling package `MASim`;
the evaluation pipeline lives in the sibling package `eval`. Both are
installed alongside `memarena` by `pip install memarena`.
"""
from __future__ import annotations

from memarena.runtime import configure_live_output

configure_live_output()

__version__ = "1.0.0"
__all__ = ["__version__"]
