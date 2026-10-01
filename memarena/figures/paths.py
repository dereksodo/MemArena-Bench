"""Shared output paths for MemArena figure/table generators."""

from __future__ import annotations

import os
from pathlib import Path


FIGURE_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIGURE_DIR.parent.parent


def _repo_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def artifact_root() -> Path:
    """Return the root directory for generated figure/table artifacts.

    Preference order:
    1. ``MEMARENA_FIGURE_OUT_DIR``: explicit artifact output root.
    2. ``MEMARENA_RUN_DIR``: keep artifacts next to the staged input under out/.
    3. ``out/reproduced_figures`` so reruns never write generated artifacts
       into the source tree by accident.
    """
    override = os.getenv("MEMARENA_FIGURE_OUT_DIR")
    if override:
        return _repo_path(override)
    run_dir = os.getenv("MEMARENA_RUN_DIR")
    if run_dir:
        return _repo_path(run_dir)
    return REPO_ROOT / "out" / "reproduced_figures"


def figures_dir() -> Path:
    return artifact_root() / "figures"


def tables_dir() -> Path:
    return artifact_root() / "tables"


def figure_path(filename: str) -> Path:
    return figures_dir() / filename


def table_path(filename: str) -> Path:
    return tables_dir() / filename
