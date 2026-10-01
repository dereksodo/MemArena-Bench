from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def _run_path_probe(env: dict[str, str]) -> list[str]:
    code = (
        "from memarena.figures.paths import figure_path, table_path\n"
        "print(figure_path('fig.pdf'))\n"
        "print(table_path('tab.tex'))\n"
    )
    out = subprocess.check_output(
        [sys.executable, "-c", code],
        cwd=str(REPO),
        env=env,
        text=True,
    )
    return out.strip().splitlines()


def test_figure_outputs_follow_memarena_run_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "paper_input"
    env = os.environ.copy()
    env.pop("MEMARENA_FIGURE_OUT_DIR", None)
    env["MEMARENA_RUN_DIR"] = str(run_dir)

    assert _run_path_probe(env) == [
        str(run_dir / "figures" / "fig.pdf"),
        str(run_dir / "tables" / "tab.tex"),
    ]


def test_figure_output_dir_overrides_run_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "paper_input"
    out_dir = tmp_path / "artifacts"
    env = os.environ.copy()
    env["MEMARENA_RUN_DIR"] = str(run_dir)
    env["MEMARENA_FIGURE_OUT_DIR"] = str(out_dir)

    assert _run_path_probe(env) == [
        str(out_dir / "figures" / "fig.pdf"),
        str(out_dir / "tables" / "tab.tex"),
    ]


def test_figure_outputs_default_to_out_directory() -> None:
    env = os.environ.copy()
    env.pop("MEMARENA_FIGURE_OUT_DIR", None)
    env.pop("MEMARENA_RUN_DIR", None)

    assert _run_path_probe(env) == [
        str(REPO / "out" / "reproduced_figures" / "figures" / "fig.pdf"),
        str(REPO / "out" / "reproduced_figures" / "tables" / "tab.tex"),
    ]
