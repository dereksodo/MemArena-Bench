"""Unit tests for the production MASim launcher."""

from __future__ import annotations

from pathlib import Path

from scripts.run_masim import (
    DEFAULT_CONFIG,
    DEFAULT_RUNS_DIR,
    _default_output_dir,
    _load_yaml_with_base,
    _normalise_sglang_url,
    _resolve_repo_path,
)

REPO = Path(__file__).resolve().parents[1]


def test_launcher_default_config_is_memarena_l() -> None:
    cfg = _load_yaml_with_base(_resolve_repo_path(DEFAULT_CONFIG))

    assert DEFAULT_CONFIG.name == "memarena_l.yaml"
    assert cfg["graph"]["n_agents"] == 50
    assert cfg["time_range"] == [0.0, 15.0]
    assert cfg["llm"]["model"] == "qwen3"
    assert "base" not in cfg


def test_launcher_small_gpu_config_shape() -> None:
    cfg = _load_yaml_with_base(REPO / "MASim" / "configs" / "memarena_5a10d_5k.yaml")

    assert cfg["graph"]["n_agents"] == 5
    assert cfg["time_range"] == [0.0, 10.0]
    assert cfg["sessions_per_day_per_agent"] == 4
    assert cfg["target_tokens"] == 0
    assert cfg["llm"]["concurrency"] == 32


def test_launcher_normalises_sglang_url() -> None:
    assert _normalise_sglang_url("http://localhost:8000") == "http://localhost:8000/v1"
    assert _normalise_sglang_url("http://localhost:8000/v1") == "http://localhost:8000/v1"


def test_launcher_default_output_dir_uses_masim_runs() -> None:
    path = _default_output_dir(REPO / "MASim" / "configs" / "memarena_5a10d_5k.yaml", "20260501_120000")

    assert path == REPO / DEFAULT_RUNS_DIR / "20260501_120000_memarena_5a10d_5k"


def test_masim_runtime_dependencies_are_declared() -> None:
    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")

    for package in ("networkx", "rich", "tiktoken"):
        assert f'"{package}>=' in pyproject
