"""Dry-run test for `scripts/run_latency.py`.

Uses the deterministic fake-latency sampler (no sglang server required).
Verifies per-query CSV and summary JSON outputs.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "mini_latency_queries.jsonl"


def _run(out_dir: Path, extra: list[str] | None = None) -> None:
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "run_latency.py"),
        "--dry-run",
        "--queries", str(FIXTURE),
        "--out-dir", str(out_dir),
        "--n", "10",
        "--warmup", "2",
    ]
    if extra:
        cmd.extend(extra)
    subprocess.run(cmd, check=True, cwd=str(REPO))


def _latest_subdir(root: Path) -> Path:
    subs = [p for p in root.iterdir() if p.is_dir()]
    assert subs, f"no subdir under {root}"
    return max(subs, key=lambda p: p.stat().st_mtime)


def test_latency_dryrun_csv_and_json(tmp_path: Path) -> None:
    out_root = tmp_path / "lat_out"
    _run(out_root)
    run_dir = _latest_subdir(out_root)

    csv_path = run_dir / "latency_per_query.csv"
    json_path = run_dir / "latency_summary.json"
    assert csv_path.exists(), f"missing {csv_path}"
    assert json_path.exists(), f"missing {json_path}"

    with csv_path.open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 10
    for r in rows:
        assert {"backend", "query_id", "ttft_ms", "decode_tok_per_s",
                "total_ms", "prompt_tokens", "completion_tokens",
                "mode"}.issubset(r)
        assert r["mode"] == "dry-run"
        assert float(r["ttft_ms"]) > 0
        assert float(r["total_ms"]) >= float(r["ttft_ms"])

    summary = json.loads(json_path.read_text())
    assert summary["dry_run"] is True
    assert summary["n"] == 10
    for metric in ("ttft_ms", "decode_tok_per_s", "total_ms"):
        m = summary["metrics"][metric]
        assert "mean" in m and "p50" in m and "p95" in m
        assert m["p95"] >= m["p50"]
    # ingest was not requested
    assert summary["ingest_ms"] is None


def test_latency_dryrun_with_ingest(tmp_path: Path) -> None:
    out_root = tmp_path / "lat_out_ingest"
    _run(out_root, extra=["--ingest"])
    run_dir = _latest_subdir(out_root)
    summary = json.loads((run_dir / "latency_summary.json").read_text())
    assert summary["ingest_ms"] is not None
    assert summary["ingest_ms"] > 0


def test_latency_dryrun_is_deterministic(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    _run(a, extra=["--seed", "11"])
    _run(b, extra=["--seed", "11"])
    sa = json.loads((_latest_subdir(a) / "latency_summary.json").read_text())
    sb = json.loads((_latest_subdir(b) / "latency_summary.json").read_text())
    assert sa["metrics"] == sb["metrics"], "fake-latency sampler must be deterministic"
