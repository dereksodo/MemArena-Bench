#!/usr/bin/env python3
"""Download MemArena-L from a Hugging Face dataset repository.

Set ``MEMARENA_HF_REPO_ID`` or pass ``--repo-id`` to point this script at the
MemArena-L Hugging Face dataset repository.

Usage
-----
  python scripts/download_dataset.py
  python scripts/download_dataset.py --out data
  python scripts/download_dataset.py --repo-id <owner>/<dataset>

The script snapshots the Hub repo, verifies ``SHA256SUMS`` when present, and
materializes the benchmark under ``data/benchmark`` for the rest of the repo.
Baseline result files are intentionally not part of the hosted dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from memarena.runtime import configure_live_output

RELEASE_VERSION = "1.0.0"
DEFAULT_REPO_ID = os.environ.get(
    "MEMARENA_HF_REPO_ID",
    "dereksodo/memarena-l",
)

BENCHMARK_ENTRIES = {
    "agents_personas.jsonl.gz",
    "agent_schedules.jsonl.gz",
    "corpus_sessions.jsonl.gz",
    "events.jsonl",
    "groups.json",
    "locations.json",
    "graph_edges.jsonl",
    "ego_projections.json",
    "ego_session_map.json",
    "day_checkpoint.json",
    "pipeline_report.json",
    "eval_instances",
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_extract(tar_path: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    root = dst.resolve()
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            target = (dst / member.name).resolve()
            if not str(target).startswith(str(root)):
                raise SystemExit(f"refusing tar entry outside dst: {member.name}")
        tar.extractall(dst)


def copy_entry(src: Path, dst: Path) -> None:
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def verify_manifest(snapshot_dir: Path) -> None:
    manifest = snapshot_dir / "SHA256SUMS"
    if not manifest.exists():
        croissant = snapshot_dir / "croissant.json"
        if not croissant.exists():
            print("[verify] no SHA256SUMS or croissant.json found; skipping checksum verification")
            return
        print("[verify] checking sha256 in croissant.json")
        for dist in json.loads(croissant.read_text(encoding="utf-8")).get("distribution", []):
            rel, expected = dist.get("contentUrl"), dist.get("sha256")
            if not rel or not expected or "://" in rel:
                continue
            path = snapshot_dir / rel
            if not path.exists():
                raise SystemExit(f"croissant entry missing: {rel}")
            actual = sha256_of(path)
            if actual != expected:
                raise SystemExit(f"SHA256 mismatch for {rel}: expected {expected}, got {actual}")
        return

    print("[verify] checking SHA256SUMS")
    for raw in manifest.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        expected, rel = line.split(maxsplit=1)
        path = snapshot_dir / rel
        if not path.exists():
            raise SystemExit(f"manifest entry missing: {rel}")
        actual = sha256_of(path)
        if actual != expected:
            raise SystemExit(f"SHA256 mismatch for {rel}: expected {expected}, got {actual}")


def materialize_benchmark(snapshot_dir: Path, out_dir: Path) -> Path:
    benchmark_dir = out_dir / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    tarball = snapshot_dir / f"memarena-l-v{RELEASE_VERSION}.tar.gz"
    if tarball.exists():
        print(f"[materialize] extracting {tarball.name} -> {benchmark_dir}")
        safe_extract(tarball, benchmark_dir)
        return benchmark_dir

    missing = [name for name in BENCHMARK_ENTRIES if not (snapshot_dir / name).exists()]
    if missing:
        raise SystemExit(
            "snapshot does not look like a MemArena-L repo; missing: "
            + ", ".join(sorted(missing))
        )

    print(f"[materialize] copying benchmark files -> {benchmark_dir}")
    for name in sorted(BENCHMARK_ENTRIES):
        copy_entry(snapshot_dir / name, benchmark_dir / name)
    return benchmark_dir


def main() -> int:
    configure_live_output()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face dataset repo id")
    ap.add_argument("--revision", default="main", help="Hub revision, branch, or commit")
    ap.add_argument("--out", type=Path, default=Path("data"), help="download directory")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF token for private/gated repos")
    ap.add_argument("--no-verify", action="store_true", help="skip SHA256SUMS verification")
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required. Activate .venv, then run: "
            "python -m pip install huggingface_hub"
        ) from exc

    out = args.out.resolve()
    snapshot_dir = out / "hf_snapshot"
    out.mkdir(parents=True, exist_ok=True)

    ignore_patterns = ["baselines/**", "*baseline*", "eval_results/**", "spark_results/**"]
    print(f"[download] snapshot {args.repo_id}@{args.revision} -> {snapshot_dir}")
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=snapshot_dir,
        token=args.token,
        ignore_patterns=ignore_patterns,
    )

    if not args.no_verify:
        verify_manifest(snapshot_dir)
    benchmark_dir = materialize_benchmark(snapshot_dir, out)

    print()
    print(f"Done. Benchmark files are under: {benchmark_dir}")
    print(f"Hub snapshot is under: {snapshot_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
