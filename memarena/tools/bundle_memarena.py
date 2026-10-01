#!/usr/bin/env python3
"""Build the release tarball for MemArena-L.

Produces under --out:
  memarena-l-v{VERSION}.tar.gz                       (benchmark proper, ~42 MB)
  memarena-l-v{VERSION}.SHA256SUMS
  LICENSE, CITATION.cff                              (copies from docs/)

Usage:
  python -m memarena.tools.bundle_memarena --version 1.0.0 --out out/release/
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = Path(os.environ.get(
    "MEMARENA_DATASET_DIR",
    str(REPO / "MASim" / "datasets" / "memarena_l"),
))
DOCS_DIR = REPO / "docs"
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_tar(out_path: Path, members: list[tuple[Path, str]]) -> None:
    """Write a gzip tarball with (source_path, arcname) members."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_path, "w:gz") as tar:
        for src, arc in members:
            if not src.exists():
                raise FileNotFoundError(src)
            tar.add(src, arcname=arc, recursive=True)


def bundle_benchmark(version: str, out_dir: Path, dataset_dir: Path) -> Path:
    if not dataset_dir.exists():
        sys.exit(f"dataset dir missing: {dataset_dir}")

    out = out_dir / f"memarena-l-v{version}.tar.gz"
    members = []
    for child in sorted(dataset_dir.iterdir()):
        members.append((child, child.name))
    # Ship LICENSE + DATASET_CARD.md + CITATION.cff inside the archive too,
    # so the tarball is self-describing on its own.
    for extra in ("LICENSE", "DATASET_CARD.md", "CITATION.cff"):
        src = DOCS_DIR / extra
        if src.exists():
            members.append((src, extra))
    write_tar(out, members)
    return out


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()
    except Exception:
        return "unknown"


def write_manifest(out_dir: Path, version: str, tarballs: list[Path]) -> Path:
    manifest = out_dir / f"memarena-l-v{version}.SHA256SUMS"
    with manifest.open("w") as f:
        f.write(f"# MemArena-L v{version} — bundle checksums\n")
        f.write(f"# Generated from git commit {git_sha()}\n")
        for tb in tarballs:
            f.write(f"{sha256_of(tb)}  {tb.name}\n")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, help="e.g. 1.0.0")
    ap.add_argument("--out", type=Path, default=REPO / "out" / "release")
    ap.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    produced: list[Path] = []
    print(f"[bundle] benchmark bundle → {args.out}")
    produced.append(bundle_benchmark(args.version, args.out, args.dataset_dir))

    manifest = write_manifest(args.out, args.version, produced)

    # Also copy LICENSE and CITATION.cff alongside the tarballs for convenience.
    for extra in ("LICENSE", "CITATION.cff"):
        src = DOCS_DIR / extra
        if src.exists():
            shutil.copy2(src, args.out / extra)

    print("\n=== done ===")
    for p in produced:
        size_mb = p.stat().st_size / (1 << 20)
        print(f"  {p.name:60s} {size_mb:8.1f} MB")
    print(f"  {manifest.name}")


if __name__ == "__main__":
    main()
