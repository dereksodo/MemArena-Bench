#!/usr/bin/env python3
"""Stitch H100 + H200 + Spark output dirs into one ``MEMARENA_RUN_DIR`` tree
that ``memarena/figures/paper_data.py`` can scan.

The figure scripts expect:

  $RUN_DIR/
    eval_results_s2/
      vanilla/   evaluation_results_vanilla_${model}.json
      oracle/    evaluation_results_oracle_${model}.json
      inmem/     evaluation_results_rag_${model}.json
      memory_cache/
                 evaluation_results_memcache_${backend}_${config}_${model_long}_s2.json
    eval_results_s3/  ...
    eval_results_s4/  ...

But our actual files (per docs/data_merge.md §1) live as:

  out/accuracy_memarena_l_${model}/eval_results_${trial}/memory_cache/
        evaluation_results_${backend}_${model}_${trial}_judge_remote.json
  out/accuracy_memarena_l_baselines_${model}/eval_results_${trial}/(vanilla|oracle|inmem)/
        evaluation_results_(vanilla|oracle|inmem)_${model}_${trial}_judge_remote.json

This script symlinks each source file into the expected target path with the
right name. We only do symlinks (no copies) so re-running is cheap and the
underlying data remains a single source of truth.

Usage:
  python scripts/build_run_merged.py [--root out/run_merged_v1]
  MEMARENA_RUN_DIR=out/run_merged_v1 \
    .venv/bin/python scripts/reproduce_figures.py --all
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, Optional


# Map our short model tag to the *_long_* fragment paper_data expects inside
# memcache filenames (see _normalise_memcache_model in paper_data.py).
MODEL_LONG: dict[str, str] = {
    "0_6b":    "qwen3_0_6b",
    "llama3b": "llama3_2_3b",
    "7b":      "mistral_7b",
    "8b":      "qwen3_8b_awq",
    "32b":     "qwen3_32b_awq",
}

# H200's "inmem" baseline backend is paper_data's "rag".
BASELINE_DIR_MAP = {"vanilla": "vanilla", "oracle": "oracle", "inmem": "inmem"}
BASELINE_NAME_MAP = {"vanilla": "vanilla", "oracle": "oracle", "inmem": "rag"}

# Memcache backends paper_data understands.
MEMCACHE_BACKENDS = {"memobase", "memos", "mem0", "graphiti", "hipporag"}

# All trial directories we expect to find.
TRIAL_DIRS = ("eval_results_s1", "eval_results_s2", "eval_results_s3", "eval_results_s4")

_BASELINE_FNAME_RE = re.compile(
    r"^evaluation_results_(?P<backend>vanilla|oracle|inmem)_"
    r"(?P<model>[^_]+(?:_[^_]+)*?)_(?P<trial>s[1-9])_judge_remote\.json$"
)
_MEMCACHE_FNAME_RE = re.compile(
    r"^evaluation_results_(?P<backend>memobase|memos|mem0|graphiti|hipporag)_"
    r"(?P<model>[^_]+(?:_[^_]+)*?)_(?P<trial>s[1-9])_judge_remote\.json$"
)


def _model_short_from_filename(raw: str) -> Optional[str]:
    """Map raw filename model fragment back to canonical short tag."""
    if raw in MODEL_LONG:
        return raw
    return None


def _link(src: Path, dst: Path, *, dry: bool) -> str:
    if dst.is_symlink() or dst.exists():
        return "skip"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dry:
        return "dry"
    rel = src.resolve().relative_to(dst.parent.resolve(), walk_up=True) if hasattr(Path, "walk_up") else None
    target = src.resolve()
    dst.symlink_to(target)
    return "ok"


def link_baseline(src: Path, root: Path, *, dry: bool) -> Optional[str]:
    m = _BASELINE_FNAME_RE.match(src.name)
    if not m:
        return None
    backend = m.group("backend")
    model = _model_short_from_filename(m.group("model"))
    trial = m.group("trial")
    if model is None:
        return None
    out_backend_dir = BASELINE_DIR_MAP[backend]
    out_backend_label = BASELINE_NAME_MAP[backend]
    dst = root / f"eval_results_{trial}" / out_backend_dir / \
        f"evaluation_results_{out_backend_label}_{model}.json"
    return _link(src, dst, dry=dry)


def link_memcache(src: Path, root: Path, *, dry: bool) -> Optional[str]:
    m = _MEMCACHE_FNAME_RE.match(src.name)
    if not m:
        return None
    backend = m.group("backend")
    model = _model_short_from_filename(m.group("model"))
    trial = m.group("trial")
    if model is None or backend not in MEMCACHE_BACKENDS:
        return None
    long_name = MODEL_LONG[model]
    # paper_data scans memory_cache/memory_cache/ (double-nested).
    dst = root / f"eval_results_{trial}" / "memory_cache" / "memory_cache" / \
        f"evaluation_results_memcache_{backend}_A_paired_{long_name}_{trial}.json"
    return _link(src, dst, dry=dry)


def iter_eval_files() -> Iterable[Path]:
    """Yield every evaluation_results_*judge_remote.json file under
    memarena-L source directories ONLY.

    We deliberately exclude:
      - non-L benchmarks (out/accuracy_5a10d_*, out/accuracy_10a5d_*)
      - paper input snapshots (out/paper_accuracy_*)
      - broken-run sentinels (.bad_subnet_failure*, .too_weak*, etc.)
      - timestamped per-run snapshots under runs/ subdirs
    """
    out_root = Path("out")
    if not out_root.exists():
        return
    # Allowlist of top-level dir prefixes that contain valid memarena-L cells.
    L_PREFIXES = (
        "accuracy_memarena_l_",        # H100 main runs
        "ablations_l_",                # H200 ablations
        "ablation_l_",                 # tolerated alternate naming
        # Latency runs separately (run_latency_spark.sh outputs)
        # — we exclude here because their answer_results live in latency-only
        # cells without a matching paper_data backend label.
    )
    SKIP_SENTINELS = (".bad_subnet_failure_20260502", ".too_weak_0_6b_archived",
                      ".day0_only_failed_at_day1", ".broken_port_bug",
                      ".bad_partial", ".day0_only", ".broken")

    for top in sorted(p for p in out_root.iterdir() if p.is_dir()):
        # Must start with one of the allowed prefixes.
        if not any(top.name.startswith(pref) for pref in L_PREFIXES):
            continue
        # Skip dirs flagged as broken / archived.
        if any(top.name.endswith(s) for s in SKIP_SENTINELS):
            continue
        for p in top.rglob("evaluation_results_*judge*.json"):
            # Skip the timestamped per-run replicas under runs/ subdirs.
            if "runs" in p.relative_to(top).parts:
                continue
            yield p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="out/run_merged_v1",
                    help="output run dir (will be created)")
    ap.add_argument("--dry-run", action="store_true",
                    help="don't actually symlink; just report what would happen")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    counts = {"baseline_ok": 0, "memcache_ok": 0, "skip": 0,
              "unmatched": 0, "dry": 0}

    for f in iter_eval_files():
        for fn in (link_baseline, link_memcache):
            r = fn(f, root, dry=args.dry_run)
            if r is None:
                continue
            if r == "skip":
                counts["skip"] += 1
            elif r == "dry":
                counts["dry"] += 1
            else:
                key = "baseline_ok" if fn is link_baseline else "memcache_ok"
                counts[key] += 1
            break
        else:
            # Neither pattern matched.
            counts["unmatched"] += 1

    print(f"[build_run_merged] root: {root}")
    for k, v in counts.items():
        print(f"  {k:14s}: {v}")
    print(f"\nNext step:\n  MEMARENA_RUN_DIR={root.relative_to(Path.cwd()) if root.is_relative_to(Path.cwd()) else root} \\")
    print(f"    .venv/bin/python scripts/reproduce_figures.py --all")
    return 0


if __name__ == "__main__":
    sys.exit(main())
