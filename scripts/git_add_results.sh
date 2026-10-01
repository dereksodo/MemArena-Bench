#!/usr/bin/env bash
# git_add_results.sh — force-add the whitelisted stats/data files under a
# results directory so they can be committed to a data-snapshot branch.
# `out/` is gitignored; this wrapper bypasses that for the small set of
# files needed for the paper aggregation (eval results + per-question
# answer + run meta + wrapper logs + manifests).
#
# Skipped (huge / regenerable, see docs/data_merge.md §2):
#   memcache_*.jsonl, masim_transcript_*.jsonl, masim_qa_*.json,
#   openclaw_cleanup_*, input_output_*, search_results_*, *_light_*.
#
# Usage:
#   bash scripts/git_add_results.sh out/accuracy_memarena_l_baselines_7b
#   bash scripts/git_add_results.sh out/latency_spark_0_6b_s2

set -euo pipefail

[[ "$#" -ge 1 ]] || { echo "usage: $0 <out/dir> [<out/dir> ...]"; exit 2; }

for d in "$@"; do
  if [[ ! -d "$d" ]]; then
    echo "[git_add_results] skip $d (not a directory)"
    continue
  fi
  echo "[git_add_results] adding from $d"

  # Per-cell evaluation summaries (accuracy + per-Q judging)
  find "$d" -name "evaluation_results_*judge*.json" -exec git add -f {} +

  # Per-question answer results (full version only — drop the *_light_*
  # duplicates which omit raw_response).
  find "$d" -name "answer_results_*.json" -not -name "*_light_*" -exec git add -f {} +

  # Per-cell config metadata
  find "$d" -name "run_meta_*.json" -exec git add -f {} +

  # Wrapper logs (per-day cadence + errors)
  find "$d/wrapper_logs" -maxdepth 1 -name "*.log" -exec git add -f {} + 2>/dev/null

  # Top-level run manifests
  find "$d" -maxdepth 1 -name "*manifest*" -exec git add -f {} +

  # Latency runs also stash a one-line elapsed summary
  find "$d" -maxdepth 1 -name "_elapsed_seconds.txt" -exec git add -f {} +
done

echo "[git_add_results] done. Run 'git status --short | head' to verify."
