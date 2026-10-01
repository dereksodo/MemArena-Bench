#!/usr/bin/env bash
# Run a single-model matrix where backends are sequential and trials within
# each backend run in parallel. Designed for the "small focused" pattern:
# one reader model gets DP=8 across all GPUs, so we want only one backend
# active at a time but maximize trial parallelism within that backend.
#
# Layout:
#   for backend in $BACKENDS:
#     fan out trials concurrently
#     wait
#
# Same memobase/memos server instance handles all 3 trials simultaneously
# (each trial uses different namespace so ego users don't collide).
#
# Required env: OPENROUTER_API_KEY (for remote judge).
# Required args: --run-dir PATH --out-dir PATH --models-file PATH
#                --backends LIST --trials LIST
# Optional: anything else passed through to scripts/run_eval_matrix.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MATRIX="$SCRIPT_DIR/run_eval_matrix.sh"

usage() {
  cat <<EOF
Usage: $(basename "$0") --run-dir PATH --out-dir PATH --models-file PATH \\
                       --backends b1,b2 --trials t1,t2,t3 [matrix args ...]

For each backend (sequential), launches one matrix instance per trial in
parallel and waits before moving to the next backend.

Examples:
  $(basename "$0") \\
    --run-dir MASim/runs/l_20260408_111046 \\
    --out-dir out/L_3b \\
    --models-file /tmp/models_3b_only.tsv \\
    --backends memobase,memos \\
    --trials s2,s3,s4 \\
    --judge-preset remote \\
    --remote-judge-model openai/gpt-4o-mini-2024-07-18 \\
    --remote-judge-endpoint https://openrouter.ai/api/v1
EOF
}

RUN_DIR="" OUT_DIR="" MODELS_FILE="" BACKENDS="" TRIALS=""
PASSTHROUGH=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir)     RUN_DIR="$2"; shift 2 ;;
    --out-dir)     OUT_DIR="$2"; shift 2 ;;
    --models-file) MODELS_FILE="$2"; shift 2 ;;
    --backends)    BACKENDS="$2"; shift 2 ;;
    --trials)      TRIALS="$2"; shift 2 ;;
    -h|--help)     usage; exit 0 ;;
    *)             PASSTHROUGH+=("$1"); shift ;;
  esac
done

[[ -n "$RUN_DIR" && -n "$OUT_DIR" && -n "$MODELS_FILE" && -n "$BACKENDS" && -n "$TRIALS" ]] \
  || { usage; exit 2; }

# Auto-add OpenRouter API key arg if env is set and caller didn't pass --judge-api-key
if [[ -n "${OPENROUTER_API_KEY:-}" ]]; then
  has_key=0
  for a in "${PASSTHROUGH[@]:-}"; do
    [[ "$a" == "--judge-api-key" ]] && { has_key=1; break; }
  done
  if [[ "$has_key" -eq 0 ]]; then
    PASSTHROUGH+=(--judge-api-key "$OPENROUTER_API_KEY")
  fi
fi

mkdir -p "$OUT_DIR"
LOG_DIR="${OUT_DIR}/wrapper_logs"
mkdir -p "$LOG_DIR"

IFS=',' read -r -a BACKEND_ARR <<<"$BACKENDS"
IFS=',' read -r -a TRIAL_ARR <<<"$TRIALS"

echo "[wrapper] backends (sequential): ${BACKEND_ARR[*]}"
echo "[wrapper] trials (parallel):     ${TRIAL_ARR[*]}"
echo "[wrapper] models-file:           $MODELS_FILE"
echo "[wrapper] run-dir:               $RUN_DIR"
echo "[wrapper] out-dir:               $OUT_DIR"
echo "[wrapper] passthrough args:      ${PASSTHROUGH[*]}"
echo

for backend in "${BACKEND_ARR[@]}"; do
  echo "[wrapper] ===== backend=$backend (3 trials in parallel) ====="
  pids=()
  for trial in "${TRIAL_ARR[@]}"; do
    log="${LOG_DIR}/${backend}_${trial}.log"
    echo "[wrapper] launching $backend × $trial → $log"
    bash "$MATRIX" \
      --run-dir "$RUN_DIR" \
      --out-dir "$OUT_DIR" \
      --models-file "$MODELS_FILE" \
      --backends "$backend" \
      --trials "$trial" \
      --force \
      "${PASSTHROUGH[@]}" \
      >"$log" 2>&1 &
    pids+=("$!")
  done
  # Wait for all trial cells of this backend to finish
  failed=0
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      echo "[wrapper] WARNING: ${BACKEND_ARR[$(( i % ${#BACKEND_ARR[@]} ))]} trial ${TRIAL_ARR[$i]} exited non-zero"
      failed=$(( failed + 1 ))
    fi
  done
  echo "[wrapper] backend $backend done ($failed trial failure(s))"
  echo
done

echo "[wrapper] all backends complete"
