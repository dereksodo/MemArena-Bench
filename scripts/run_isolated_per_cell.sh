#!/usr/bin/env bash
# Run a per-cell isolated matrix where every (backend, model, trial)
# triple gets its own memobase + memos stack (no shared postgres / redis
# / qdrant / neo4j). This is the "permanent" variant of the L matrix run
# — no ports are baked in, everything is computed from
# scripts/spin_per_trial_stacks.sh.
#
# Steps performed automatically:
#   1) for each (model, trial): bring up isolated memobase + memos stacks
#      via spin_per_trial_stacks.sh
#   2) wait for memobase healthcheck on each new stack
#   3) materialize a per-trial models.tsv pointing at the right
#      memobase_url / memos_url
#   4) launch the matrix once per (backend, trial) cell, in parallel
#   5) wait for all cells; print summary
#
# Optional teardown: pass `--down-after`. By default stacks remain up so a
# follow-up rejudge / search-only step can reuse them.
#
# Required env: OPENROUTER_API_KEY (forwarded to remote judge).
# Required args: --run-dir, --out-dir, --models, --backends, --trials,
#                --reader-endpoint, --reader-model.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SPIN="$SCRIPT_DIR/spin_per_trial_stacks.sh"
MATRIX="$SCRIPT_DIR/run_eval_matrix.sh"

usage() {
  cat <<EOF
Usage: $(basename "$0") --run-dir DIR --out-dir DIR \\
                       --models M1[,M2,...] --backends b1[,b2,...] \\
                       --trials t1[,t2,...] \\
                       --reader-model NAME --reader-endpoint URL \\
                       [--down-after] [matrix args ...]

Each (model, trial) pair gets its own dedicated memobase + memos stack.
Within a model, all (backend, trial) cells run in parallel.

Examples:
  $(basename "$0") \\
    --run-dir data/benchmark \\
    --out-dir out/accuracy_memarena_l_3b \\
    --models llama3b \\
    --reader-model meta-llama/Llama-3.2-3B-Instruct \\
    --reader-endpoint http://localhost:16001 \\
    --backends memobase,memos \\
    --trials s2,s3,s4 \\
    --judge-preset remote \\
    --remote-judge-model openai/gpt-4o-mini-2024-07-18 \\
    --remote-judge-endpoint https://openrouter.ai/api/v1
EOF
}

RUN_DIR="" OUT_DIR="" MODELS="" BACKENDS="" TRIALS=""
READER_MODEL="" READER_ENDPOINT=""
DOWN_AFTER=0
PASSTHROUGH=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir)         RUN_DIR="$2"; shift 2 ;;
    --out-dir)         OUT_DIR="$2"; shift 2 ;;
    --models)          MODELS="$2"; shift 2 ;;
    --backends)        BACKENDS="$2"; shift 2 ;;
    --trials)          TRIALS="$2"; shift 2 ;;
    --reader-model)    READER_MODEL="$2"; shift 2 ;;
    --reader-endpoint) READER_ENDPOINT="$2"; shift 2 ;;
    --down-after)      DOWN_AFTER=1; shift ;;
    -h|--help)         usage; exit 0 ;;
    *)                 PASSTHROUGH+=("$1"); shift ;;
  esac
done

[[ -n "$RUN_DIR$OUT_DIR$MODELS$BACKENDS$TRIALS$READER_MODEL$READER_ENDPOINT" ]] || { usage; exit 2; }

# Auto-attach OpenRouter key if caller did not pass --judge-api-key
if [[ -n "${OPENROUTER_API_KEY:-}" ]]; then
  has_key=0
  for a in "${PASSTHROUGH[@]:-}"; do [[ "$a" == "--judge-api-key" ]] && { has_key=1; break; }; done
  [[ "$has_key" -eq 0 ]] && PASSTHROUGH+=(--judge-api-key "$OPENROUTER_API_KEY")
fi

mkdir -p "$OUT_DIR/wrapper_logs" "$OUT_DIR/_models_files"

IFS=',' read -r -a MODEL_ARR    <<<"$MODELS"
IFS=',' read -r -a BACKEND_ARR  <<<"$BACKENDS"
IFS=',' read -r -a TRIAL_ARR    <<<"$TRIALS"

cell_ports_line() { bash "$SPIN" urls "$1" "$2" 2>/dev/null | grep -E "MEMOBASE_BASE_URL|MEMOS_BASE_URL"; }

echo "[wrapper] models:    ${MODEL_ARR[*]}"
echo "[wrapper] backends:  ${BACKEND_ARR[*]}"
echo "[wrapper] trials:    ${TRIAL_ARR[*]}"
echo "[wrapper] run-dir:   $RUN_DIR"
echo "[wrapper] out-dir:   $OUT_DIR"
echo "[wrapper] reader:    $READER_MODEL @ $READER_ENDPOINT"
echo

# ---- Step 1: spin up stacks per (model, trial) ----------------------
for model in "${MODEL_ARR[@]}"; do
  echo "[wrapper] spinning stacks for $model on trials: ${TRIAL_ARR[*]}"
  bash "$SPIN" up "$model" "$(IFS=,; echo "${TRIAL_ARR[*]}")" 2>&1 | tail -10
done

# ---- Step 2: wait for memobase healthchecks ------------------------
echo "[wrapper] waiting for memobase healthchecks..."
for model in "${MODEL_ARR[@]}"; do
  for trial in "${TRIAL_ARR[@]}"; do
    mb_url=$(bash "$SPIN" urls "$model" "$trial" | awk -F= '/MEMOBASE_BASE_URL/{print $2}')
    deadline=$((SECONDS + 180))
    while ! curl -fsS "${mb_url}/api/v1/healthcheck" >/dev/null 2>&1; do
      if [[ "$SECONDS" -ge "$deadline" ]]; then
        echo "[wrapper] WARNING: memobase $model/$trial not healthy after 180s ($mb_url)"
        break
      fi
      sleep 2
    done
    echo "  ✓ $model/$trial → $mb_url"
  done
done

# ---- Step 3: materialize per-trial models.tsv -----------------------
for model in "${MODEL_ARR[@]}"; do
  for trial in "${TRIAL_ARR[@]}"; do
    mb_url=$(bash "$SPIN" urls "$model" "$trial" | awk -F= '/MEMOBASE_BASE_URL/{print $2}')
    mos_url=$(bash "$SPIN" urls "$model" "$trial" | awk -F= '/MEMOS_BASE_URL/{print $2}')
    tsv="$OUT_DIR/_models_files/${model}_${trial}.tsv"
    printf '%s|%s|%s|%s|%s\n' "$model" "$READER_MODEL" "$READER_ENDPOINT" "$mb_url" "$mos_url" >"$tsv"
    echo "[wrapper] models tsv: $tsv"
  done
done

# ---- Step 4: launch matrix per (backend, model, trial) ----------------
echo
echo "[wrapper] launching matrix cells in parallel..."
pids=()
labels=()
for model in "${MODEL_ARR[@]}"; do
  for backend in "${BACKEND_ARR[@]}"; do
    for trial in "${TRIAL_ARR[@]}"; do
      tsv="$OUT_DIR/_models_files/${model}_${trial}.tsv"
      log="$OUT_DIR/wrapper_logs/${backend}_${model}_${trial}.log"
      label="${backend}_${model}_${trial}"
      bash "$MATRIX" \
        --run-dir "$RUN_DIR" \
        --out-dir "$OUT_DIR" \
        --models-file "$tsv" \
        --backends "$backend" \
        --trials "$trial" \
        --force \
        "${PASSTHROUGH[@]}" \
        >"$log" 2>&1 &
      pids+=("$!")
      labels+=("$label")
      echo "  → $label (pid=$!)"
    done
  done
done

# ---- Step 5: wait + summary ------------------------------------------
echo
echo "[wrapper] waiting for ${#pids[@]} cells..."
failed=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "[wrapper] FAILED: ${labels[$i]} (see $OUT_DIR/wrapper_logs/${labels[$i]}.log)"
    failed=$((failed + 1))
  fi
done
echo
echo "[wrapper] done: $((${#pids[@]} - failed))/${#pids[@]} cells succeeded"

if [[ "$DOWN_AFTER" -eq 1 ]]; then
  echo "[wrapper] tearing down stacks..."
  for model in "${MODEL_ARR[@]}"; do
    bash "$SPIN" down "$model" "$(IFS=,; echo "${TRIAL_ARR[*]}")"
  done
fi

exit "$failed"
