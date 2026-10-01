#!/usr/bin/env bash
# D6 self-probe — remaining 8b and 32b cells only.
# 8b on port 16003 (dp=3), 32b on port 16004 (dp=5).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate
if [[ -f .env ]]; then set -a; . .env; set +a; fi

PYTHON=".venv/bin/python"
RUN_DIR="data/benchmark"
OUT_BASE="out/d6_self_probe"

BACKENDS=(vanilla oracle baseline_simplerag memobase memsearch)
TRIALS=(s2 s3 s4)

declare -A MODEL_NAME MODEL_PORT MODEL_CACHE_DIR
MODEL_NAME[8b]="Qwen/Qwen3-8B"
MODEL_NAME[32b]="Qwen/Qwen3-32B-AWQ"
MODEL_PORT[8b]=16003
MODEL_PORT[32b]=16004
MODEL_CACHE_DIR[8b]="out/accuracy_memarena_l_8b/memory_cache"
MODEL_CACHE_DIR[32b]="out/accuracy_memarena_l_32b/memory_cache"

log() { echo "[d6-8b32b $(date -u +%H:%M:%S)] $*"; }

run_cell() {
  local tag="$1" backend="$2" trial="$3"
  local out_dir="$OUT_BASE/${backend}_${tag}/${trial}"

  # Skip if already done
  local ans_file
  ans_file=$(find "$out_dir" -name "answer_results_run.json" -not -path "*/runs/*" 2>/dev/null | head -1)
  if [[ -n "$ans_file" ]]; then
    log "  SKIP (done) ${backend}/${tag}/${trial}"
    return 0
  fi

  mkdir -p "$out_dir"
  local extra_args=() stages=()
  case "$backend" in
    memobase|memsearch)
      local cache_path="${MODEL_CACHE_DIR[$tag]}/${backend}/${tag}/${trial}/memcache_${backend}_A_paired_${tag}_${trial}.jsonl"
      extra_args+=(--system memory_cache --cache-path "$cache_path" --expected-memory-system "$backend")
      stages=(search answer)
      ;;
    baseline_simplerag)
      extra_args+=(--system baseline_simplerag)
      stages=(search answer)
      ;;
    *)
      extra_args+=(--system "$backend")
      stages=(answer)
      ;;
  esac

  local log_file="/tmp/d6sp_${backend}_${tag}_${trial}.log"
  log "  starting ${backend}/${tag}/${trial}"
  "$PYTHON" eval/cli.py \
    --run-dir "$RUN_DIR" \
    --output-dir "$out_dir" \
    "${extra_args[@]}" \
    --model "${MODEL_NAME[$tag]}" \
    --endpoint "http://localhost:${MODEL_PORT[$tag]}/v1" \
    --trial-name "$trial" \
    --stages "${stages[@]}" \
    --dimensions d4_permission \
    --d6-arm B \
    --d6-probe-mode self_ego \
    --answer-concurrency 8 \
    >"$log_file" 2>&1
  local rc=$?
  [[ $rc -eq 0 ]] && log "OK    ${backend}/${tag}/${trial}" || log "FAILED ${backend}/${tag}/${trial} (rc=$rc)"
  return $rc
}

log "=== D6 self-probe: 8b + 32b remaining cells ==="

pids=(); labels=()
for tag in 8b 32b; do
  for backend in "${BACKENDS[@]}"; do
    for trial in "${TRIALS[@]}"; do
      run_cell "$tag" "$backend" "$trial" &
      pids+=("$!"); labels+=("${backend}_${tag}_${trial}")
    done
  done
done

log "waiting for ${#pids[@]} cells..."
failed=0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || { log "FAILED: ${labels[$i]}"; failed=$((failed+1)); }
done
log "=== done: $((${#pids[@]}-failed))/${#pids[@]} OK, $failed failed ==="
