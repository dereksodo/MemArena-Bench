#!/usr/bin/env bash
# run_latency_answer_sweep.sh — 5 reader × 3 backend = 15 cell sweep.
#
# For each reader: serve sglang with --max-running-requests=1 (so the GPU
# CANNOT batch our single-stream client with anything else, even theoretically)
# then run vanilla / oracle / inmem in sequence with --answer-concurrency 1.
# 90 instances per cell (out/latency_selection/queries_s2.json). No memcache
# needed — these three backends are all self-contained.
#
# Output:
#   out/latency2_answer/<backend>_<reader>_s2/
#       eval_results_s2/<backend>/answer_results_*.json
#       latency_summary.txt
#       logs/run_<backend>_<reader>_s2.log
#
# This script does NOT itself do ingest. It's pure answer-phase replay.
# Memobase / memsearch are handled by separate ingest+answer pipelines.
#
# Reader order: cheapest → heaviest, with 7b late so we can fail fast on
# the recently-fixed safetensors issue without burning 32b GPU time first.
#
# Run on the Spark host from the repository root. Designed to be backgrounded.

set -uo pipefail   # not -e: a single cell failure shouldn't abort the sweep

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
[[ -z "${OPENROUTER_API_KEY:-}" && -f .env ]] && { set -a; . .env; set +a; }

READERS=("32b")  # 0_6b/llama3b/8b vanilla+oracle done; 7b deferred (user investigating weight issue)
BACKENDS=("vanilla" "oracle")  # drop inmem; search_time covered by 0_6b inmem already
SGLANG_PORT="${SGLANG_PORT:-17000}"
GPU_DEVICES="${GPU_DEVICES:-0}"
N_GPUS="${N_GPUS:-1}"
OUT_ROOT="${OUT_ROOT:-out/latency2_answer}"
SWEEP_LOG="${SWEEP_LOG:-/tmp/sweep_latency_answer.log}"

declare -A MODEL_NAME=(
  [0_6b]="Qwen/Qwen3-0.6B"
  [llama3b]="meta-llama/Llama-3.2-3B-Instruct"
  [7b]="mistralai/Mistral-7B-Instruct-v0.3"
  [8b]="Qwen/Qwen3-8B"
  [32b]="Qwen/Qwen3-32B-AWQ"
)

log() { echo "[sweep $(date -u +%H:%M:%S)] $*" | tee -a "$SWEEP_LOG"; }

teardown_sglang() {
  docker rm -f lat-spark-sglang >/dev/null 2>&1 || true
}

start_sglang() {
  local reader="$1"
  local mname="${MODEL_NAME[$reader]}"
  log "starting sglang reader=$reader served=$mname --max-running-requests=1"
  teardown_sglang

  docker run -d --name lat-spark-sglang \
    --gpus "\"device=${GPU_DEVICES}\"" --ipc=host --shm-size 32g --restart no \
    -p "${SGLANG_PORT}:${SGLANG_PORT}" \
    -v "${MODEL_ROOT:-$HOME/models}":/models:ro \
    -v "${HF_HOME:-$HOME/.cache/huggingface}":/root/.cache/huggingface \
    --env "SGLANG_PIP_INSTALL=protobuf sentencepiece" \
    lmsysorg/sglang:spark \
    bash -lc 'set -euo pipefail
if [[ -n "${SGLANG_PIP_INSTALL:-}" ]]; then python3 -m pip install --break-system-packages --no-cache-dir ${SGLANG_PIP_INSTALL}; fi
exec "$@"
' bash python3 -m sglang.launch_server \
      --model-path "/models/${reader}" --served-model-name "$mname" \
      --host 0.0.0.0 --port "$SGLANG_PORT" --tp "$N_GPUS" --dp 1 \
      --context-length 16384 --mem-fraction-static 0.85 \
      --max-running-requests 1 \
    >/dev/null

  log "  waiting for sglang :$SGLANG_PORT (timeout 5 min)..."
  local deadline=$((SECONDS + 300))
  until curl -fsS -m 3 "http://127.0.0.1:${SGLANG_PORT}/v1/models" >/dev/null 2>&1; do
    if [[ $SECONDS -ge $deadline ]]; then
      log "  ERROR: sglang failed to come up in 5 min for $reader"
      docker logs --tail 60 lat-spark-sglang 2>&1 | sed 's/^/    /' | tee -a "$SWEEP_LOG"
      return 1
    fi
    sleep 5
  done
  log "  sglang ready"

  # Sanity probe — short, deterministic. If output is token salad we abort.
  local probe
  probe="$(curl -fsS -m 60 "http://127.0.0.1:${SGLANG_PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${mname}\",\"messages\":[{\"role\":\"user\",\"content\":\"Say only the word OK.\"}],\"max_tokens\":50,\"temperature\":0}" \
    2>/dev/null | python3 -c 'import sys, json; d=json.load(sys.stdin); print(d["choices"][0]["message"]["content"])' 2>/dev/null || true)"
  log "  sanity probe output: ${probe:0:80}"
  if [[ -z "$probe" ]]; then
    log "  ERROR: sanity probe produced no output for $reader"
    return 1
  fi
}

run_one_cell() {
  local backend="$1" reader="$2"
  local ns="${backend}_${reader}_s2"
  local out_dir="${OUT_ROOT}/${ns}"

  if [[ -f "${out_dir}/latency_summary.txt" ]]; then
    log "  cell=$ns already has latency_summary.txt — skip (delete to redo)"
    return 0
  fi

  log "  cell=$ns starting"
  local t0=$SECONDS
  rm -rf "$out_dir"
  BACKEND="$backend" MODEL="$reader" \
    SGLANG_URL="http://localhost:${SGLANG_PORT}" \
    bash scripts/run_latency_answer.sh \
    >>"$SWEEP_LOG" 2>&1
  local rc=$?
  local elapsed=$((SECONDS - t0))
  if [[ $rc -ne 0 ]]; then
    log "  cell=$ns FAILED rc=$rc elapsed=${elapsed}s"
    return $rc
  fi
  log "  cell=$ns OK elapsed=${elapsed}s"
  if [[ -f "${out_dir}/latency_summary.txt" ]]; then
    sed 's/^/    /' "${out_dir}/latency_summary.txt" | tee -a "$SWEEP_LOG"
  fi
}

main() {
  log "=== sweep start ==="
  log "READERS=${READERS[*]}  BACKENDS=${BACKENDS[*]}"
  log "OUT_ROOT=${OUT_ROOT}  SGLANG_PORT=${SGLANG_PORT}  GPU=${GPU_DEVICES}"

  local total=0 ok=0 fail=0
  for reader in "${READERS[@]}"; do
    if ! start_sglang "$reader"; then
      log "skipping reader=$reader (sglang failed to start)"
      teardown_sglang
      continue
    fi
    for backend in "${BACKENDS[@]}"; do
      total=$((total + 1))
      if run_one_cell "$backend" "$reader"; then
        ok=$((ok + 1))
      else
        fail=$((fail + 1))
      fi
    done
    log "tearing down sglang (reader=$reader done)"
    teardown_sglang
  done

  log "=== sweep done: total=$total ok=$ok fail=$fail ==="
}

main
