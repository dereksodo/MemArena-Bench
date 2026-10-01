#!/usr/bin/env bash
# run_latency_memobase_ingest.sh — measure memobase ingest cost on a single
# (reader, ego_set=8) cell. Memobase server-side LLM extraction depends on the
# extractor (=reader in A_paired), so this script is per-reader. We typically
# only run MODEL=0_6b (cheapest) and use the fitted LLM-gen model to extrapolate
# wall time/energy for other readers.
#
# Output:
#   out/latency2_ingest/memobase_${MODEL}_${TRIAL}/
#       memcache_memobase_A_paired_${MODEL}_${TRIAL}.jsonl
#       memcache_memobase_A_paired_${MODEL}_${TRIAL}.summary.json
#       hw/hw_probe_memobase_${MODEL}_${TRIAL}_ingest.csv
#       hw/hw_agg_memobase_${MODEL}_${TRIAL}_ingest.json
#
# Usage:
#   MODEL=0_6b bash scripts/run_latency_memobase_ingest.sh
#   MODEL=8b   bash scripts/run_latency_memobase_ingest.sh   (if you want a 2nd cell)

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
[[ -z "${OPENROUTER_API_KEY:-}" && -f .env ]] && { set -a; . .env; set +a; }

MODEL="${MODEL:-0_6b}"
TRIAL="${TRIAL:-s2}"
N_EGOS="${N_EGOS:-8}"
RUN_DIR="${RUN_DIR:-data/benchmark}"
OUT_ROOT="${OUT_ROOT:-out/latency2_ingest}"
NS="memobase_${MODEL}_${TRIAL}"
OUT_DIR="${OUT_ROOT}/${NS}"
SGLANG_PORT="${SGLANG_PORT:-17000}"
GPU_DEVICES="${GPU_DEVICES:-0}"
SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.85}"   # memobase ingest doesn't need ollama
SERVICES_ROOT="${MEMARENA_MEMORY_SERVICES_ROOT:-$HOME/memarena-memory-services}"

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/hw"
LOG="$OUT_DIR/logs/ingest_memobase_${MODEL}_${TRIAL}.log"
HW_CSV="$OUT_DIR/hw/hw_probe_memobase_${MODEL}_${TRIAL}_ingest.csv"
HW_AGG="$OUT_DIR/hw/hw_agg_memobase_${MODEL}_${TRIAL}_ingest.json"
HW_LOG="$OUT_DIR/logs/hw_probe_memobase_${MODEL}_${TRIAL}_ingest.log"
CACHE_OUT="$OUT_DIR/memcache_memobase_A_paired_${MODEL}_${TRIAL}.jsonl"

declare -A MODEL_NAME=(
  [0_6b]="Qwen/Qwen3-0.6B"
  [llama3b]="meta-llama/Llama-3.2-3B-Instruct"
  [7b]="mistralai/Mistral-7B-Instruct-v0.3"
  [8b]="Qwen/Qwen3-8B"
  [32b]="Qwen/Qwen3-32B-AWQ"
)

log() { echo "[memobase-ingest $(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

cleanup() {
  log "cleanup..."
  if [[ -n "${HW_PID:-}" ]]; then
    kill -TERM "$HW_PID" 2>/dev/null || true
    wait "$HW_PID" 2>/dev/null || true
    if [[ -s "$HW_CSV" ]]; then
      python3 scripts/aggregate_hw_probe_simple.py \
        --csv "$HW_CSV" --outfile "$HW_AGG" \
        >> "$HW_LOG" 2>&1 || log "  ! hw_aggregate failed"
    fi
  fi
  # Tear down memobase compose stack
  bash scripts/spin_per_trial_stacks.sh down "$MODEL" "$TRIAL" 2>&1 | tee -a "$LOG" | tail -3 || true
  docker rm -f lat-spark-sglang 2>/dev/null || true
}
trap cleanup EXIT

# ---- 1. start sglang serving the extractor (=reader in A_paired)
log "starting sglang reader=$MODEL --max-running-requests=1 --mem-fraction=$SGLANG_MEM_FRACTION"
docker rm -f lat-spark-sglang 2>/dev/null || true
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
    --model-path "/models/${MODEL}" --served-model-name "${MODEL_NAME[$MODEL]}" \
    --host 0.0.0.0 --port "$SGLANG_PORT" --tp 1 --dp 1 \
    --context-length 16384 --mem-fraction-static "$SGLANG_MEM_FRACTION" \
    --max-running-requests 1 \
  >/dev/null

log "  waiting for sglang :$SGLANG_PORT..."
deadline=$((SECONDS + 300))
until curl -fsS -m 3 "http://127.0.0.1:${SGLANG_PORT}/v1/models" >/dev/null 2>&1; do
  [[ $SECONDS -ge $deadline ]] && { log "  ERROR sglang failed to come up"; exit 1; }
  sleep 5
done
log "  sglang ready"

# Sanity probe
probe="$(curl -fsS -m 60 "http://127.0.0.1:${SGLANG_PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL_NAME[$MODEL]}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply OK.\"}],\"max_tokens\":20,\"temperature\":0}" \
  2>/dev/null | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d["choices"][0]["message"]["content"])' 2>/dev/null || true)"
log "  sanity probe: ${probe:0:60}"
[[ -z "$probe" ]] && { log "  ERROR sanity probe empty"; exit 1; }

# ---- 2. spin memobase compose stack
log "spinning memobase stack ($MODEL/$TRIAL)..."
bash scripts/spin_per_trial_stacks.sh up "$MODEL" "$TRIAL" 2>&1 | tee -a "$LOG" | tail -5

# Resolve memobase URL
URLS="$(bash scripts/spin_per_trial_stacks.sh urls "$MODEL" "$TRIAL")"
MB_URL="$(awk -F= '/MEMOBASE_BASE_URL/{print $2}' <<<"$URLS")"
[[ -n "$MB_URL" ]] || { log "ERROR no MEMOBASE_BASE_URL"; exit 2; }
log "memobase: $MB_URL"

log "waiting for memobase healthcheck..."
deadline=$((SECONDS + 240))
while ! curl -fsS "${MB_URL}/api/v1/healthcheck" >/dev/null 2>&1; do
  [[ $SECONDS -ge $deadline ]] && { log "  ERROR memobase not healthy"; exit 1; }
  sleep 3
done
log "  memobase ready"

# ---- 3. start hw_probe sidecar
log "starting hw_probe → $HW_CSV"
python3 scripts/hw_probe.py \
  --outfile "$HW_CSV" --interval-ms 100 \
  ${HW_PROBE_SKIP_DOCKER:+--skip-docker} \
  > "$HW_LOG" 2>&1 &
HW_PID=$!
sleep 0.5

# ---- 4. run build_memory_cache for memobase
log "running build_memory_cache (memobase, A_paired, N_EGOS=$N_EGOS)"
T_START=$SECONDS
export MEMOBASE_BASE_URL="$MB_URL"
export MEMOBASE_API_TOKEN="${MEMOBASE_API_TOKEN:-secret}"
.venv/bin/python scripts/reproduce/build_memory_cache.py \
  --system memobase \
  --run-dir "$RUN_DIR" \
  --extractor-model "${MODEL_NAME[$MODEL]}" \
  --extractor-endpoint "http://localhost:${SGLANG_PORT}" \
  --extractor-api-key EMPTY \
  --extractor-config A_paired \
  --output "$CACHE_OUT" \
  --trial-seed 1002 \
  --user-limit "$N_EGOS" \
  --concurrency 1 \
  >> "$LOG" 2>&1
RC=$?
ELAPSED=$((SECONDS - T_START))
log "build_memory_cache rc=$RC elapsed=${ELAPSED}s"

# ---- 5. report from summary.json
SUMMARY="${CACHE_OUT%.jsonl}.summary.json"
if [[ -f "$SUMMARY" ]]; then
  log "summary:"
  python3 -c "
import json
d = json.load(open('$SUMMARY'))
for k in ['status','elapsed_seconds','n_ingest_batches','n_rows_written','n_errored','n_ego_agents','n_days','extractor_model']:
    if k in d: print(f'  {k}: {d[k]}')
"  | tee -a "$LOG"
fi

# ---- 6. stop ingest hw_probe, then run answer phase on same sglang+reader
log "stopping ingest hw_probe"
if [[ -n "${HW_PID:-}" ]]; then
  kill -TERM "$HW_PID" 2>/dev/null || true
  wait "$HW_PID" 2>/dev/null || true
  HW_PID=""
  if [[ -s "$HW_CSV" ]]; then
    python3 scripts/aggregate_hw_probe_simple.py \
      --csv "$HW_CSV" --outfile "$HW_AGG" \
      >> "$HW_LOG" 2>&1 || log "  ! hw_aggregate failed"
  fi
fi

log "running answer phase (memobase × $MODEL) using existing sglang + memobase stack"
BACKEND=memobase MODEL="$MODEL" \
  SGLANG_URL="http://localhost:${SGLANG_PORT}" \
  OUT_ROOT="${OUT_ROOT/_ingest/_answer}" \
  bash scripts/run_latency_answer.sh \
  >> "$LOG" 2>&1 || log "  ! answer phase failed"

log "DONE"
