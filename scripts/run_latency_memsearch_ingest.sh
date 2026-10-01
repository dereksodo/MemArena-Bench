#!/usr/bin/env bash
# run_latency_memsearch_ingest.sh — measure memsearch ingest cost (single-shot,
# reader-independent: memsearch.add() is "No LLM call on this path — chunking
# is deterministic, only ollama embeddings are computed").
#
# What gets measured:
#   - wall-clock ingest time (8 ego × 15 day, alphabetical-first 8 egos)
#   - hw_probe sidecar (GPU power/temp/util while ollama embeds)
#   - n_ingest_batches from memcache .summary.json
#
# Output:
#   out/latency2_ingest/memsearch_0_6b_s2/
#       memcache_memsearch_A_paired_0_6b_s2.jsonl  (one row per QA, with
#                                                   retrieved memories pre-cached)
#       memcache_memsearch_A_paired_0_6b_s2.summary.json (elapsed_seconds, n_batches)
#       hw/hw_probe_memsearch_0_6b_s2_ingest.csv
#       hw/hw_agg_memsearch_0_6b_s2_ingest.json
#       logs/ingest_memsearch_0_6b_s2.log
#
# Why MODEL=0_6b in filename: memsearch ingest does NOT use the reader, so
# this run is reader-independent. We tag the output with 0_6b purely for
# naming convention. The same milvus index serves all 5 readers in answer
# phase (after the answer-phase wrapper points at this milvus dir).
#
# Usage:
#   bash scripts/run_latency_memsearch_ingest.sh

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
[[ -z "${OPENROUTER_API_KEY:-}" && -f .env ]] && { set -a; . .env; set +a; }

MODEL="${MODEL:-0_6b}"
TRIAL="${TRIAL:-s2}"
N_EGOS="${N_EGOS:-8}"
RUN_DIR="${RUN_DIR:-data/benchmark}"
OUT_ROOT="${OUT_ROOT:-out/latency2_ingest}"
NS="memsearch_${MODEL}_${TRIAL}"
OUT_DIR="${OUT_ROOT}/${NS}"
SGLANG_PORT="${SGLANG_PORT:-17000}"
GPU_DEVICES="${GPU_DEVICES:-0}"
SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.5}"   # leave room for ollama BGE-M3
SERVICES_ROOT="${MEMARENA_MEMORY_SERVICES_ROOT:-$HOME/memarena-memory-services}"

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/hw"
LOG="$OUT_DIR/logs/ingest_memsearch_${MODEL}_${TRIAL}.log"
HW_CSV="$OUT_DIR/hw/hw_probe_memsearch_${MODEL}_${TRIAL}_ingest.csv"
HW_AGG="$OUT_DIR/hw/hw_agg_memsearch_${MODEL}_${TRIAL}_ingest.json"
HW_LOG="$OUT_DIR/logs/hw_probe_memsearch_${MODEL}_${TRIAL}_ingest.log"
CACHE_OUT="$OUT_DIR/memcache_memsearch_A_paired_${MODEL}_${TRIAL}.jsonl"

declare -A MODEL_NAME=(
  [0_6b]="Qwen/Qwen3-0.6B"
  [llama3b]="meta-llama/Llama-3.2-3B-Instruct"
  [7b]="mistralai/Mistral-7B-Instruct-v0.3"
  [8b]="Qwen/Qwen3-8B"
  [32b]="Qwen/Qwen3-32B-AWQ"
)

log() { echo "[memsearch-ingest $(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

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
  docker rm -f lat-spark-sglang 2>/dev/null || true
}
trap cleanup EXIT

# ---- 1. start sglang at half mem-fraction (memsearch ingest needs reader endpoint
# ----    technically — build_memory_cache.py constructs adapter with extractor
# ----    info even when not used. Plus answer phase later will use sglang.)
log "starting sglang (mem-fraction=$SGLANG_MEM_FRACTION) — memsearch ingest only embeds, but adapter init expects an endpoint"
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

# ---- 2. start hw_probe sidecar
log "starting hw_probe → $HW_CSV"
python3 scripts/hw_probe.py \
  --outfile "$HW_CSV" --interval-ms 100 \
  ${HW_PROBE_SKIP_DOCKER:+--skip-docker} \
  > "$HW_LOG" 2>&1 &
HW_PID=$!
sleep 0.5

# ---- 3. run build_memory_cache for memsearch
log "running build_memory_cache (memsearch, N_EGOS=$N_EGOS)"
T_START=$SECONDS
.venv/bin/python scripts/reproduce/build_memory_cache.py \
  --system memsearch \
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

# ---- 4. report from summary.json
SUMMARY="${CACHE_OUT%.jsonl}.summary.json"
if [[ -f "$SUMMARY" ]]; then
  log "summary:"
  python3 -c "
import json
d = json.load(open('$SUMMARY'))
for k in ['status','elapsed_seconds','n_ingest_batches','n_rows_written','n_errored','n_ego_agents','n_days','extractor_model','extractor_endpoint_label']:
    if k in d: print(f'  {k}: {d[k]}')
"  | tee -a "$LOG"
fi

# ---- 5. stop ingest hw_probe, then run answer phase on same sglang+reader
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

log "running answer phase (memsearch × $MODEL) using existing sglang"
BACKEND=memsearch MODEL="$MODEL" \
  SGLANG_URL="http://localhost:${SGLANG_PORT}" \
  OUT_ROOT="${OUT_ROOT/_ingest/_answer}" \
  bash scripts/run_latency_answer.sh \
  >> "$LOG" 2>&1 || log "  ! answer phase failed"

log "DONE"
