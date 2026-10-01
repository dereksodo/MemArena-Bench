#!/usr/bin/env bash
# Parallel memos build for 8b + 32b on a single 8-GPU H100.
#
# Layout (avoids GPU and port conflicts):
#   sglang-8b       GPU 0,1     DP=2 TP=1  port 16003   Qwen3-8B
#   sglang-32b      GPU 2-7     DP=3 TP=2  port 16004   Qwen3-32B-AWQ
#   ollama-memos_s{2,3,4}        CPU       11443-11445  shared by both
#   memos-server-8b-{s2,s3,s4}             18132-18134  (MOS_API)
#   memos-server-32b-{s2,s3,s4}            18142-18144  (MOS_API offset +10)
#
# Adapter / chain-compatible behavior:
#   - Sequential per-namespace chunks (already in memos_adapter, commit 6cd161a)
#   - SAFE_NO_RM equivalent: never wipes any existing out/ files
#   - Volumes/networks/scratch nuked on entry to avoid stale neo4j data
#
# Usage:
#   SAFE_NO_RM=1 bash scripts/run_l_parallel_8b_32b.sh

set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"

SERVICES_ROOT="${MEMARENA_MEMORY_SERVICES_ROOT:-$HOME/memarena-memory-services}"
[[ -d "$SERVICES_ROOT/8b/MemOS/docker" ]] || { echo "missing $SERVICES_ROOT/8b/MemOS/docker"; exit 2; }
[[ -d "$SERVICES_ROOT/32b/MemOS/docker" ]] || { echo "missing $SERVICES_ROOT/32b/MemOS/docker"; exit 2; }

# Load OPENROUTER_API_KEY for judge step
[[ -f .env ]] && { set -a; . .env; set +a; }
[[ -n "${OPENROUTER_API_KEY:-}" ]] || { echo "OPENROUTER_API_KEY required"; exit 2; }

CONCURRENCY="${MEMORY_CACHE_CONCURRENCY:-32}"
BATCH="${MEMOS_MAX_BATCH:-15}"
TRIALS=(s2 s3 s4)
log() { echo "[parallel $(date -u +%H:%M:%S)] $*"; }

trial_idx() { case "$1" in s2) echo 2;; s3) echo 3;; s4) echo 4;; esac; }
trial_seed() { case "$1" in s2) echo 1002;; s3) echo 1003;; s4) echo 1004;; esac; }
mos_api_port_8b() { echo $((18130 + $(trial_idx "$1"))); }
mos_api_port_32b() { echo $((18140 + $(trial_idx "$1"))); }
ollama_port() { case "$1" in s2) echo 11443;; s3) echo 11444;; s4) echo 11445;; esac; }

# ---- Step 0: nuclear cleanup ---------------------------------------------
log "nuclear cleanup..."
docker ps -aq | xargs -r docker rm -f >/dev/null 2>&1 || true
docker volume ls --format '{{.Name}}' | grep -E '^(memarena|ollama)' | xargs -r docker volume rm -f >/dev/null 2>&1 || true
docker network ls --format '{{.Name}}' | grep '^memarena_' | xargs -r docker network rm >/dev/null 2>&1 || true
rm -rf /tmp/mem0_qdrant_memarena /tmp/mem0_history_memcache_*.db 2>/dev/null || true
sleep 2

# ---- Step 1: start sglang-8b on GPU 0,1 ----------------------------------
log "starting sglang-8b on GPU 0,1 (DP=2)"
docker run -d --name memarena-sglang-8b \
  --gpus '"device=0,1"' --ipc=host --shm-size 32g --restart no \
  -p 16003:16003 \
  -v "$HOME/models:/models:ro" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  --env "SGLANG_PIP_INSTALL=protobuf sentencepiece" \
  lmsysorg/sglang:latest \
  bash -lc 'set -euo pipefail
if [[ -n "${SGLANG_PIP_INSTALL:-}" ]]; then python3 -m pip install --no-cache-dir ${SGLANG_PIP_INSTALL}; fi
exec "$@"
' bash python3 -m sglang.launch_server \
    --model-path /models/8b --served-model-name Qwen/Qwen3-8B \
    --host 0.0.0.0 --port 16003 --tp 1 --dp 2 \
    --context-length 16384 --mem-fraction-static 0.85 --max-running-requests 64 >/dev/null

# ---- Step 2: start sglang-32b on GPU 2-7 ---------------------------------
log "starting sglang-32b on GPU 2-7 (DP=3 TP=2)"
docker run -d --name memarena-sglang-32b \
  --gpus '"device=2,3,4,5,6,7"' --ipc=host --shm-size 32g --restart no \
  -p 16004:16004 \
  -v "$HOME/models:/models:ro" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  --env "SGLANG_PIP_INSTALL=protobuf sentencepiece vllm==0.7.2" \
  lmsysorg/sglang:latest \
  bash -lc 'set -euo pipefail
if [[ -n "${SGLANG_PIP_INSTALL:-}" ]]; then python3 -m pip install --no-cache-dir ${SGLANG_PIP_INSTALL}; fi
exec "$@"
' bash python3 -m sglang.launch_server \
    --model-path /models/32b --served-model-name Qwen/Qwen3-32B-AWQ \
    --host 0.0.0.0 --port 16004 --tp 2 --dp 3 \
    --context-length 16384 --mem-fraction-static 0.85 --max-running-requests 96 >/dev/null

# ---- Step 3: wait for both sglangs ---------------------------------------
for url_name in "16003:8b" "16004:32b"; do
  port="${url_name%%:*}"; name="${url_name##*:}"
  log "waiting for sglang-$name on :$port..."
  waited=0
  until curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; do
    sleep 5; waited=$((waited+5))
    [[ $waited -gt 600 ]] && { log "TIMEOUT waiting for sglang-$name"; exit 3; }
  done
  log "sglang-$name READY (${waited}s)"
done

# ---- Step 4: start 3 shared ollama containers (CPU) ----------------------
log "starting 3 ollama-memos containers (shared 8b + 32b, CPU embedding)"
for t in "${TRIALS[@]}"; do
  port=$(ollama_port "$t")
  docker run -d --name "ollama-memos_$t" \
    -p "${port}:11434" \
    -v "ollama-memos_${t}_data:/root/.ollama" \
    --restart no \
    ollama/ollama:latest >/dev/null
done
sleep 3
log "pulling nomic-embed-text in each ollama..."
for t in "${TRIALS[@]}"; do
  docker exec "ollama-memos_$t" ollama pull nomic-embed-text:latest >/dev/null 2>&1 &
done
wait
log "ollamas ready"

# ---- Step 5: write per-cell override yml ---------------------------------
mkdir -p /tmp/memarena_cells
for model in 8b 32b; do
  for t in "${TRIALS[@]}"; do
    op=$(ollama_port "$t")
    cat > "/tmp/memarena_cells/${model}_memos_${t}_override.yml" <<EOF
services:
  memos:
    environment:
      OLLAMA_API_BASE: "http://host.docker.internal:${op}"
EOF
  done
done

# ---- Step 6: spin 6 memos compose stacks (3 per model) -------------------
log "spinning 6 memos compose stacks..."
for model in 8b 32b; do
  mos_src="$SERVICES_ROOT/$model/MemOS"
  for t in "${TRIALS[@]}"; do
    if [[ "$model" == "8b" ]]; then
      mos_api=$(mos_api_port_8b "$t")
    else
      mos_api=$(mos_api_port_32b "$t")
    fi
    ( cd "$mos_src/docker" && \
      COMPOSE_PROJECT_NAME="memarena_${model}_${t}_memos" \
      MEMOS_EXPORT_PORT="$mos_api" \
      docker compose -f docker-compose.yml \
        -f "/tmp/memarena_cells/${model}_memos_${t}_override.yml" \
        up -d 2>&1 | tail -1 ) &
  done
done
wait
log "stacks started; waiting 15s for neo4j init..."
sleep 15

# ---- Step 7: launch 6 build cells in parallel ---------------------------
log "launching 6 cells (3×8b + 3×32b) with CONCURRENCY=$CONCURRENCY BATCH=$BATCH..."
declare -A CELL_PIDS
for model in 8b 32b; do
  if [[ "$model" == "8b" ]]; then
    sglang_port=16003; sglang_model="Qwen/Qwen3-8B"; out_dir="out/accuracy_memarena_l_8b"
  else
    sglang_port=16004; sglang_model="Qwen/Qwen3-32B-AWQ"; out_dir="out/accuracy_memarena_l_32b"
  fi
  mkdir -p "$out_dir/wrapper_logs" "$out_dir/memory_cache/memos/$model"
  for t in "${TRIALS[@]}"; do
    if [[ "$model" == "8b" ]]; then
      mos_api=$(mos_api_port_8b "$t")
    else
      mos_api=$(mos_api_port_32b "$t")
    fi
    seed=$(trial_seed "$t")
    cache_path="$out_dir/memory_cache/memos/$model/$t/memcache_memos_A_paired_${model}_${t}.jsonl"
    mkdir -p "$(dirname "$cache_path")"
    log_path="$out_dir/wrapper_logs/memos_${model}_${t}.log"
    label="${model}_${t}"

    log "  → ${label} (sglang :$sglang_port, memos :$mos_api)"
    (
      MEMOS_BASE_URL="http://localhost:${mos_api}" \
      MEMOS_API_KEY=EMPTY \
      MEMOS_MAX_BATCH="$BATCH" \
      .venv/bin/python scripts/reproduce/build_memory_cache.py \
        --system memos --run-dir data/benchmark \
        --extractor-model "$sglang_model" \
        --extractor-endpoint "http://localhost:${sglang_port}" \
        --extractor-api-key EMPTY \
        --extractor-config A_paired \
        --output "$cache_path" \
        --trial-seed "$seed" \
        --concurrency "$CONCURRENCY" \
        > "$log_path" 2>&1 \
      && \
      .venv/bin/python scripts/run_accuracy.py \
        --run-dir data/benchmark --system memory_cache \
        --judge-preset remote \
        --sglang-url "http://localhost:${sglang_port}" \
        --model-name "$sglang_model" --model-tag "$model" \
        --trial-name "$t" --seed "$seed" \
        --namespace "memos_${model}_${t}" \
        --out-dir "$out_dir" --temperature 0.3 \
        --cache-path "$cache_path" \
        --expected-extractor "$sglang_model" \
        --expected-memory-system memos --expected-config A_paired \
        --answer-concurrency 32 --eval-concurrency 512 \
        --remote-judge-model openai/gpt-4o-mini-2024-07-18 \
        --remote-judge-endpoint https://openrouter.ai/api/v1 \
        --judge-api-key "$OPENROUTER_API_KEY" \
        --force \
        >> "$log_path" 2>&1
    ) &
    CELL_PIDS[$label]=$!
  done
done

log "waiting for ${#CELL_PIDS[@]} cells..."
ok=0; fail=0
for label in "${!CELL_PIDS[@]}"; do
  if wait "${CELL_PIDS[$label]}"; then
    log "  ✓ $label"; ok=$((ok+1))
  else
    log "  ✗ $label (rc=$?)"; fail=$((fail+1))
  fi
done

log "================ DONE ================"
log "result: ok=$ok fail=$fail / 6 cells"
log "artifacts:"
log "  out/accuracy_memarena_l_8b/memory_cache/memos/8b/{s2,s3,s4}/memcache_memos_A_paired_8b_*.jsonl"
log "  out/accuracy_memarena_l_32b/memory_cache/memos/32b/{s2,s3,s4}/memcache_memos_A_paired_32b_*.jsonl"
log "  out/accuracy_memarena_l_*/eval_results_*/memory_cache/evaluation_results_memos_*_judge_remote.json"
