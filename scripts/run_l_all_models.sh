#!/usr/bin/env bash
# Sequentially run memobase + memos × s2/s3/s4 cells across reader models
# (default: 0_6b, 7b, 8b, 32b — llama3b is skipped since it's already done)
# on the memarena-L benchmark.
#
# For each model:
#   - sglang restarted to use the model with DP=8 across all 8 GPUs
#   - 6 dedicated ollama containers (one per cell, on GPUs 0-5,
#     OLLAMA_NUM_PARALLEL=2)
#   - 6 isolated memobase + memos compose stacks (one per (backend, trial))
#   - per-cell memobase config.yaml override pointing at that cell's ollama
#   - per-cell memos compose override forcing that cell's OLLAMA_API_BASE
#   - 6 build_memory_cache + answer cells run in parallel
#   - 6 LLM judges (concurrency 512 each) run in parallel after build done
#   - Tear down stacks + ollamas before moving to the next model
#
# Models loop sequentially (one model active at a time). Within a model
# all 6 cells run concurrently. Output dir per model:
#   out/accuracy_memarena_l_${model}/
#
# Ports (reused across models since models are sequential):
#   memobase api    18120 + trial_idx (s2=2, s3=3, s4=4)
#   memobase db     19120 + trial_idx
#   memobase redis  19130 + trial_idx
#   memos api       18130 + trial_idx
#   ollama          11440 + cell_idx (memobase_s2=0..memos_s4=5)
#
# Required env: OPENROUTER_API_KEY (for the remote judge).
# Optional env: MODELS=...space-separated...  RUN_DIR=...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${OPENROUTER_API_KEY:-}" && -f .env ]]; then set -a; . .env; set +a; fi
[[ -n "${OPENROUTER_API_KEY:-}" ]] || { echo "OPENROUTER_API_KEY missing" >&2; exit 2; }

MODELS=(${MODELS:-0_6b 7b 8b 32b})
TRIALS=(s2 s3 s4)
BACKENDS=(${BACKENDS:-memobase memos})
RUN_DIR="${RUN_DIR:-data/benchmark}"
SERVICES_ROOT="${MEMARENA_MEMORY_SERVICES_ROOT:-$HOME/memarena-memory-services}"
JUDGE_MODEL="${REMOTE_JUDGE_MODEL:-openai/gpt-4o-mini-2024-07-18}"
JUDGE_ENDPOINT="${REMOTE_JUDGE_ENDPOINT:-https://openrouter.ai/api/v1}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-512}"

declare -A MODEL_NAME MODEL_PORT
MODEL_NAME[0_6b]="Qwen/Qwen3-0.6B"
MODEL_NAME[llama3b]="meta-llama/Llama-3.2-3B-Instruct"
MODEL_NAME[7b]="mistralai/Mistral-7B-Instruct-v0.3"
MODEL_NAME[8b]="Qwen/Qwen3-8B"
MODEL_NAME[32b]="Qwen/Qwen3-32B-AWQ"
MODEL_PORT[0_6b]=16000
MODEL_PORT[llama3b]=16001
MODEL_PORT[7b]=16002
MODEL_PORT[8b]=16003
MODEL_PORT[32b]=16004

log() { echo "[run-l-all $(date -u +%H:%M:%S)] $*"; }

trial_idx() { case "$1" in s2) echo 2 ;; s3) echo 3 ;; s4) echo 4 ;; *) echo 0 ;; esac; }
cell_idx() {
  case "$1_$2" in
    memobase_s2)  echo 0 ;; memobase_s3)  echo 1 ;; memobase_s4)  echo 2 ;;
    memos_s2)     echo 3 ;; memos_s3)     echo 4 ;; memos_s4)     echo 5 ;;
    memsearch_s2) echo 6 ;; memsearch_s3) echo 7 ;; memsearch_s4) echo 8 ;;
  esac
}
ports_for_trial() {
  local t="$1"; local i; i="$(trial_idx "$t")"
  MB_API=$((18120 + i)); MB_DB=$((19120 + i))
  MB_REDIS=$((19130 + i)); MOS_API=$((18130 + i))
}
ollama_port_for_cell() { echo $((11440 + $(cell_idx "$1" "$2"))); }
ollama_gpu_for_cell()  { cell_idx "$1" "$2"; }

# ---- teardown ------------------------------------------------------
# Aggressive cleanup: removes ALL memarena containers (any model/trial),
# every per-cell ollama, every sglang container. Runs before the loop
# (initial cleanup) AND after each model finishes (so the next model
# starts from a fully clean state — no leftover stacks, no GPU memory
# held by the previous model's sglang).
teardown_everything() {
  log "tearing down ALL memarena docker containers..."
  pkill -f "build_memory_cache" 2>/dev/null || true
  pkill -f "run_eval_matrix"   2>/dev/null || true
  pkill -f "scripts/llmjudge"  2>/dev/null || true
  sleep 2

  # Every memarena_* compose stack (memobase + memos for any model/trial)
  local stack_ids
  stack_ids="$(docker ps -aq --filter 'name=memarena_' 2>/dev/null || true)"
  [[ -n "$stack_ids" ]] && docker rm -f $stack_ids >/dev/null 2>&1 || true

  # Every sglang container (any model tag)
  for tag in 0_6b llama3b 7b 8b 32b; do
    docker rm -f "memarena-sglang-${tag}" >/dev/null 2>&1 || true
  done

  # Every per-cell ollama
  local ollama_ids
  ollama_ids="$(docker ps -aq --filter 'name=ollama-' 2>/dev/null || true)"
  [[ -n "$ollama_ids" ]] && docker rm -f $ollama_ids >/dev/null 2>&1 || true

  # Compose networks linger after `docker rm -f`. Each per-cell stack
  # creates its own /24 from Docker's default address pool, so 30+
  # networks across a 5-model sweep exhausts the pool and the next
  # `docker compose up` fails with "all predefined address pools have
  # been fully subnetted". Reap them here too.
  local nets
  nets="$(docker network ls --format '{{.Name}}' 2>/dev/null | grep '^memarena_' || true)"
  if [[ -n "$nets" ]]; then
    echo "$nets" | xargs -r docker network rm >/dev/null 2>&1 || true
  fi

  # Named volumes (neo4j data, qdrant data, redis dumps) persist across
  # `docker rm -f` of their containers — without this nuke, every run
  # accumulates ~12 volumes per (model, trial, backend) cell. After a
  # few sweeps, that exhausts disk AND lets a fresh `docker compose up`
  # silently re-mount stale neo4j data, which kills memos-server's
  # MATCH (n:Memory) queries with O(N) full scans on accumulated nodes.
  local vols
  vols="$(docker volume ls --format '{{.Name}}' 2>/dev/null | grep -E '^(memarena|ollama)' || true)"
  if [[ -n "$vols" ]]; then
    echo "$vols" | xargs -r docker volume rm -f >/dev/null 2>&1 || true
  fi

  # build_memory_cache scratch (mem0 qdrant + sqlite history). Build
  # cleans its own per-run-tag entries but leaves the parent dirs
  # behind across runs.
  rm -rf /tmp/mem0_qdrant_memarena /tmp/mem0_history_memcache_*.db 2>/dev/null || true

  log "teardown complete"
}

# ---- sglang DP=8 ---------------------------------------------------
start_sglang() {
  local model="$1"
  log "starting sglang $model DP=8 on GPUs 0-7..."
  docker rm -f "memarena-sglang-${model}" 2>/dev/null || true
  local mname="${MODEL_NAME[$model]}"
  local port="${MODEL_PORT[$model]}"
  local pip_install="protobuf sentencepiece"
  [[ "$model" == "32b" ]] && pip_install="$pip_install vllm==0.7.2"

  docker run -d --name "memarena-sglang-${model}" \
    --gpus '"device=0,1,2,3,4,5,6,7"' --ipc=host --shm-size 32g --restart no \
    -p "${port}:${port}" \
    -v "${HOME}/models:/models:ro" \
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
    --env "SGLANG_PIP_INSTALL=${pip_install}" \
    lmsysorg/sglang:latest \
    bash -lc 'set -euo pipefail
if [[ -n "${SGLANG_PIP_INSTALL:-}" ]]; then python3 -m pip install --no-cache-dir ${SGLANG_PIP_INSTALL}; fi
exec "$@"
' bash python3 -m sglang.launch_server \
      --model-path "/models/${model}" --served-model-name "$mname" \
      --host 0.0.0.0 --port "$port" --tp 1 --dp 8 \
      --context-length 16384 --mem-fraction-static 0.85 --max-running-requests 128 >/dev/null

  log "waiting for sglang $model on :$port..."
  until curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; do sleep 5; done
  log "sglang $model ready"
}

# ---- per-cell ollamas (only for backends that need extractor embed) -----
start_ollamas() {
  # Only memobase + memos call ollama at ingest/extract time. memsearch
  # does its own embedding during build (a separate ollama-memsearch
  # spun up by the Phase 1 pre-build), and the answer-phase chain run
  # only reads pre-built memcache jsonl, so no per-cell ollama needed.
  local need_ollama=()
  for b in "${BACKENDS[@]}"; do
    [[ "$b" == "memobase" || "$b" == "memos" ]] && need_ollama+=("$b")
  done
  if [[ "${#need_ollama[@]}" -eq 0 ]]; then
    log "no per-cell ollama needed for backends: ${BACKENDS[*]}"
    return 0
  fi
  log "starting ${#need_ollama[@]} × ${#TRIALS[@]} per-cell ollamas (NUM_PARALLEL=2)"
  for backend in "${need_ollama[@]}"; do for t in "${TRIALS[@]}"; do
    local name="ollama-${backend}_${t}"
    local port; port="$(ollama_port_for_cell "$backend" "$t")"
    local gpu;  gpu="$(ollama_gpu_for_cell "$backend" "$t")"
    docker rm -f "$name" 2>/dev/null || true
    docker run -d --name "$name" --restart unless-stopped \
      --gpus "\"device=${gpu}\"" -e OLLAMA_NUM_PARALLEL=2 \
      -p "${port}:11434" -v "ollama_data:/root/.ollama" \
      ollama/ollama >/dev/null
  done; done
  sleep 8
  for backend in "${need_ollama[@]}"; do for t in "${TRIALS[@]}"; do
    timeout 120 docker exec "ollama-${backend}_${t}" ollama pull nomic-embed-text:latest 2>&1 | tail -1
  done; done
  log "ollamas ready"
}

# ---- per-cell config + override files -----------------------------
write_cell_overrides() {
  local model="$1"
  local mb_src="$SERVICES_ROOT/$model/memobase/src/server"
  mkdir -p /tmp/memarena_cells

  for t in "${TRIALS[@]}"; do
    local mb_ollama; mb_ollama="$(ollama_port_for_cell memobase "$t")"
    local mos_ollama; mos_ollama="$(ollama_port_for_cell memos "$t")"

    local mb_cfg="/tmp/memarena_cells/${model}_memobase_${t}_config.yaml"
    cp "$mb_src/api/config.yaml" "$mb_cfg"
    sed -i "s|^embedding_base_url:.*|embedding_base_url: http://host.docker.internal:${mb_ollama}|" "$mb_cfg"
    cat > "/tmp/memarena_cells/${model}_memobase_${t}_override.yml" <<EOF
services:
  memobase-server-api:
    volumes:
      - $mb_cfg:/app/config.yaml
EOF
    cat > "/tmp/memarena_cells/${model}_memos_${t}_override.yml" <<EOF
services:
  memos:
    environment:
      OLLAMA_API_BASE: "http://host.docker.internal:${mos_ollama}"
EOF
  done
}

# ---- 6 stacks ------------------------------------------------------
start_stacks() {
  local model="$1"
  local mb_src="$SERVICES_ROOT/$model/memobase/src/server"
  local mos_src="$SERVICES_ROOT/$model/MemOS"

  local want_mb=0 want_mos=0
  for b in "${BACKENDS[@]}"; do
    [[ "$b" == "memobase" ]] && want_mb=1
    [[ "$b" == "memos" ]] && want_mos=1
  done

  # Validate service dirs only for backends that need them. memsearch
  # has no compose stack — pure file/embed in-process — so we skip the
  # SERVICES_ROOT check entirely when neither memobase nor memos is in
  # BACKENDS.
  if [[ "$want_mb" -eq 1 || "$want_mos" -eq 1 ]]; then
    [[ -d "$mb_src" && -d "$mos_src/docker" ]] || { log "ERROR: missing $model service dirs"; return 2; }
    sudo -n rm -rf "$mb_src"/db_s*/data "$mb_src"/db_s*/redis 2>/dev/null || true
  fi

  log "spinning stacks for $model (memobase=$want_mb memos=$want_mos)..."
  for t in "${TRIALS[@]}"; do
    ports_for_trial "$t"
    if [[ "$want_mb" -eq 1 ]]; then
      ( cd "$mb_src" && \
        COMPOSE_PROJECT_NAME="memarena_${model}_${t}_memobase" \
        DATABASE_NAME=memobase DATABASE_USER=memobase DATABASE_PASSWORD=memobase \
        DATABASE_LOCATION="./db_${t}/data" \
        REDIS_PASSWORD=memobase REDIS_LOCATION="./db_${t}/redis" \
        DATABASE_EXPORT_PORT="$MB_DB" REDIS_EXPORT_PORT="$MB_REDIS" \
        API_EXPORT_PORT="$MB_API" \
        API_HOSTS="http://0.0.0.0:${MB_API},http://localhost:${MB_API}" \
        USE_CORS=false PROJECT_ID=memobase_dev ACCESS_TOKEN=secret \
        docker compose -f docker-compose.yml \
          -f "/tmp/memarena_cells/${model}_memobase_${t}_override.yml" \
          up -d 2>&1 | tail -1 ) &
    fi
    if [[ "$want_mos" -eq 1 ]]; then
      ( cd "$mos_src/docker" && \
        COMPOSE_PROJECT_NAME="memarena_${model}_${t}_memos" \
        MEMOS_EXPORT_PORT="$MOS_API" \
        docker compose -f docker-compose.yml \
          -f "/tmp/memarena_cells/${model}_memos_${t}_override.yml" \
          up -d 2>&1 | tail -1 ) &
    fi
  done
  wait

  if [[ "$want_mb" -eq 1 ]]; then
    log "waiting for memobase healthchecks..."
    for t in "${TRIALS[@]}"; do
      ports_for_trial "$t"
      local deadline=$((SECONDS + 240))
      while ! curl -fsS "http://127.0.0.1:${MB_API}/api/v1/healthcheck" >/dev/null 2>&1; do
        [[ "$SECONDS" -ge "$deadline" ]] && { log "WARNING: memobase $t not healthy"; break; }
        sleep 3
      done
      log "  ✓ memobase $t (:$MB_API)"
    done
  fi
}

write_models_tsv() {
  local model="$1"; local out_dir="$2"
  mkdir -p "$out_dir/_models_files"
  for t in "${TRIALS[@]}"; do
    ports_for_trial "$t"
    printf '%s|%s|http://localhost:%s|http://localhost:%s|http://localhost:%s\n' \
      "$model" "${MODEL_NAME[$model]}" "${MODEL_PORT[$model]}" "$MB_API" "$MOS_API" \
      > "$out_dir/_models_files/${model}_${t}.tsv"
  done
}

launch_cells() {
  local model="$1"; local out_dir="$2"
  mkdir -p "$out_dir/wrapper_logs"
  log "launching 6 cells for $model..."
  local pids=() labels=()
  for backend in "${BACKENDS[@]}"; do for t in "${TRIALS[@]}"; do
    local label="${backend}_${model}_${t}"
    local lf="$out_dir/wrapper_logs/${label}.log"
    bash "$SCRIPT_DIR/run_eval_matrix.sh" \
      --run-dir "$RUN_DIR" --out-dir "$out_dir" \
      --models-file "$out_dir/_models_files/${model}_${t}.tsv" \
      --backends "$backend" --trials "$t" \
      --judge-preset remote \
      --remote-judge-model "$JUDGE_MODEL" \
      --remote-judge-endpoint "$JUDGE_ENDPOINT" \
      --judge-api-key "$OPENROUTER_API_KEY" \
      --force >"$lf" 2>&1 &
    pids+=("$!"); labels+=("$label")
    log "  → $label (pid=$!)"
  done; done

  log "waiting for ${#pids[@]} cells..."
  local failed=0
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      log "FAILED: ${labels[$i]}"; failed=$((failed + 1))
    fi
  done
  log "cells: $((${#pids[@]} - failed))/${#pids[@]} succeeded for $model"
}

run_judges() {
  local model="$1"; local out_dir="$2"
  log "running 6 judges for $model (concurrency=$JUDGE_CONCURRENCY)..."
  local pids=()
  for backend in "${BACKENDS[@]}"; do for t in "${TRIALS[@]}"; do
    local ans="$out_dir/eval_results_${t}/memory_cache/answer_results_${backend}_${model}_${t}.json"
    [[ -f "$ans" ]] || { log "  skip ${backend} ${t}: no answer file"; continue; }
    .venv/bin/python scripts/llmjudge.py \
      --run-dir "$RUN_DIR" --answer-path "$ans" \
      --judge-preset remote --judge-model "$JUDGE_MODEL" \
      --judge-endpoint "$JUDGE_ENDPOINT" --judge-api-key "$OPENROUTER_API_KEY" \
      --concurrency "$JUDGE_CONCURRENCY" --force \
      >"/tmp/judge_${backend}_${model}_${t}.log" 2>&1 &
    pids+=("$!")
  done; done
  for p in "${pids[@]}"; do wait "$p" || true; done
  log "judges done for $model"
}

# ============= main loop ==============
log "models in order: ${MODELS[*]}"
log "trials per model: ${TRIALS[*]}"
log "backends: ${BACKENDS[*]}"
log "run dir: $RUN_DIR"

# Initial cleanup: nuke any leftover containers from previous runs
teardown_everything

for model in "${MODELS[@]}"; do
  [[ -n "${MODEL_NAME[$model]:-}" ]] || { log "skip unknown model $model"; continue; }
  log "================ MODEL: $model ================"

  # Per-model concurrency overrides (memos/memobase only). Larger readers
  # spend much longer per /product/add (decode time scales with model
  # size), so smaller chunks + fewer in-flight namespaces keep the
  # memos-server uvicorn loop from accumulating transient `Can not write
  # request body` failures.
  if [[ "$model" == "32b" ]]; then
    export MEMOS_MAX_BATCH="${MEMOS_MAX_BATCH_32B:-8}"
    export MEMORY_CACHE_CONCURRENCY="${MEMORY_CACHE_CONCURRENCY_32B:-16}"
    log "32b overrides: MEMOS_MAX_BATCH=$MEMOS_MAX_BATCH MEMORY_CACHE_CONCURRENCY=$MEMORY_CACHE_CONCURRENCY"
  fi

  start_sglang "$model"
  start_ollamas
  write_cell_overrides "$model"
  start_stacks "$model"

  out_dir="out/accuracy_memarena_l_${model}"
  if [[ -n "${SAFE_NO_RM:-}" ]]; then
    log "SAFE_NO_RM=1 set — keeping existing files in $out_dir"
  else
    rm -rf "$out_dir" || true
  fi
  mkdir -p "$out_dir"
  write_models_tsv "$model" "$out_dir"

  launch_cells "$model" "$out_dir"
  run_judges  "$model" "$out_dir"

  log "================ $model DONE → $out_dir ================"

  # Always teardown between models so the next one starts clean.
  teardown_everything
done

log "all models complete: ${MODELS[*]}"
