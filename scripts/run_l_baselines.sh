#!/usr/bin/env bash
# Sequentially run baseline backends × s2/s3/s4 across reader models on the
# memarena-L benchmark. Companion to run_l_all_models.sh — that script
# handles memobase/memos (which need per-cell isolated stacks). This one
# handles backends that talk only to sglang (no postgres/qdrant/neo4j).
#
# Default backends: vanilla, oracle, baseline_simplerag (RAG).
# Default models : 0_6b, llama3b, 7b, 8b, 32b
# Trials         : s2, s3, s4
#
# For each model:
#   - Stop previous model's sglang
#   - Start this model's sglang with DP=2 across GPUs 1,2 (max_running=512)
#   - Launch one matrix instance per (backend, trial) cell IN PARALLEL
#     (no shared state to contend over — only sglang)
#   - Wait for all cells to finish
#   - Run per-cell LLM judges in parallel (concurrency=512 each)
#
# Models loop sequentially. Output dir per model:
#   out/accuracy_memarena_l_baselines_${model}/
#
# Required env: OPENROUTER_API_KEY
# Optional env: MODELS=...space-separated...   BACKENDS=...comma-separated...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${OPENROUTER_API_KEY:-}" && -f .env ]]; then set -a; . .env; set +a; fi
[[ -n "${OPENROUTER_API_KEY:-}" ]] || { echo "OPENROUTER_API_KEY missing" >&2; exit 2; }

MODELS=(${MODELS:-0_6b llama3b 7b 8b 32b})
TRIALS=(${TRIALS:-s2 s3 s4})
BACKENDS_CSV="${BACKENDS:-vanilla,oracle,baseline_simplerag}"
IFS=',' read -r -a BACKENDS <<<"$BACKENDS_CSV"
RUN_DIR="${RUN_DIR:-data/benchmark}"
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

log() { echo "[run-l-baselines $(date -u +%H:%M:%S)] $*"; }

# ---- cleanup all sglang containers (called at exit / on signal) -----
cleanup_all_sglang() {
  for tag in 0_6b llama3b 7b 8b 32b; do
    docker rm -f "memarena-sglang-${tag}" >/dev/null 2>&1 || true
  done
  log "cleanup: removed any remaining memarena-sglang-* containers"
}
trap cleanup_all_sglang EXIT INT TERM

# ---- sglang DP=2 for current model on GPUs 1,2 ---------------------
start_sglang() {
  local model="$1"
  log "starting sglang $model DP=2 on GPUs 1,2..."
  # Stop any other sglang containers (only one model at a time)
  for tag in 0_6b llama3b 7b 8b 32b; do
    [[ "$tag" == "$model" ]] && continue
    docker rm -f "memarena-sglang-${tag}" 2>/dev/null || true
  done
  docker rm -f "memarena-sglang-${model}" 2>/dev/null || true

  local mname="${MODEL_NAME[$model]}"
  local port="${MODEL_PORT[$model]}"
  local pip_install="protobuf sentencepiece"
  # NOTE: vllm==0.7.2 was previously installed for 32b AWQ but it upgrades torch
  # and breaks sgl_kernel C++ ABI on H200 (SM90). sglang v0.5.9 supports AWQ
  # natively, so skip the extra vllm install.

  docker run -d --name "memarena-sglang-${model}" \
    --gpus '"device=1,2"' --ipc=host --shm-size 32g --restart no \
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
      --host 0.0.0.0 --port "$port" --tp 1 --dp 2 \
      --context-length 16384 --mem-fraction-static 0.85 --max-running-requests 512 >/dev/null

  log "waiting for sglang $model on :$port..."
  until curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; do sleep 5; done
  log "sglang $model ready"
}

# ---- Per-model models.tsv (no memobase/memos URLs since baselines don't need them)
write_models_tsv() {
  local model="$1"; local out_dir="$2"
  mkdir -p "$out_dir/_models_files"
  local tsv="$out_dir/_models_files/${model}.tsv"
  printf '%s|%s|http://localhost:%s\n' \
    "$model" "${MODEL_NAME[$model]}" "${MODEL_PORT[$model]}" >"$tsv"
}

# ---- Launch all (backend × trial) cells in parallel ---------------
launch_cells() {
  local model="$1"; local out_dir="$2"
  mkdir -p "$out_dir/wrapper_logs"
  log "launching ${#BACKENDS[@]}×${#TRIALS[@]} = $((${#BACKENDS[@]} * ${#TRIALS[@]})) cells for $model..."
  local pids=() labels=()
  for backend in "${BACKENDS[@]}"; do for t in "${TRIALS[@]}"; do
    local label="${backend}_${model}_${t}"
    local lf="$out_dir/wrapper_logs/${label}.log"
    bash "$SCRIPT_DIR/run_eval_matrix.sh" \
      --run-dir "$RUN_DIR" --out-dir "$out_dir" \
      --models-file "$out_dir/_models_files/${model}.tsv" \
      --backends "$backend" --trials "$t" \
      --judge-preset remote \
      --remote-judge-model "$JUDGE_MODEL" \
      --remote-judge-endpoint "$JUDGE_ENDPOINT" \
      --judge-api-key "$OPENROUTER_API_KEY" \
      --answer-concurrency 64 \
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

# ---- Per-cell judge with high concurrency -------------------------
run_judges() {
  local model="$1"; local out_dir="$2"
  log "running judges for $model (concurrency=$JUDGE_CONCURRENCY)..."
  local pids=()
  for backend in "${BACKENDS[@]}"; do for t in "${TRIALS[@]}"; do
    # vanilla/oracle live in eval_results_${t}/${backend}/, not memory_cache
    local subdir
    case "$backend" in
      vanilla|oracle|inmem) subdir="$backend" ;;
      memory_cache|memobase|memos) subdir="memory_cache" ;;
      baseline_simplerag|baseline_session) subdir="$backend" ;;
      *) subdir="$backend" ;;
    esac
    local ans="$out_dir/eval_results_${t}/${subdir}/answer_results_${backend}_${model}_${t}.json"
    [[ -f "$ans" ]] || { log "  skip ${backend} ${t}: no answer file at $ans"; continue; }
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
log "backends: ${BACKENDS[*]}"
log "trials:   ${TRIALS[*]}"
log "run dir:  $RUN_DIR"

for model in "${MODELS[@]}"; do
  [[ -n "${MODEL_NAME[$model]:-}" ]] || { log "skip unknown model $model"; continue; }
  log "================ MODEL: $model ================"

  start_sglang "$model"

  out_dir="out/accuracy_memarena_l_baselines_${model}"
  rm -rf "$out_dir" || true
  mkdir -p "$out_dir"
  write_models_tsv "$model" "$out_dir"

  launch_cells "$model" "$out_dir"
  run_judges  "$model" "$out_dir"

  log "================ $model DONE → $out_dir ================"
done

log "all models complete: ${MODELS[*]}"
