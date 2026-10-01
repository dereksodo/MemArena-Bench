#!/usr/bin/env bash
# memarena-L ablation runner — independent of run_l_baselines.sh.
#
# Sweeps "ablation" backends (retrieval / context controls) × trials × readers,
# using the same per-model sglang lifecycle as run_l_baselines.sh:
#   - one DP=2 sglang on GPUs 1,2 per reader,
#   - all (backend, trial) cells launched in parallel for each reader,
#   - per-cell LLM judge after all cells of that reader finish.
#
# Default ablation backends (all native MemArena adapters; no external services):
#   oracle_with_distractors           : Oracle + recency-tail same-ego distractors
#   oracle_with_random_distractors    : Oracle + uniformly-random distractors
#   inmem_text_sessions               : BM25 + TEXT_SESSIONS prompt format
#   hybrid_bm25rerank                 : BM25 top-20 + bge-reranker-v2-m3 → top-K
#   dense_bge_m3                      : Pure dense retriever (BGE-M3)
#
# Output dir (one per model):
#   out/ablations_l_${model}/
#
# Required env: OPENROUTER_API_KEY
# Optional env:
#   MODELS=...space-separated tags...
#   BACKENDS=...comma-separated systems...
#   TRIALS=...space-separated, default s2 s3 s4...
#   JUDGE_CONCURRENCY=512
#
# This script is the canonical ablation entrypoint. It does NOT shell out to or
# depend on anything outside this repository.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${OPENROUTER_API_KEY:-}" && -f .env ]]; then set -a; . .env; set +a; fi
[[ -n "${OPENROUTER_API_KEY:-}" ]] || { echo "OPENROUTER_API_KEY missing" >&2; exit 2; }

MODELS=(${MODELS:-0_6b llama3b 7b 8b 32b})
TRIALS=(${TRIALS:-s2 s3 s4})
BACKENDS_CSV="${BACKENDS:-oracle_with_distractors,oracle_with_random_distractors,inmem_text_sessions}"
IFS=',' read -r -a BACKENDS <<<"$BACKENDS_CSV"
RUN_DIR="${RUN_DIR:-data/benchmark}"
JUDGE_MODEL="${REMOTE_JUDGE_MODEL:-openai/gpt-4o-mini-2024-07-18}"
JUDGE_ENDPOINT="${REMOTE_JUDGE_ENDPOINT:-https://openrouter.ai/api/v1}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-512}"
ANSWER_CONCURRENCY="${ANSWER_CONCURRENCY:-64}"
OUT_PREFIX="${OUT_PREFIX:-out/ablations_l_}"
TOP_K_OVERRIDE="${TOP_K:-}"
D6_ARM_OVERRIDE="${D6_ARM:-}"
EVAL_CONFIG_OVERRIDE="${EVAL_CONFIG:-}"

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

log() { echo "[run-l-ablations $(date -u +%H:%M:%S)] $*"; }

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
  for tag in 0_6b llama3b 7b 8b 32b; do
    [[ "$tag" == "$model" ]] && continue
    docker rm -f "memarena-sglang-${tag}" 2>/dev/null || true
  done
  docker rm -f "memarena-sglang-${model}" 2>/dev/null || true

  local mname="${MODEL_NAME[$model]}"
  local port="${MODEL_PORT[$model]}"
  local pip_install="protobuf sentencepiece"
  # NOTE: 32b is AWQ; sglang v0.5.9 supports AWQ natively. Do NOT install
  # vllm here — it upgrades torch and breaks sgl_kernel C++ ABI on H200.

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

write_models_tsv() {
  local model="$1"; local out_dir="$2"
  mkdir -p "$out_dir/_models_files"
  local tsv="$out_dir/_models_files/${model}.tsv"
  printf '%s|%s|http://localhost:%s\n' \
    "$model" "${MODEL_NAME[$model]}" "${MODEL_PORT[$model]}" >"$tsv"
}

launch_cells() {
  local model="$1"; local out_dir="$2"
  mkdir -p "$out_dir/wrapper_logs"
  log "launching ${#BACKENDS[@]}×${#TRIALS[@]} = $((${#BACKENDS[@]} * ${#TRIALS[@]})) cells for $model..."
  local pids=() labels=()
  for backend in "${BACKENDS[@]}"; do for t in "${TRIALS[@]}"; do
    local label="${backend}_${model}_${t}"
    local lf="$out_dir/wrapper_logs/${label}.log"
    local matrix_args=(
      --run-dir "$RUN_DIR" --out-dir "$out_dir"
      --models-file "$out_dir/_models_files/${model}.tsv"
      --backends "$backend" --trials "$t"
      --judge-preset remote
      --remote-judge-model "$JUDGE_MODEL"
      --remote-judge-endpoint "$JUDGE_ENDPOINT"
      --judge-api-key "$OPENROUTER_API_KEY"
      --answer-concurrency "$ANSWER_CONCURRENCY"
      --force
    )
    [[ -n "$TOP_K_OVERRIDE" ]] && matrix_args+=(--top-k "$TOP_K_OVERRIDE")
    [[ -n "$D6_ARM_OVERRIDE" ]] && matrix_args+=(--d6-arm "$D6_ARM_OVERRIDE")
    [[ -n "$EVAL_CONFIG_OVERRIDE" ]] && matrix_args+=(--eval-config "$EVAL_CONFIG_OVERRIDE")
    bash "$SCRIPT_DIR/run_eval_matrix.sh" "${matrix_args[@]}" >"$lf" 2>&1 &
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
  log "running judges for $model (concurrency=$JUDGE_CONCURRENCY)..."
  local pids=()
  for backend in "${BACKENDS[@]}"; do for t in "${TRIALS[@]}"; do
    # ablation backends use eval_results_${t}/${backend}/ as the subdir
    local subdir="$backend"
    local ans="$out_dir/eval_results_${t}/${subdir}/answer_results_${backend}_${model}_${t}.json"
    [[ -f "$ans" ]] || { log "  skip ${backend} ${t}: no answer file at $ans"; continue; }
    .venv/bin/python scripts/llmjudge.py \
      --run-dir "$RUN_DIR" --answer-path "$ans" \
      --judge-preset remote --judge-model "$JUDGE_MODEL" \
      --judge-endpoint "$JUDGE_ENDPOINT" --judge-api-key "$OPENROUTER_API_KEY" \
      --concurrency "$JUDGE_CONCURRENCY" --force \
      >"/tmp/judge_ablation_${backend}_${model}_${t}.log" 2>&1 &
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
log "out prefix: $OUT_PREFIX"

for model in "${MODELS[@]}"; do
  [[ -n "${MODEL_NAME[$model]:-}" ]] || { log "skip unknown model $model"; continue; }
  log "================ MODEL: $model ================"

  start_sglang "$model"

  out_dir="${OUT_PREFIX}${model}"
  rm -rf "$out_dir" || true
  mkdir -p "$out_dir"
  write_models_tsv "$model" "$out_dir"

  launch_cells "$model" "$out_dir"
  run_judges  "$model" "$out_dir"

  log "================ $model DONE → $out_dir ================"
done

log "all models complete: ${MODELS[*]}"
