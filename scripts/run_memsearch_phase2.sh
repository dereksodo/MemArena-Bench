#!/usr/bin/env bash
# Phase 2: 15 memsearch eval cells (5 readers × 3 trials) running answer +
# judge against the SAME pre-built memsearch memcache jsonl.
#
# Why one orchestrator instead of run_l_all_models.sh:
#   - run_l_all_models.sh sequences models (load reader N → run all trials →
#     teardown sglang N → load reader N+1). For memsearch, the memcache is
#     reader-agnostic and small, so we can hold all 5 sglangs in parallel
#     across the 8 GPUs and finish in one wave.
#   - Avoids the strict-extractor-tag mismatch: chain expects
#     extractor_model = reader_name; the shared memsearch jsonl was built
#     with extractor_model="shared". We pass --no-cache-strict to bypass
#     the validation (memsearch's chunks are reader-independent by design).
#
# GPU allocation (8 × H100 80GB):
#   GPU 0:   ollama-memsearch (Phase 1 leftover, ~1 GB; harmless if absent)
#   GPU 1:   sglang-0_6b      (DP=1)
#   GPU 2:   sglang-llama3b   (DP=1)
#   GPU 3:   sglang-7b        (DP=1)
#   GPU 4:   sglang-8b        (DP=1)
#   GPU 5,6: sglang-32b       (TP=2, AWQ ~24 GB)
#   GPU 7:   spare (room to scale 32b TP up if needed)
#
# Usage:
#   bash scripts/run_memsearch_phase2.sh \
#     [--shared-cache out/_memsearch_shared/memcache_memsearch_A_paired_shared_s2.jsonl]

set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"

[[ -f .env ]] && { set -a; . .env; set +a; }
[[ -n "${OPENROUTER_API_KEY:-}" ]] || { echo "OPENROUTER_API_KEY required (for judge)"; exit 2; }

SHARED_CACHE="${1:-out/_memsearch_shared/memcache_memsearch_A_paired_shared_s2.jsonl}"
[[ -f "$SHARED_CACHE" ]] || { echo "shared memcache missing: $SHARED_CACHE"; exit 2; }
echo "[phase2] shared memcache: $SHARED_CACHE ($(wc -l < "$SHARED_CACHE") rows)"

declare -A MODEL_NAME=(
  [0_6b]="Qwen/Qwen3-0.6B"
  [llama3b]="meta-llama/Llama-3.2-3B-Instruct"
  [7b]="mistralai/Mistral-7B-Instruct-v0.3"
  [8b]="Qwen/Qwen3-8B"
  [32b]="Qwen/Qwen3-32B-AWQ"
)
declare -A MODEL_PORT=(
  [0_6b]=16000 [llama3b]=16001 [7b]=16002 [8b]=16003 [32b]=16004
)
declare -A MODEL_GPUS=(
  [0_6b]="1" [llama3b]="2" [7b]="3" [8b]="4" [32b]="5,6"
)
declare -A MODEL_TP=(
  [0_6b]=1 [llama3b]=1 [7b]=1 [8b]=1 [32b]=2
)
declare -A MODEL_SLUG=(
  [0_6b]=0_6b [llama3b]=3b [7b]=7b [8b]=8b [32b]=32b
)
TRIALS=(s2 s3 s4)
SEED_S2=1002; SEED_S3=1003; SEED_S4=1004

log() { echo "[phase2 $(date -u +%H:%M:%S)] $*"; }

# ---- Step 1: symlink shared memcache to 15 cell-specific paths ----------
log "step 1: symlink shared memcache to 15 expected paths"
for model in "${!MODEL_NAME[@]}"; do
  slug="${MODEL_SLUG[$model]}"
  for t in "${TRIALS[@]}"; do
    cell_dir="out/accuracy_memarena_l_${slug}/memory_cache/memsearch/${model}/${t}"
    cell_cache="${cell_dir}/memcache_memsearch_A_paired_${model}_${t}.jsonl"
    mkdir -p "$cell_dir"
    if [[ ! -e "$cell_cache" ]]; then
      ln -s "$ROOT/$SHARED_CACHE" "$cell_cache"
    fi
    log "  → $cell_cache"
  done
done

# ---- Step 2: stop any sglangs left over from previous runs --------------
log "step 2: stop leftover sglangs (preserve ollama-memsearch)"
docker ps -aq --filter 'name=memarena-sglang-' | xargs -r docker rm -f >/dev/null 2>&1

# ---- Step 3: start 5 sglangs in parallel on GPU 1-7 ---------------------
log "step 3: start 5 sglangs in parallel"
for model in "${!MODEL_NAME[@]}"; do
  port="${MODEL_PORT[$model]}"
  mname="${MODEL_NAME[$model]}"
  gpus="${MODEL_GPUS[$model]}"
  tp="${MODEL_TP[$model]}"
  pip_install="protobuf sentencepiece"
  [[ "$model" == "32b" ]] && pip_install="$pip_install vllm==0.7.2"
  log "  → sglang-${model}: GPU=$gpus TP=$tp port=$port"
  docker run -d --name "memarena-sglang-${model}" \
    --gpus "\"device=${gpus}\"" --ipc=host --shm-size 32g --restart no \
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
      --host 0.0.0.0 --port "$port" --tp "$tp" --dp 1 \
      --context-length 16384 --mem-fraction-static 0.85 --max-running-requests 64 >/dev/null
done

# ---- Step 4: wait for all 5 sglangs ready -------------------------------
log "step 4: wait for all 5 sglangs ready"
for model in "${!MODEL_NAME[@]}"; do
  port="${MODEL_PORT[$model]}"
  log "  waiting sglang-$model :$port..."
  waited=0
  until curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; do
    sleep 5; waited=$((waited+5))
    [[ $waited -gt 600 ]] && { log "  TIMEOUT sglang-$model"; exit 3; }
  done
  log "  ✓ sglang-$model ready (${waited}s)"
done

# ---- Step 5: launch 15 cells in parallel --------------------------------
log "step 5: launch 15 cells (3 trials × 5 readers) in parallel"
declare -A CELL_PIDS
for model in "${!MODEL_NAME[@]}"; do
  slug="${MODEL_SLUG[$model]}"
  port="${MODEL_PORT[$model]}"
  mname="${MODEL_NAME[$model]}"
  out_dir="out/accuracy_memarena_l_${slug}"
  mkdir -p "$out_dir/wrapper_logs"
  for t in "${TRIALS[@]}"; do
    case "$t" in s2) seed=$SEED_S2 ;; s3) seed=$SEED_S3 ;; s4) seed=$SEED_S4 ;; esac
    label="${model}_${t}"
    logf="$out_dir/wrapper_logs/memsearch_${label}.log"
    cache_path="$out_dir/memory_cache/memsearch/${model}/${t}/memcache_memsearch_A_paired_${model}_${t}.jsonl"
    log "  → ${label} (sglang :$port, cache=$(basename "$cache_path"))"
    .venv/bin/python scripts/run_accuracy.py \
      --run-dir data/benchmark \
      --system memory_cache \
      --judge-preset remote \
      --sglang-url "http://localhost:${port}" \
      --model-name "$mname" --model-tag "$model" \
      --trial-name "$t" --seed "$seed" \
      --namespace "memsearch_${model}_${t}" \
      --out-dir "$out_dir" \
      --temperature 0.3 \
      --cache-path "$cache_path" \
      --no-cache-strict \
      --expected-memory-system memsearch \
      --expected-config A_paired \
      --answer-concurrency 32 --eval-concurrency 512 \
      --remote-judge-model openai/gpt-4o-mini-2024-07-18 \
      --remote-judge-endpoint https://openrouter.ai/api/v1 \
      --judge-api-key "$OPENROUTER_API_KEY" \
      --force \
      > "$logf" 2>&1 &
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
log "result: ok=$ok fail=$fail / 15 cells"
log "evaluation_results files written under out/accuracy_memarena_l_*/eval_results_*/memory_cache/"
