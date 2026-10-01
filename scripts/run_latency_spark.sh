#!/usr/bin/env bash
# run_latency_spark.sh — single-button latency measurement for Spark
# (or any single-GPU / low-resource node). Mirrors run_l_all_models.sh
# logic but with Spark-friendly defaults:
#
#   - 1 GPU, sglang TP=N_GPUS DP=1 (no DP=8)
#   - 1 model (default 0_6b), 1 trial (s2)
#   - --user-limit N_EGOS at BUILD AND ANSWER time (default 4)
#     vs the old "build 50, eval 8" pattern, this is ~12× faster on
#     the build phase since memobase/memos extractor only runs on N egos.
#   - Backends run SEQUENTIALLY (one stack at a time on a slow box)
#   - Judging disabled by default (--judge-preset none); per-question
#     latency comes from answer_results_*.json directly:
#         answer_time_ms, ttft_ms, search_time_ms, prompt_tokens
#
# Output:
#   out/latency_spark_${MODEL}_${TRIAL}/eval_results_${TRIAL}/...
#                               .../answer_results_${backend}_${MODEL}_${TRIAL}.json
#   plus a printed p50/p95 summary at the end.
#
# Knobs (env vars):
#   MODEL          default 0_6b               # 0_6b llama3b 7b 8b 32b
#   TRIAL          default s2
#   N_EGOS         default 4
#   BACKENDS       default "vanilla baseline_simplerag memobase memos"
#                  (oracle is bounded-context too; add "oracle" if wanted)
#   GPU_DEVICES    default "0"                # comma-sep, e.g. "0,1"
#   N_GPUS         default 1                  # for --tp
#   SGLANG_PORT    default 17000
#   RUN_DIR        default data/benchmark
#
# Examples:
#   bash scripts/run_latency_spark.sh
#   MODEL=llama3b N_EGOS=4 bash scripts/run_latency_spark.sh
#   BACKENDS="vanilla baseline_simplerag" bash scripts/run_latency_spark.sh
#   GPU_DEVICES="0,1" N_GPUS=2 bash scripts/run_latency_spark.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${OPENROUTER_API_KEY:-}" && -f .env ]]; then set -a; . .env; set +a; fi

MODEL="${MODEL:-0_6b}"
TRIAL="${TRIAL:-s2}"
N_EGOS="${N_EGOS:-4}"
BACKENDS_DEFAULT="vanilla baseline_simplerag memobase memos"
read -r -a BACKENDS <<<"${BACKENDS:-$BACKENDS_DEFAULT}"
GPU_DEVICES="${GPU_DEVICES:-0}"
N_GPUS="${N_GPUS:-1}"
SGLANG_PORT="${SGLANG_PORT:-17000}"
RUN_DIR="${RUN_DIR:-data/benchmark}"
SERVICES_ROOT="${MEMARENA_MEMORY_SERVICES_ROOT:-$HOME/memarena-memory-services}"

declare -A MODEL_NAME
MODEL_NAME[0_6b]="Qwen/Qwen3-0.6B"
MODEL_NAME[llama3b]="meta-llama/Llama-3.2-3B-Instruct"
MODEL_NAME[7b]="mistralai/Mistral-7B-Instruct-v0.3"
MODEL_NAME[8b]="Qwen/Qwen3-8B"
MODEL_NAME[32b]="Qwen/Qwen3-32B-AWQ"
[[ -n "${MODEL_NAME[$MODEL]:-}" ]] || { echo "unknown MODEL=$MODEL"; exit 2; }

OUT_DIR="out/latency_spark_${MODEL}_${TRIAL}"

log() { echo "[lat-spark $(date -u +%H:%M:%S)] $*"; }

# ---- teardown ------------------------------------------------------
teardown_all() {
  log "teardown..."
  pkill -f "build_memory_cache" 2>/dev/null || true
  pkill -f "run_eval_matrix"   2>/dev/null || true
  sleep 1
  docker rm -f "lat-spark-sglang" >/dev/null 2>&1 || true
  for proj in "memarena_${MODEL}_${TRIAL}_memobase" "memarena_${MODEL}_${TRIAL}_memos"; do
    local cs
    cs="$(docker ps -aq --filter "label=com.docker.compose.project=${proj}" 2>/dev/null || true)"
    [[ -n "$cs" ]] && docker rm -f $cs >/dev/null 2>&1 || true
  done
}

# ---- sglang (single GPU, TP=N_GPUS) -------------------------------
start_sglang() {
  log "starting sglang $MODEL TP=$N_GPUS port=$SGLANG_PORT GPUs=$GPU_DEVICES"
  docker rm -f "lat-spark-sglang" 2>/dev/null || true
  local mname="${MODEL_NAME[$MODEL]}"
  local pip_install="protobuf sentencepiece"
  [[ "$MODEL" == "32b" ]] && pip_install="$pip_install vllm==0.7.2"

  docker run -d --name "lat-spark-sglang" \
    --gpus "\"device=${GPU_DEVICES}\"" --ipc=host --shm-size 32g --restart no \
    -p "${SGLANG_PORT}:${SGLANG_PORT}" \
    -v "${HOME}/models:/models:ro" \
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
    --env "SGLANG_PIP_INSTALL=${pip_install}" \
    lmsysorg/sglang:latest \
    bash -lc 'set -euo pipefail
if [[ -n "${SGLANG_PIP_INSTALL:-}" ]]; then python3 -m pip install --no-cache-dir ${SGLANG_PIP_INSTALL}; fi
exec "$@"
' bash python3 -m sglang.launch_server \
      --model-path "/models/${MODEL}" --served-model-name "$mname" \
      --host 0.0.0.0 --port "$SGLANG_PORT" --tp "$N_GPUS" --dp 1 \
      --context-length 16384 --mem-fraction-static 0.85 --max-running-requests 32 >/dev/null

  log "waiting for sglang on :$SGLANG_PORT..."
  until curl -fsS "http://127.0.0.1:${SGLANG_PORT}/v1/models" >/dev/null 2>&1; do sleep 5; done
  log "sglang ready"
}

# ---- memobase + memos stacks (only when needed) -------------------
need_stack() {
  for b in "${BACKENDS[@]}"; do
    case "$b" in memobase|memos) return 0 ;; esac
  done
  return 1
}

spin_stacks() {
  [[ -d "$SERVICES_ROOT/$MODEL/memobase/src/server" && -d "$SERVICES_ROOT/$MODEL/MemOS/docker" ]] || {
    log "ERROR: missing $SERVICES_ROOT/$MODEL/{memobase,MemOS} — run setup_memory_backends.sh first or skip memobase/memos backends"
    exit 3
  }
  log "spinning memobase + memos stacks for $MODEL/$TRIAL..."
  bash "$SCRIPT_DIR/spin_per_trial_stacks.sh" up "$MODEL" "$TRIAL" 2>&1 | tail -5

  local urls; urls="$(bash "$SCRIPT_DIR/spin_per_trial_stacks.sh" urls "$MODEL" "$TRIAL")"
  MB_URL="$(awk -F= '/MEMOBASE_BASE_URL/{print $2}' <<<"$urls")"
  MOS_URL="$(awk -F= '/MEMOS_BASE_URL/{print $2}' <<<"$urls")"
  log "memobase: $MB_URL   memos: $MOS_URL"

  log "waiting for memobase healthcheck..."
  local deadline=$((SECONDS + 240))
  while ! curl -fsS "${MB_URL}/api/v1/healthcheck" >/dev/null 2>&1; do
    [[ "$SECONDS" -ge "$deadline" ]] && { log "WARNING: memobase not healthy after 240s"; break; }
    sleep 3
  done
  log "  ✓ memobase ready"
}

# ---- write models.tsv ---------------------------------------------
write_tsv() {
  mkdir -p "$OUT_DIR/_models_files"
  local tsv="$OUT_DIR/_models_files/${MODEL}_${TRIAL}.tsv"
  if need_stack; then
    printf '%s|%s|http://localhost:%s|%s|%s\n' \
      "$MODEL" "${MODEL_NAME[$MODEL]}" "$SGLANG_PORT" "$MB_URL" "$MOS_URL" >"$tsv"
  else
    printf '%s|%s|http://localhost:%s\n' \
      "$MODEL" "${MODEL_NAME[$MODEL]}" "$SGLANG_PORT" >"$tsv"
  fi
  log "models tsv: $tsv"
}

# ---- hw_probe sidecar lifecycle -----------------------------------
# Spawn a 100ms hw_probe sampler against $1 (CSV path); echo back the PID.
# Caller stops it with `_hw_probe_stop $pid $csv $aggregate_outfile`.
_hw_probe_start() {
  local out_csv="$1" log_path="$2"
  if [[ "${HW_PROBE_DISABLE:-0}" == "1" ]]; then
    echo ""
    return 0
  fi
  mkdir -p "$(dirname "$out_csv")"
  "${PYTHON_BIN:-python3}" "$SCRIPT_DIR/hw_probe.py" \
      --outfile "$out_csv" \
      --interval-ms "${HW_PROBE_INTERVAL_MS:-100}" \
      ${HW_PROBE_SKIP_DOCKER:+--skip-docker} \
      > "$log_path" 2>&1 &
  local pid=$!
  sleep 0.5  # let probe open CSV header before workload starts
  echo "$pid"
}

_hw_probe_stop() {
  local pid="$1" csv="$2" out_json="$3" log_path="$4" answer_json="${5:-}"
  [[ -z "$pid" ]] && return 0
  kill -TERM "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  [[ -s "$csv" ]] || return 0
  "${PYTHON_BIN:-python3}" "$SCRIPT_DIR/aggregate_hw_probe_simple.py" \
      --csv "$csv" \
      ${answer_json:+--answer-results "$answer_json"} \
      --outfile "$out_json" \
      >> "$log_path" 2>&1 || \
      log "  ! hw_aggregate failed for $csv (see $log_path)"
}

# ---- run one cell --------------------------------------------------
# Two phases for memory_cache backends (memobase/memos/memsearch):
#   1. INGEST: pre-build the memcache JSONL with hw_probe wrapping the
#      build_memory_cache.py invocation -> hw_agg_<be>_<reader>_<seed>_ingest.json
#   2. ANSWER: run run_eval_matrix.sh which finds the prebuilt cache and
#      skips its own build, with hw_probe wrapping ONLY the answer phase
#      -> hw_agg_<be>_<reader>_<seed>.json
# Non-memcache backends (vanilla / inmem / oracle) skip phase 1; the answer
# probe captures the entire cell.
#
# HW_PROBE_DISABLE=1 disables both probes.
run_cell() {
  local backend="$1"
  local lf="$OUT_DIR/wrapper_logs/${backend}_${MODEL}_${TRIAL}.log"
  local hw_log="$OUT_DIR/wrapper_logs/hw_probe_${backend}_${MODEL}_${TRIAL}.log"
  mkdir -p "$OUT_DIR/wrapper_logs" "$OUT_DIR/hw"
  log "RUN $backend (egos=$N_EGOS)"

  local rc=0
  local needs_build=0
  case "$backend" in
    memobase|memos|memsearch) needs_build=1 ;;
  esac

  # Phase 1: pre-build memcache with ingest probe.
  if [[ "$needs_build" -eq 1 ]]; then
    local model_safe
    model_safe="$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')"
    local trial_safe="$TRIAL"
    local backend_safe="$backend"
    local cache_dir="$OUT_DIR/memory_cache/${backend_safe}/${model_safe}/${trial_safe}"
    local cache_path="${cache_dir}/memcache_${backend_safe}_A_paired_${model_safe}_${trial_safe}.jsonl"
    mkdir -p "$cache_dir"
    rm -f "$cache_path"  # build from zero per N_EGOS sweep semantics
    local ingest_csv="$OUT_DIR/hw/hw_probe_${backend}_${MODEL}_${TRIAL}_ingest.csv"
    local ingest_pid
    ingest_pid="$(_hw_probe_start "$ingest_csv" "$hw_log")"
    log "  build memcache (ingest probe pid=${ingest_pid:-disabled})"

    local mname
    mname="$(awk -F'|' "/^${MODEL}\\|/ {print \$2; exit}" "$OUT_DIR/_models_files/${MODEL}_${TRIAL}.tsv")"
    local endpoint="http://localhost:${SGLANG_PORT}"
    local extractor_api_key="${MEMORY_CACHE_EXTRACTOR_API_KEY:-EMPTY}"
    local memobase_url memos_url
    memobase_url="$(awk -F'|' "/^${MODEL}\\|/ {print \$4; exit}" "$OUT_DIR/_models_files/${MODEL}_${TRIAL}.tsv" 2>/dev/null)"
    memos_url="$(awk -F'|' "/^${MODEL}\\|/ {print \$5; exit}" "$OUT_DIR/_models_files/${MODEL}_${TRIAL}.tsv" 2>/dev/null)"

    local -a build_env=()
    [[ "$backend" == "memobase" && -n "$memobase_url" ]] && build_env+=(MEMOBASE_BASE_URL="$memobase_url" MEMOBASE_API_TOKEN="${MEMOBASE_API_TOKEN:-secret}")
    [[ "$backend" == "memos"    && -n "$memos_url"    ]] && build_env+=(MEMOS_BASE_URL="$memos_url" MEMOS_API_KEY="${MEMOS_API_KEY:-EMPTY}")

    local trial_seed
    case "$TRIAL" in
      s2) trial_seed=1002 ;; s3) trial_seed=1003 ;; s4) trial_seed=1004 ;; *) trial_seed=1002 ;;
    esac

    # Build_memory_cache.py imports `eval.src.types` and `MASim.*`; both
    # only resolve when PYTHONPATH points at the repo root. Invoke from the
    # repo root + use the project venv's python so memsearch / memobase
    # adapters are findable.
    local python_bin
    if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
        python_bin="$REPO_ROOT/.venv/bin/python"
    else
        python_bin="${PYTHON_BIN:-python3}"
    fi
    if (cd "$REPO_ROOT" && env PYTHONPATH="$REPO_ROOT" "${build_env[@]}" "$python_bin" \
          "$REPO_ROOT/scripts/reproduce/build_memory_cache.py" \
          --system "$backend" \
          --run-dir "$RUN_DIR" \
          --extractor-model "$mname" \
          --extractor-endpoint "$endpoint" \
          --extractor-api-key "$extractor_api_key" \
          --extractor-config A_paired \
          --output "$cache_path" \
          --trial-seed "$trial_seed" \
          --user-limit "$N_EGOS" \
          --concurrency "${MEMORY_CACHE_CONCURRENCY:-32}" \
          >>"$lf" 2>&1); then
      _hw_probe_stop "$ingest_pid" "$ingest_csv" \
          "$OUT_DIR/hw/hw_agg_${backend}_${MODEL}_${TRIAL}_ingest.json" \
          "$hw_log" ""
      log "  ✓ ingest done"
    else
      rc=$?
      _hw_probe_stop "$ingest_pid" "$ingest_csv" \
          "$OUT_DIR/hw/hw_agg_${backend}_${MODEL}_${TRIAL}_ingest.json" \
          "$hw_log" ""
      log "  ✗ ingest FAILED (rc=$rc) — see $lf; skipping answer phase"
      return 0
    fi
  fi

  # Phase 2: answer with hw probe. run_eval_matrix.sh will find the prebuilt
  # cache (no --force) and skip its own build for memcache backends.
  local answer_csv="$OUT_DIR/hw/hw_probe_${backend}_${MODEL}_${TRIAL}.csv"
  local answer_pid
  answer_pid="$(_hw_probe_start "$answer_csv" "$hw_log")"
  log "  answer (probe pid=${answer_pid:-disabled})"

  local matrix_args=(--run-dir "$RUN_DIR" --out-dir "$OUT_DIR"
        --models-file "$OUT_DIR/_models_files/${MODEL}_${TRIAL}.tsv"
        --backends "$backend" --trials "$TRIAL"
        --user-limit "$N_EGOS"
        --judge-preset none)
  [[ "$needs_build" -eq 0 ]] && matrix_args+=(--force)

  bash "$SCRIPT_DIR/run_eval_matrix.sh" "${matrix_args[@]}" >>"$lf" 2>&1 || rc=$?

  local ans_glob
  ans_glob=$(ls -1 "$OUT_DIR"/eval_results_"$TRIAL"/*/answer_results_*"${MODEL}"_"${TRIAL}".json 2>/dev/null | grep -v _light | grep -v openclaw | head -1 || true)
  _hw_probe_stop "$answer_pid" "$answer_csv" \
      "$OUT_DIR/hw/hw_agg_${backend}_${MODEL}_${TRIAL}.json" \
      "$hw_log" "$ans_glob"

  if [[ $rc -eq 0 ]]; then
    log "  ✓ $backend done"
  else
    log "  ✗ $backend FAILED — see $lf"
  fi
}

# ---- summary -------------------------------------------------------
# Two latency views per backend:
#   - ANSWER (online):  per-question answer_time / ttft / search_time / prompt
#                       from answer_results_*.json
#   - BUILD  (offline): one-shot indexing cost. Parses
#                       wrapper_logs/${backend}_${MODEL}_${TRIAL}.log for
#                       "[day X/15] ... elapsed=Ns" lines and reports
#                       total + per-day mean. vanilla/oracle have no
#                       build phase → "—".
summarize() {
  log "=== latency summary (out: $OUT_DIR) ==="
  OUT_DIR="$OUT_DIR" python3 - <<'PY'
import json, glob, statistics, os, re
out = os.environ['OUT_DIR']
files = sorted(glob.glob(f'{out}/eval_results_*/**/answer_results_*.json', recursive=True))
if not files:
    print("(no answer_results files found)")
    raise SystemExit
day_re = re.compile(r'\[day \d+/\d+\][^\n]*elapsed=([0-9.]+)s')

def build_stats(backend):
    """Sum of '[day X/15] elapsed=Ns' lines in this backend's wrapper log."""
    matches = []
    for w in glob.glob(f'{out}/wrapper_logs/{backend}_*.log'):
        with open(w) as f:
            matches.extend(float(m.group(1)) for m in day_re.finditer(f.read()))
    return matches  # list of per-day elapsed seconds

hdr = (f'{"backend":18s}  {"n":>4s}  {"answer_p50":>10s}  {"answer_p95":>10s}  '
       f'{"ttft_p50":>10s}  {"search_p50":>10s}  {"prompt_mean":>11s}  '
       f'{"build_total_s":>13s}  {"build_per_day":>13s}  {"build_days":>10s}')
print(hdr); print("-" * len(hdr))
for p in files:
    with open(p) as f: rows = json.load(f)
    if not rows: continue
    backend = os.path.basename(p).split("_", 2)[2].rsplit("_", 2)[0]
    a = sorted(r.get("answer_time_ms", 0) or 0 for r in rows)
    t = sorted(r.get("ttft_ms", 0) or 0 for r in rows)
    s = sorted(r.get("search_time_ms", 0) or 0 for r in rows)
    pt = [r.get("prompt_tokens", 0) or 0 for r in rows]
    def pct(xs, p): return xs[min(len(xs) - 1, int(len(xs) * p))]
    days = build_stats(backend)
    if days:
        bts = f"{sum(days):.1f}"
        bpds = f"{statistics.mean(days):.1f}"
        bdays = f"{len(days)}"
    else:
        bts = bpds = bdays = "—"
    print(f"{backend:18s}  {len(rows):>4d}  "
          f"{pct(a,0.5):>10.1f}  {pct(a,0.95):>10.1f}  "
          f"{pct(t,0.5):>10.1f}  {pct(s,0.5):>10.1f}  "
          f"{statistics.mean(pt):>11.0f}  "
          f"{bts:>13s}  {bpds:>13s}  {bdays:>10s}")

print()
print("answer_*: per-question online latency (memory search + LLM answer).")
print("build_*: one-shot indexing cost from build_memory_cache day cadence.")
print("    build_total_s = sum of all per-day elapsed times.")
print("    build_per_day = mean elapsed per day (s).")
print("vanilla/oracle have no build phase (no extractor/embedder/DB writes).")
PY
}

# ---- main ----------------------------------------------------------
log "=== run_latency_spark.sh ==="
log "MODEL=$MODEL TRIAL=$TRIAL N_EGOS=$N_EGOS GPUs=$GPU_DEVICES port=$SGLANG_PORT"
log "BACKENDS: ${BACKENDS[*]}"
log "OUT_DIR:  $OUT_DIR"

teardown_all
mkdir -p "$OUT_DIR"

start_sglang

if need_stack; then spin_stacks; fi
write_tsv

for b in "${BACKENDS[@]}"; do run_cell "$b"; done

summarize
teardown_all
log "DONE → $OUT_DIR"
