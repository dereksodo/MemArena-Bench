#!/usr/bin/env bash
# run_latency_answer.sh — answer-only latency measurement for one cell.
#
# Pipeline:
#   1. Build masim_qa (the eval_instances → QA conversion is fast, CPU-only).
#   2. Filter masim_qa to the 160 instance_ids in out/latency_selection/queries_s2.json
#      (via the new --instance-id-file flag wired into eval/cli.py).
#   3. Run answer phase against an existing sglang endpoint, concurrency=1
#      (single-stream serving — gives the realistic per-request latency the
#      paper cares about, not throughput).
#   4. answer_results_<backend>_<model>_s2.json gets the per-question
#      answer_time_ms / ttft_ms / search_time_ms; we summarise p50/p95 inline.
#
# Assumes:
#   * sglang serving the reader on http://localhost:17000 (start it separately).
#   * If backend ∈ {memobase, memsearch}, the corresponding docker stack is
#     already up AND the memory cache is already built (this script does NOT
#     ingest — pair it with run_latency_ingest.sh).
#
# Output layout:
#   out/latency2_answer/${backend}_${model}_s2/...
#     eval_results_s2/<backend>/answer_results_<backend>_<model>_s2.json
#     latency_summary.txt   <-- p50/p95 from this run
#
# Usage:
#   BACKEND=vanilla MODEL=0_6b bash scripts/run_latency_answer.sh
#   BACKEND=memsearch MODEL=7b bash scripts/run_latency_answer.sh
#
# Knobs (env):
#   BACKEND         vanilla | baseline_simplerag | oracle | memobase | memsearch
#   MODEL           0_6b | llama3b | 7b | 8b | 32b
#   TRIAL           default s2
#   SGLANG_URL      default http://localhost:17000
#   IDS_FILE        default out/latency_selection/queries_s2.json
#   OUT_ROOT        default out/latency2_answer
#   RUN_DIR         default data/benchmark
#   ANSWER_CONCURRENCY   default 1   (single-stream, the operating point)
#   EVAL_CONFIG     default eval/config/pipeline_vanilla512.yaml
#                   (the vanilla-512 ablation config from commit 7cb42d9 — for
#                    non-vanilla backends the extra `vanilla:` block is ignored)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
[[ -z "${OPENROUTER_API_KEY:-}" && -f .env ]] && { set -a; . .env; set +a; }

BACKEND="${BACKEND:?BACKEND required}"
MODEL="${MODEL:?MODEL required}"
TRIAL="${TRIAL:-s2}"
SGLANG_URL="${SGLANG_URL:-http://localhost:17000}"
IDS_FILE="${IDS_FILE:-out/latency_selection/queries_s2.json}"
OUT_ROOT="${OUT_ROOT:-out/latency2_answer}"
RUN_DIR="${RUN_DIR:-data/benchmark}"
ANSWER_CONCURRENCY="${ANSWER_CONCURRENCY:-1}"
EVAL_CONFIG="${EVAL_CONFIG:-eval/config/pipeline_vanilla512.yaml}"

declare -A MODEL_NAME=(
  [0_6b]="Qwen/Qwen3-0.6B"
  [llama3b]="meta-llama/Llama-3.2-3B-Instruct"
  [7b]="mistralai/Mistral-7B-Instruct-v0.3"
  [8b]="Qwen/Qwen3-8B"
  [32b]="Qwen/Qwen3-32B-AWQ"
)
[[ -n "${MODEL_NAME[$MODEL]:-}" ]] || { echo "unknown MODEL=$MODEL"; exit 2; }

[[ -f "$IDS_FILE" ]] || { echo "missing IDS_FILE=$IDS_FILE — run scripts/select_latency_queries.py first"; exit 2; }

NS="${BACKEND}_${MODEL}_${TRIAL}"
OUT_DIR="${OUT_ROOT}/${NS}"
mkdir -p "$OUT_DIR/logs"

echo "[lat-ans $(date -u +%H:%M:%S)] cell=$NS endpoint=$SGLANG_URL"
echo "[lat-ans $(date -u +%H:%M:%S)] ids_file=$IDS_FILE  concurrency=$ANSWER_CONCURRENCY"
echo "[lat-ans $(date -u +%H:%M:%S)] out_dir=$OUT_DIR"

# Health-check sglang before doing anything heavy
if ! curl -fsS -m 5 "$SGLANG_URL/v1/models" >/dev/null 2>&1; then
  echo "[lat-ans] sglang not reachable at $SGLANG_URL"; exit 3
fi

# ---- hw_probe sidecar ---------------------------------------------------
HW_DIR="$OUT_DIR/hw"
mkdir -p "$HW_DIR"
HW_CSV="$HW_DIR/hw_probe_${BACKEND}_${MODEL}_${TRIAL}.csv"
HW_AGG="$HW_DIR/hw_agg_${BACKEND}_${MODEL}_${TRIAL}.json"
HW_LOG="$OUT_DIR/logs/hw_probe_${BACKEND}_${MODEL}_${TRIAL}.log"
HW_PID=""
if [[ "${HW_PROBE_DISABLE:-0}" != "1" ]]; then
  python3 scripts/hw_probe.py \
    --outfile "$HW_CSV" \
    --interval-ms "${HW_PROBE_INTERVAL_MS:-100}" \
    ${HW_PROBE_SKIP_DOCKER:+--skip-docker} \
    > "$HW_LOG" 2>&1 &
  HW_PID=$!
  sleep 0.5
  echo "[lat-ans] hw_probe pid=$HW_PID -> $HW_CSV"
fi

stop_hw_probe() {
  [[ -z "$HW_PID" ]] && return 0
  kill -TERM "$HW_PID" 2>/dev/null || true
  wait "$HW_PID" 2>/dev/null || true
  if [[ -s "$HW_CSV" ]]; then
    python3 scripts/aggregate_hw_probe_simple.py \
      --csv "$HW_CSV" \
      ${ANS_FILE:+--answer-results "$ANS_FILE"} \
      --outfile "$HW_AGG" \
      >> "$HW_LOG" 2>&1 || echo "[lat-ans] WARN: hw_aggregate failed (see $HW_LOG)"
  fi
}
trap stop_hw_probe EXIT

# memsearch / memobase have no live answer-phase adapter in eval/cli.py — they
# replay from the memcache jsonl built during ingest. The cache jsonl carries
# `retrieve_latency_ms` per row (true live search time captured during ingest),
# so even though `search_time_ms` from the answer pass is 0, total latency
# = answer_time_ms + retrieve_latency_ms (combined post-hoc).
SYS_FOR_CLI="$BACKEND"
EXTRA_CLI_ARGS=()
if [[ "$BACKEND" == "memsearch" || "$BACKEND" == "memobase" ]]; then
  CACHE_PATH="${CACHE_PATH:-${OUT_ROOT/_answer/_ingest}/${BACKEND}_${MODEL}_${TRIAL}/memcache_${BACKEND}_A_paired_${MODEL}_${TRIAL}.jsonl}"
  [[ -f "$CACHE_PATH" ]] || { echo "[lat-ans] missing cache jsonl for $BACKEND: $CACHE_PATH"; exit 4; }
  SYS_FOR_CLI="memory_cache"
  EXTRA_CLI_ARGS+=(--cache-path "$CACHE_PATH" --expected-memory-system "$BACKEND")
  echo "[lat-ans] $BACKEND → --system memory_cache --cache-path $CACHE_PATH"
fi

LOG="$OUT_DIR/logs/run_${BACKEND}_${MODEL}_${TRIAL}.log"
.venv/bin/python scripts/run_accuracy.py \
  --run-dir "$RUN_DIR" \
  --system "$SYS_FOR_CLI" \
  "${EXTRA_CLI_ARGS[@]}" \
  --judge-preset none \
  --eval-config "$EVAL_CONFIG" \
  --sglang-url "$SGLANG_URL" \
  --model-name "${MODEL_NAME[$MODEL]}" \
  --model-tag "$MODEL" \
  --trial-name "$TRIAL" \
  --seed 1002 \
  --namespace "$NS" \
  --out-dir "$OUT_DIR" \
  --temperature 0.3 \
  --answer-concurrency "$ANSWER_CONCURRENCY" \
  --eval-concurrency 512 \
  --instance-id-file "$IDS_FILE" \
  --force \
  2>&1 | tee "$LOG"

# Locate answer_results and summarise
ANS_FILE="$(find "$OUT_DIR" -name "answer_results_${BACKEND}_${MODEL}_${TRIAL}.json" -not -path '*/runs/*' | head -1)"
[[ -z "$ANS_FILE" ]] && ANS_FILE="$(find "$OUT_DIR" -name "answer_results_*${MODEL}_${TRIAL}.json" -not -path '*/runs/*' | head -1)"

if [[ -n "$ANS_FILE" && -f "$ANS_FILE" ]]; then
  python3 - "$ANS_FILE" "$OUT_DIR/latency_summary.txt" <<'PY'
import json, sys, statistics
ans, summary = sys.argv[1], sys.argv[2]
d = json.load(open(ans))
rs = d if isinstance(d, list) else d.get("results") or d.get("details") or []
def stat(field):
    xs = [r.get(field) for r in rs if isinstance(r, dict) and r.get(field) is not None]
    if not xs: return f"{field}: no data"
    xs.sort()
    n = len(xs); m = sum(xs)/n
    p50 = xs[n//2]; p95 = xs[max(0, int(n*0.95)-1)]
    return f"{field:>18s}  n={n:4d}  p50={p50:8.1f}  p95={p95:9.1f}  mean={m:8.1f}  min={xs[0]:7.1f}  max={xs[-1]:8.1f}"
out = [
    f"# Cell: {ans}",
    f"# Records: {len(rs)}",
    stat("answer_time_ms"),
    stat("ttft_ms"),
    stat("search_time_ms"),
    stat("prompt_tokens"),
    stat("completion_tokens"),
]
print("\n".join(out))
open(summary, "w").write("\n".join(out)+"\n")
PY
else
  echo "[lat-ans] WARN: no answer_results_*.json found under $OUT_DIR"
fi
