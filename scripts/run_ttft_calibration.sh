#!/usr/bin/env bash
# run_ttft_calibration.sh — cold-prefill TTFT/decode calibration, one reader at a time.
#
# For each reader: serve sglang single-stream with the prefix cache OFF
# (--disable-radix-cache; --enable-cache-report so usage shows cached_tokens),
# run scripts/calibrate_ttft.py against it, save the server log, tear down.
# Output: $OUT_DIR/calib_<reader>.jsonl, $OUT_DIR/logs/, $OUT_DIR/run_meta.json
#
# Run on the Spark host:
#   BUNDLE=prompt_bundle.json OUT_DIR=out bash run_ttft_calibration.sh
#
# Knobs (env): READERS, IMAGE, PORT, GPU_DEVICES, MODEL_ROOT, PY, CALIB_PY

set -uo pipefail   # not -e: one reader failing should not abort the rest

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
READERS=(${READERS:-0_6b llama3b 8b 32b})
IMAGE="${IMAGE:-scitrera/dgx-spark-sglang:0.5.12}"
PORT="${PORT:-17100}"
GPU_DEVICES="${GPU_DEVICES:-0}"
MODEL_ROOT="${MODEL_ROOT:-$HOME/models}"
PY="${PY:-$HOME/MemArena/.venv/bin/python}"
CALIB_PY="${CALIB_PY:-$HERE/calibrate_ttft.py}"
BUNDLE="${BUNDLE:?BUNDLE required}"
OUT_DIR="${OUT_DIR:?OUT_DIR required}"
CONTAINER=ttft-calib-sglang

declare -A MODEL_NAME=(
  [0_6b]="Qwen/Qwen3-0.6B"
  [llama3b]="meta-llama/Llama-3.2-3B-Instruct"
  [7b]="mistralai/Mistral-7B-Instruct-v0.3"
  [8b]="Qwen/Qwen3-8B"
  [32b]="Qwen/Qwen3-32B-AWQ"
)
SGLANG_ARGS=(--tp 1 --dp 1 --context-length 16384 --mem-fraction-static 0.85
             --max-running-requests 1 --disable-radix-cache --enable-cache-report)

mkdir -p "$OUT_DIR/logs"
LOG="$OUT_DIR/logs/sweep.log"
log() { echo "[calib $(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
teardown() { docker rm -f "$CONTAINER" >/dev/null 2>&1 || true; }

cat > "$OUT_DIR/run_meta.json" <<EOF
{"host": "$(hostname)", "started_utc": "$(date -u +%FT%TZ)", "image": "$IMAGE",
 "image_id": "$(docker image inspect --format '{{.Id}}' "$IMAGE" 2>/dev/null)",
 "sglang_args": "${SGLANG_ARGS[*]}", "readers": "${READERS[*]}"}
EOF

for reader in "${READERS[@]}"; do
  mname="${MODEL_NAME[$reader]}"
  log "reader=$reader ($mname): starting sglang"
  teardown
  docker run -d --name "$CONTAINER" \
    --gpus "\"device=${GPU_DEVICES}\"" --ipc=host --shm-size 32g --restart no \
    -p "${PORT}:${PORT}" -v "${MODEL_ROOT}":/models:ro \
    "$IMAGE" \
    python3 -m sglang.launch_server --model-path "/models/${reader}" \
      --served-model-name "$mname" --host 0.0.0.0 --port "$PORT" "${SGLANG_ARGS[@]}" \
    >/dev/null

  deadline=$((SECONDS + 1200))
  until curl -fsS -m 3 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; do
    if [[ $SECONDS -ge $deadline ]] || ! docker ps -q -f name="$CONTAINER" | grep -q .; then
      log "  ERROR: sglang did not come up for $reader"
      docker logs --tail 80 "$CONTAINER" > "$OUT_DIR/logs/sglang_${reader}.log" 2>&1
      teardown
      continue 2
    fi
    sleep 5
  done
  log "  sglang ready"

  "$PY" "$CALIB_PY" --bundle "$BUNDLE" --reader "$reader" --served-model "$mname" \
    --endpoint "http://127.0.0.1:${PORT}/v1" --out "$OUT_DIR/calib_${reader}.jsonl" \
    >> "$OUT_DIR/logs/calib_${reader}.log" 2>&1
  rc=$?
  log "  calibration rc=$rc ($(grep -c '"timed"' "$OUT_DIR/calib_${reader}.jsonl" 2>/dev/null) timed rows)"
  docker logs "$CONTAINER" > "$OUT_DIR/logs/sglang_${reader}.log" 2>&1
  teardown
done
log "done"
