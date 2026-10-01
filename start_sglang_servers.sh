#!/usr/bin/env bash
# Start the five MemArena SGLang reader servers with Docker.
#
# Defaults:
#   0_6b    -> GPU 0       -> http://localhost:16000
#   llama3b -> GPU 1       -> http://localhost:16001
#   7b      -> GPUs 2,3    -> http://localhost:16002, TP=1, DP=2
#   8b      -> GPUs 4,5    -> http://localhost:16003, TP=1, DP=2
#   32b     -> GPUs 6,7    -> http://localhost:16004, TP=1, DP=2
#
# Single-GPU servers use TP=1, DP=1. Multi-GPU servers use pure data
# parallelism, so TP stays 1 and DP equals the number of visible GPUs.
# All servers use context length 16384, static memory fraction 0.85, and
# max running requests 64 unless overridden by environment variables below.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SGLANG_IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang:latest}"
MODEL_ROOT="${MODEL_ROOT:-$HOME/models}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
HOST="${HOST:-0.0.0.0}"
SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-16384}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.85}"
SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-128}"
if [[ -z "${SGLANG_PIP_INSTALL+x}" ]]; then
  SGLANG_PIP_INSTALL="protobuf sentencepiece"
fi
if [[ -z "${SGLANG_32B_AWQ_PIP_INSTALL+x}" ]]; then
  SGLANG_32B_AWQ_PIP_INSTALL="vllm==0.7.2"
fi
SHM_SIZE="${SHM_SIZE:-32g}"
PULL_IMAGE="${PULL_IMAGE:-auto}"          # auto | always | never
REPLACE_EXISTING="${REPLACE_EXISTING:-1}" # 1 removes same-name containers first
WAIT_READY="${WAIT_READY:-1}"             # 1 waits for /v1/models after launch
READY_TIMEOUT="${READY_TIMEOUT:-900}"
POLL_INTERVAL="${POLL_INTERVAL:-2}"
DOCKER_RESTART_POLICY="${DOCKER_RESTART_POLICY:-no}"

MODEL_ROOT="${MODEL_ROOT/#\~/$HOME}"
HF_CACHE_DIR="${HF_CACHE_DIR/#\~/$HOME}"

MODEL_SPECS=(
  "0_6b|Qwen/Qwen3-0.6B|/models/0_6b|16000|0"
  "llama3b|meta-llama/Llama-3.2-3B-Instruct|/models/llama3b|16001|0,1,2,3,4,5,6,7"
  "7b|mistralai/Mistral-7B-Instruct-v0.3|/models/7b|16002|2,3"
  "8b|Qwen/Qwen3-8B|/models/8b|16003|4,5"
  "32b|Qwen/Qwen3-32B-AWQ|/models/32b|16004|6,7"
)

usage() {
  cat <<'EOF'
Usage:
  ./start_sglang_servers.sh [--image IMAGE] [start|restart|stop|status] [MODEL_TAG ...]
  ./start_sglang_servers.sh [start|restart|stop|status] [MODEL_TAG ...]
  ./start_sglang_servers.sh logs MODEL_TAG

Environment overrides:
  SGLANG_IMAGE            Docker image. Default: lmsysorg/sglang:latest
  MODEL_ROOT              Host model root. Default: ~/models
  HF_CACHE_DIR            Host Hugging Face cache. Default: ~/.cache/huggingface
  SGLANG_CONTEXT_LENGTH   Passed as --context-length. Default: 16384
  SGLANG_MEM_FRACTION_STATIC
                          Passed as --mem-fraction-static. Default: 0.85
  SGLANG_MAX_RUNNING_REQUESTS
                          Passed as --max-running-requests. Default: 64
  SGLANG_PIP_INSTALL      Python packages installed inside the container before
                          launch. Default: protobuf sentencepiece
                          Set to an empty string to skip.
  SGLANG_32B_AWQ_PIP_INSTALL
                          Extra Python packages installed only for the 32b
                          container. Default: vllm==0.7.2
                          Set to an empty string to skip.
  PULL_IMAGE              auto, always, or never. Default: auto
  REPLACE_EXISTING        Remove same-name containers before start. Default: 1
  WAIT_READY              Wait for /v1/models after launch. Default: 1
  READY_TIMEOUT           Readiness timeout in seconds. Default: 900
  DOCKER_RESTART_POLICY   Docker restart policy. Default: no

Model tags:
  0_6b, llama3b, 7b, 8b, 32b

Examples:
  ./start_sglang_servers.sh start 0_6b
  ./start_sglang_servers.sh start 0_6b llama3b 7b 8b 32b
  ./start_sglang_servers.sh --image ghcr.io/acme/sglang:cuda12 start 0_6b
EOF
}

container_name() {
  printf 'memarena-sglang-%s' "$1"
}

docker_exists() {
  docker inspect "$(container_name "$1")" >/dev/null 2>&1
}

docker_running() {
  local name
  name="$(container_name "$1")"
  [[ "$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null || true)" == "true" ]]
}

require_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "[sglang] ERROR: docker is not installed or not on PATH" >&2
    exit 2
  fi
  if ! docker info >/dev/null 2>&1; then
    echo "[sglang] ERROR: docker daemon is not reachable" >&2
    exit 2
  fi
}

ensure_image() {
  case "$PULL_IMAGE" in
    always)
      docker pull "$SGLANG_IMAGE"
      ;;
    auto)
      if ! docker image inspect "$SGLANG_IMAGE" >/dev/null 2>&1; then
        docker pull "$SGLANG_IMAGE"
      fi
      ;;
    never)
      ;;
    *)
      echo "[sglang] ERROR: PULL_IMAGE must be auto, always, or never" >&2
      exit 2
      ;;
  esac
}

validate_model_dirs() {
  local missing=0
  local spec tag served_model container_model_path port gpus local_path
  for spec in "${MODEL_SPECS[@]}"; do
    IFS='|' read -r tag served_model container_model_path port gpus <<<"$spec"
    local_path="${MODEL_ROOT}${container_model_path#/models}"
    if [[ ! -d "$local_path" ]]; then
      echo "[sglang] missing model directory for ${tag}: ${local_path}" >&2
      missing=1
    fi
  done
  if [[ "$missing" -ne 0 ]]; then
    exit 2
  fi
}

find_model_spec() {
  local wanted="$1"
  local spec tag served_model container_model_path port gpus
  FOUND_MODEL_SPEC=""
  for spec in "${MODEL_SPECS[@]}"; do
    IFS='|' read -r tag served_model container_model_path port gpus <<<"$spec"
    if [[ "$tag" == "$wanted" ]]; then
      FOUND_MODEL_SPEC="$spec"
      return 0
    fi
  done
  echo "[sglang] ERROR: unknown model tag: ${wanted}" >&2
  echo "[sglang] Valid tags: 0_6b, llama3b, 7b, 8b, 32b" >&2
  return 2
}

trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

REQUESTED_MODEL_SPECS=()

expand_requested_model_specs() {
  local raw part spec tag served_model container_model_path port gpus
  local -a pieces
  REQUESTED_MODEL_SPECS=()

  if [[ "$#" -eq 0 ]]; then
    REQUESTED_MODEL_SPECS=("${MODEL_SPECS[@]}")
    return 0
  fi

  for raw in "$@"; do
    IFS=',' read -r -a pieces <<<"$raw"
    for part in "${pieces[@]}"; do
      part="$(trim "$part")"
      [[ -n "$part" ]] || continue
      find_model_spec "$part"
      REQUESTED_MODEL_SPECS+=("$FOUND_MODEL_SPEC")
    done
  done

  if [[ "${#REQUESTED_MODEL_SPECS[@]}" -eq 0 ]]; then
    echo "[sglang] ERROR: no model tags requested" >&2
    exit 2
  fi

  for spec in "${REQUESTED_MODEL_SPECS[@]}"; do
    IFS='|' read -r tag served_model container_model_path port gpus <<<"$spec"
    :
  done
}

validate_one_model_dir() {
  local spec="$1"
  local tag served_model container_model_path port gpus local_path
  IFS='|' read -r tag served_model container_model_path port gpus <<<"$spec"
  local_path="${MODEL_ROOT}${container_model_path#/models}"
  if [[ ! -d "$local_path" ]]; then
    echo "[sglang] missing model directory for ${tag}: ${local_path}" >&2
    exit 2
  fi
}

validate_model_specs() {
  local spec
  for spec in "$@"; do
    validate_one_model_dir "$spec"
  done
}

stop_one() {
  local tag="$1"
  local name
  name="$(container_name "$tag")"
  if docker_exists "$tag"; then
    docker rm -f "$name" >/dev/null
    echo "[sglang] stopped ${name}"
  fi
}

stop_all() {
  local spec tag served_model container_model_path port gpus
  require_docker
  for spec in "${MODEL_SPECS[@]}"; do
    IFS='|' read -r tag served_model container_model_path port gpus <<<"$spec"
    stop_one "$tag"
  done
}

start_one() {
  local spec="$1"
  local tag served_model container_model_path port gpus name tp dp
  local gpu_ids=()
  local gpu_flag
  local env_args=()
  local container_pip_install

  IFS='|' read -r tag served_model container_model_path port gpus <<<"$spec"
  IFS=',' read -r -a gpu_ids <<<"$gpus"
  tp=1
  dp="${#gpu_ids[@]}"
  name="$(container_name "$tag")"
  gpu_flag="device=${gpus}"
  container_pip_install="${SGLANG_PIP_INSTALL:-}"
  if [[ "$tag" == "32b" && -n "${SGLANG_32B_AWQ_PIP_INSTALL:-}" ]]; then
    if [[ -n "$container_pip_install" ]]; then
      container_pip_install="${container_pip_install} ${SGLANG_32B_AWQ_PIP_INSTALL}"
    else
      container_pip_install="${SGLANG_32B_AWQ_PIP_INSTALL}"
    fi
  fi

  if docker_exists "$tag"; then
    if [[ "$REPLACE_EXISTING" == "1" ]]; then
      docker rm -f "$name" >/dev/null
    else
      echo "[sglang] ERROR: container already exists: ${name}" >&2
      echo "[sglang] Set REPLACE_EXISTING=1 or run: $0 stop" >&2
      exit 2
    fi
  fi

  if [[ -n "${HF_TOKEN:-}" ]]; then
    env_args+=(--env "HF_TOKEN=${HF_TOKEN}")
  fi
  if [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    env_args+=(--env "HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}")
  fi

  echo "[sglang] starting ${tag}: image=${SGLANG_IMAGE} GPUs=${gpus} TP=${tp} DP=${dp} port=${port} context=${SGLANG_CONTEXT_LENGTH} mem_fraction=${SGLANG_MEM_FRACTION_STATIC} max_running=${SGLANG_MAX_RUNNING_REQUESTS} pip_install=${container_pip_install:-<none>}"
  docker run -d \
    --name "$name" \
    --gpus "\"${gpu_flag}\"" \
    --ipc=host \
    --shm-size "$SHM_SIZE" \
    --restart "$DOCKER_RESTART_POLICY" \
    -p "${port}:${port}" \
    -v "${MODEL_ROOT}:/models:ro" \
    -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
    --env "SGLANG_PIP_INSTALL=${container_pip_install}" \
    "${env_args[@]}" \
    "$SGLANG_IMAGE" \
    bash -lc 'set -euo pipefail
if [[ -n "${SGLANG_PIP_INSTALL:-}" ]]; then
  python3 -m pip install --no-cache-dir ${SGLANG_PIP_INSTALL}
fi
exec "$@"
' bash \
      python3 -m sglang.launch_server \
      --model-path "$container_model_path" \
      --served-model-name "$served_model" \
      --host "$HOST" \
      --port "$port" \
      --tp "$tp" \
      --dp "$dp" \
      --context-length "$SGLANG_CONTEXT_LENGTH" \
      --mem-fraction-static "$SGLANG_MEM_FRACTION_STATIC" \
      --max-running-requests "$SGLANG_MAX_RUNNING_REQUESTS" >/dev/null
}

wait_ready_one() {
  local tag="$1"
  local port="$2"
  local name deadline now
  name="$(container_name "$tag")"
  deadline=$((SECONDS + READY_TIMEOUT))

  if ! command -v curl >/dev/null 2>&1; then
    echo "[sglang] curl not found; skipping readiness wait"
    return 0
  fi

  while true; do
    if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      echo "[sglang] ready ${tag}: http://localhost:${port}"
      return 0
    fi

    if ! docker_running "$tag"; then
      echo "[sglang] ERROR: ${name} exited before becoming ready" >&2
      docker logs --tail 80 "$name" >&2 || true
      return 1
    fi

    now="$SECONDS"
    if [[ "$now" -ge "$deadline" ]]; then
      echo "[sglang] ERROR: timeout waiting for ${tag} on port ${port}" >&2
      docker logs --tail 80 "$name" >&2 || true
      return 1
    fi
    sleep "$POLL_INTERVAL"
  done
}

wait_ready_many() {
  local spec tag served_model container_model_path port gpus
  local -a pids=()
  local -a labels=()
  local i failed=0

  for spec in "$@"; do
    IFS='|' read -r tag served_model container_model_path port gpus <<<"$spec"
    wait_ready_one "$tag" "$port" &
    pids+=("$!")
    labels+=("$tag")
  done

  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      echo "[sglang] ERROR: readiness failed for ${labels[$i]}" >&2
      failed=1
    fi
  done

  return "$failed"
}

start_many() {
  local spec tag served_model container_model_path port gpus
  local -a pids=()
  local -a labels=()
  local i failed=0

  for spec in "$@"; do
    IFS='|' read -r tag served_model container_model_path port gpus <<<"$spec"
    start_one "$spec" &
    pids+=("$!")
    labels+=("$tag")
  done

  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      echo "[sglang] ERROR: failed to start ${labels[$i]}" >&2
      failed=1
    fi
  done

  return "$failed"
}

print_endpoints() {
  local spec tag served_model container_model_path port gpus
  echo "[sglang] endpoints:"
  for spec in "$@"; do
    IFS='|' read -r tag served_model container_model_path port gpus <<<"$spec"
    printf '  %-8s %-40s http://localhost:%s\n' "$tag" "$served_model" "$port"
  done
}

start_all() {
  local spec tag served_model container_model_path port gpus
  require_docker
  validate_model_dirs
  mkdir -p "$HF_CACHE_DIR"
  ensure_image

  start_many "${MODEL_SPECS[@]}"

  if [[ "$WAIT_READY" == "1" ]]; then
    wait_ready_many "${MODEL_SPECS[@]}"
  fi

  print_endpoints "${MODEL_SPECS[@]}"
}

start_selected() {
  local spec

  require_docker
  expand_requested_model_specs "$@"
  validate_model_specs "${REQUESTED_MODEL_SPECS[@]}"
  mkdir -p "$HF_CACHE_DIR"
  ensure_image

  start_many "${REQUESTED_MODEL_SPECS[@]}"

  if [[ "$WAIT_READY" == "1" ]]; then
    wait_ready_many "${REQUESTED_MODEL_SPECS[@]}"
  fi
  print_endpoints "${REQUESTED_MODEL_SPECS[@]}"
}

status_all() {
  local spec tag served_model container_model_path port gpus name state
  require_docker
  printf '%-8s %-24s %-10s %-22s %s\n' "TAG" "CONTAINER" "STATE" "ENDPOINT" "GPUS"
  for spec in "${MODEL_SPECS[@]}"; do
    IFS='|' read -r tag served_model container_model_path port gpus <<<"$spec"
    name="$(container_name "$tag")"
    state="$(docker inspect -f '{{.State.Status}}' "$name" 2>/dev/null || printf 'missing')"
    printf '%-8s %-24s %-10s %-22s %s\n' "$tag" "$name" "$state" "http://localhost:${port}" "$gpus"
  done
}

status_selected() {
  local spec tag served_model container_model_path port gpus name state

  require_docker
  expand_requested_model_specs "$@"
  printf '%-8s %-24s %-10s %-22s %s\n' "TAG" "CONTAINER" "STATE" "ENDPOINT" "GPUS"
  for spec in "${REQUESTED_MODEL_SPECS[@]}"; do
    IFS='|' read -r tag served_model container_model_path port gpus <<<"$spec"
    name="$(container_name "$tag")"
    state="$(docker inspect -f '{{.State.Status}}' "$name" 2>/dev/null || printf 'missing')"
    printf '%-8s %-24s %-10s %-22s %s\n' "$tag" "$name" "$state" "http://localhost:${port}" "$gpus"
  done
}

stop_selected() {
  local spec tag served_model container_model_path port gpus

  require_docker
  expand_requested_model_specs "$@"
  for spec in "${REQUESTED_MODEL_SPECS[@]}"; do
    IFS='|' read -r tag served_model container_model_path port gpus <<<"$spec"
    stop_one "$tag"
  done
}

logs_one() {
  local tag="${1:-}"
  if [[ -z "$tag" ]]; then
    echo "[sglang] ERROR: logs requires a model tag" >&2
    usage >&2
    exit 2
  fi
  require_docker
  docker logs -f "$(container_name "$tag")"
}

parse_args() {
  local parsed=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --image)
        if [[ -z "${2:-}" ]]; then
          echo "[sglang] ERROR: --image requires a Docker image value" >&2
          exit 2
        fi
        SGLANG_IMAGE="$2"
        shift 2
        ;;
      --image=*)
        SGLANG_IMAGE="${1#*=}"
        if [[ -z "$SGLANG_IMAGE" ]]; then
          echo "[sglang] ERROR: --image requires a Docker image value" >&2
          exit 2
        fi
        shift
        ;;
      --)
        shift
        while [[ $# -gt 0 ]]; do
          parsed+=("$1")
          shift
        done
        ;;
      *)
        parsed+=("$1")
        shift
        ;;
    esac
  done
  PARSED_ARGS=("${parsed[@]}")
}

parse_args "$@"
set -- "${PARSED_ARGS[@]}"

action="${1:-start}"
case "$action" in
  start)
    shift || true
    start_selected "$@"
    ;;
  restart)
    shift || true
    stop_selected "$@"
    start_selected "$@"
    ;;
  stop)
    shift || true
    stop_selected "$@"
    ;;
  status)
    shift || true
    status_selected "$@"
    ;;
  logs)
    logs_one "${2:-}"
    ;;
  help|--help|-h)
    usage
    ;;
  *)
    echo "[sglang] ERROR: unknown action: ${action}" >&2
    usage >&2
    exit 2
    ;;
esac
