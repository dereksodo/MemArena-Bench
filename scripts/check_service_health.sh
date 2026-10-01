#!/usr/bin/env bash
# Check the 5-reader MemArena service matrix:
#   SGLang / Memobase / MemOS for each reader model.

set -euo pipefail

HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-8}"
HEALTH_CONNECT_TIMEOUT="${HEALTH_CONNECT_TIMEOUT:-2}"
HEALTH_PARALLEL="${HEALTH_PARALLEL:-1}"
MEMOBASE_API_TOKEN="${MEMOBASE_API_TOKEN:-secret}"
MEMOS_API_KEY="${MEMOS_API_KEY:-EMPTY}"

MODEL_SPECS=(
  "0_6b|Qwen/Qwen3-0.6B|16000|18100|18101"
  "llama3b|meta-llama/Llama-3.2-3B-Instruct|16001|18102|18103"
  "7b|mistralai/Mistral-7B-Instruct-v0.3|16002|18104|18105"
  "8b|Qwen/Qwen3-8B|16003|18106|18107"
  "32b|Qwen/Qwen3-32B-AWQ|16004|18108|18109"
)

usage() {
  cat <<'EOF'
Usage:
  scripts/check_service_health.sh [MODEL_TAG ...]

Checks, for each requested reader:
  sglang   GET http://localhost:<16000-16004>/v1/models
  memobase GET http://localhost:<18100,18102,...>/api/v1/healthcheck
  memos    GET http://localhost:<18101,18103,...>/health

Default model tags:
  0_6b llama3b 7b 8b 32b

Aliases:
  3b -> llama3b

Environment overrides:
  HEALTH_TIMEOUT          Per-request max time in seconds. Default: 8
  HEALTH_CONNECT_TIMEOUT  Per-request connect timeout in seconds. Default: 2
  HEALTH_PARALLEL=0       Run checks serially. Default: 1
  MEMOBASE_API_TOKEN      Memobase token. Default: secret
  MEMOS_API_KEY           MemOS API key. Default: EMPTY

Examples:
  scripts/check_service_health.sh
  scripts/check_service_health.sh 0_6b 3b 7b
  HEALTH_PARALLEL=0 scripts/check_service_health.sh 8b,32b
EOF
}

die() {
  echo "[health] ERROR: $*" >&2
  exit 2
}

trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

container_name() {
  local tag="$1"
  case "$2" in
    sglang) printf 'memarena-sglang-%s' "$tag" ;;
    memobase) printf 'memarena_%s_memobase' "$tag" ;;
    memos) printf 'memarena_%s_memos' "$tag" ;;
    *) printf '-' ;;
  esac
}

find_model_spec() {
  local wanted="$1"
  local spec tag model sglang_port memobase_port memos_port

  case "$wanted" in
    3b) wanted="llama3b" ;;
  esac

  FOUND_MODEL_SPEC=""
  for spec in "${MODEL_SPECS[@]}"; do
    IFS='|' read -r tag model sglang_port memobase_port memos_port <<<"$spec"
    if [[ "$tag" == "$wanted" ]]; then
      FOUND_MODEL_SPEC="$spec"
      return 0
    fi
  done
  die "unknown model tag: $wanted (valid: 0_6b, llama3b/3b, 7b, 8b, 32b)"
}

REQUESTED_MODEL_SPECS=()

expand_requested_model_specs() {
  local raw part
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

  [[ "${#REQUESTED_MODEL_SPECS[@]}" -gt 0 ]] || die "no model tags requested"
}

auth_value() {
  local service="$1"
  case "$service" in
    memobase)
      if [[ "${MEMOBASE_API_TOKEN,,}" == bearer\ * ]]; then
        printf '%s' "$MEMOBASE_API_TOKEN"
      else
        printf 'Bearer %s' "$MEMOBASE_API_TOKEN"
      fi
      ;;
    memos)
      printf '%s' "$MEMOS_API_KEY"
      ;;
    *)
      printf ''
      ;;
  esac
}

clean_one_line() {
  tr '\t\r\n' ' ' | sed 's/[[:space:]][[:space:]]*/ /g' | cut -c1-180
}

check_http() {
  local idx="$1"
  local tag="$2"
  local service="$3"
  local url="$4"
  local expected="$5"
  local outfile="$6"
  local auth header_args=()
  local body err result curl_rc code elapsed latency_ms status detail

  body="$(mktemp)"
  err="$(mktemp)"
  auth="$(auth_value "$service")"
  if [[ -n "$auth" && "$service" != "sglang" ]]; then
    header_args=(-H "Authorization: ${auth}")
  fi

  set +e
  result="$(
    curl -sS \
      --connect-timeout "$HEALTH_CONNECT_TIMEOUT" \
      --max-time "$HEALTH_TIMEOUT" \
      -o "$body" \
      -w $'%{http_code}\t%{time_total}' \
      "${header_args[@]}" \
      "$url" \
      2>"$err"
  )"
  curl_rc=$?
  set -e

  code="${result%%$'\t'*}"
  elapsed="${result#*$'\t'}"
  if [[ "$elapsed" == "$result" ]]; then
    elapsed="0"
  fi
  latency_ms="$(awk -v t="$elapsed" 'BEGIN { printf "%.0f", t * 1000 }')"

  status="FAIL"
  detail=""
  if [[ "$curl_rc" -ne 0 ]]; then
    detail="curl rc=${curl_rc}: $(clean_one_line <"$err")"
  elif [[ ! "$code" =~ ^[0-9][0-9][0-9]$ ]]; then
    detail="invalid http_code=${code:-empty}"
  elif [[ "$code" -lt 200 || "$code" -ge 300 ]]; then
    detail="HTTP ${code}: $(clean_one_line <"$body")"
  elif [[ "$expected" == "sglang_models" ]] && ! grep -q '"data"' "$body"; then
    detail="HTTP ${code}, but response does not look like OpenAI /v1/models"
  else
    status="OK"
    detail="HTTP ${code}"
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$idx" "$status" "$tag" "$service" "$latency_ms" "$url" "$detail" >"$outfile"

  rm -f "$body" "$err"
}

main() {
  local arg spec tag model sglang_port memobase_port memos_port
  local tmp_dir idx=0 failed=0
  local -a pids=()
  local -a files=()
  local i file status

  for arg in "$@"; do
    case "$arg" in
      --help|-h|help)
        usage
        return 0
        ;;
    esac
  done

  command -v curl >/dev/null 2>&1 || die "curl is not installed or not on PATH"
  command -v awk >/dev/null 2>&1 || die "awk is not installed or not on PATH"

  expand_requested_model_specs "$@"
  tmp_dir="$(mktemp -d)"
  trap 'rm -rf "$tmp_dir"' EXIT

  for spec in "${REQUESTED_MODEL_SPECS[@]}"; do
    IFS='|' read -r tag model sglang_port memobase_port memos_port <<<"$spec"

    idx=$((idx + 1))
    file="$tmp_dir/${idx}.tsv"
    files+=("$file")
    if [[ "$HEALTH_PARALLEL" == "0" ]]; then
      check_http "$idx" "$tag" "sglang" "http://localhost:${sglang_port}/v1/models" "sglang_models" "$file"
    else
      check_http "$idx" "$tag" "sglang" "http://localhost:${sglang_port}/v1/models" "sglang_models" "$file" &
      pids+=("$!")
    fi

    idx=$((idx + 1))
    file="$tmp_dir/${idx}.tsv"
    files+=("$file")
    if [[ "$HEALTH_PARALLEL" == "0" ]]; then
      check_http "$idx" "$tag" "memobase" "http://localhost:${memobase_port}/api/v1/healthcheck" "plain" "$file"
    else
      check_http "$idx" "$tag" "memobase" "http://localhost:${memobase_port}/api/v1/healthcheck" "plain" "$file" &
      pids+=("$!")
    fi

    idx=$((idx + 1))
    file="$tmp_dir/${idx}.tsv"
    files+=("$file")
    if [[ "$HEALTH_PARALLEL" == "0" ]]; then
      check_http "$idx" "$tag" "memos" "http://localhost:${memos_port}/health" "plain" "$file"
    else
      check_http "$idx" "$tag" "memos" "http://localhost:${memos_port}/health" "plain" "$file" &
      pids+=("$!")
    fi
  done

  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || true
  done

  printf '%-4s %-5s %-8s %-8s %-10s %-36s %s\n' \
    "IDX" "OK" "MODEL" "SERVICE" "LATENCY" "URL" "DETAIL"
  for file in "${files[@]}"; do
    if [[ ! -f "$file" ]]; then
      failed=1
      continue
    fi
    IFS=$'\t' read -r i status tag service latency_ms url detail <"$file"
    printf '%-4s %-5s %-8s %-8s %7sms  %-36s %s\n' \
      "$i" "$status" "$tag" "$service" "$latency_ms" "$url" "$detail"
    [[ "$status" == "OK" ]] || failed=1
  done

  if [[ "$failed" -eq 0 ]]; then
    echo "[health] OK: all requested services are reachable"
  else
    echo "[health] FAIL: at least one service is unhealthy" >&2
  fi
  return "$failed"
}

main "$@"
