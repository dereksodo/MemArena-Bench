#!/usr/bin/env bash
# Run a backend x reader-model x trial accuracy matrix.
#
# Model file format:
#   model_tag|model_name|endpoint[|memobase_url|memos_url]
#
# Blank lines and lines beginning with "#" are ignored.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_eval_matrix.sh --run-dir PATH --models-file models.tsv [options]

Required:
  --run-dir PATH             Completed MASim run directory.

Common options:
  --paper-full              Use the paper full-matrix preset:
                             5 readers x 5 backends x 3 trials.
  --models-file PATH         Pipe-delimited model list:
                             model_tag|model_name|endpoint
                             or:
                             model_tag|model_name|endpoint|memobase_url|memos_url
                             Default: models.tsv
  --init-models-file         Write a starter models file and exit.
  --backends LIST            Comma-separated backends.
                             Default: vanilla,oracle,inmem,baseline_simplerag,baseline_session
  --trials LIST              Comma-separated trial names. Default: s1,s2,s3
  --judge-preset PRESET      remote, local, both, custom, or none. Default: local
  --out-dir PATH             Output root. Default: out/accuracy_matrix
  --answer-concurrency N     Passed to run_accuracy.py. Default: 32.
  --eval-concurrency N       Passed to run_accuracy.py. Default: 512.
  --model-parallelism N      Number of reader models to run concurrently.
                             Default: 1, or 5 with --paper-full.
  --memory-cache-dir PATH    Where memobase/memos cache JSONL files are built.
                             Default: <out-dir>/memory_cache.
  --memory-cache-config TAG  Cache metadata tag. Default: A_paired.
  --memory-cache-concurrency N
                             Parallel agents per day while building cache.
                             Default: 32.
  --memory-cache-max-tokens N
                             Extractor max_tokens while building cache.
  --temperature FLOAT        Passed to run_accuracy.py. Default: 0.3
  --qa-limit N               Passed to run_accuracy.py.
  --message-limit N          Passed to run_accuracy.py.
  --user-limit N             Passed to run_accuracy.py.
  --dimensions LIST          Comma-separated dimensions passed as repeated values.
  --top-k N                  Passed to run_accuracy.py.
  --context-length N         Passed to run_accuracy.py.
  --max-tokens N             Passed to run_accuracy.py.
  --max-previous-context N   Passed to run_accuracy.py.
  --remote-judge-model NAME  Passed to run_accuracy.py.
  --remote-judge-endpoint U  Passed to run_accuracy.py.
  --local-judge-model NAME   Passed to run_accuracy.py.
  --local-judge-url URL      Passed to run_accuracy.py.
  --judge-model NAME         For --judge-preset custom.
  --judge-endpoint URL       For --judge-preset custom.
  --judge-api-key KEY        For remote/custom judges.
  --local-judge-api-key KEY  For local judges. Default: EMPTY.
  --test                    Run the full pipeline shape with no LLM calls.
                             Every answer-side LLM/OpenClaw call returns
                             "I don't know"; external memory services are stubbed.
  --force                    Recompute existing cells.
  --print-only               Print commands without executing them.

Environment variables with the same meaning are also accepted:
  RUN_DIR, MODELS_FILE, BACKENDS, TRIALS, JUDGE_PRESET, OUT_DIR,
  ANSWER_CONCURRENCY, EVAL_CONCURRENCY, MODEL_PARALLELISM, TEMPERATURE.
  MEMORY_CACHE_DIR, MEMORY_CACHE_CONFIG, MEMORY_CACHE_CONCURRENCY,
  MEMORY_CACHE_MAX_TOKENS.
  MEMOBASE_API_TOKEN and MEMOS_API_KEY are used for per-model structured
  memory endpoints when present in the models file.
  PYTHON can override the interpreter. Default: .venv/bin/python when present.

Example:
  scripts/run_eval_matrix.sh \
      --run-dir "${MEMARENA_RUN_DIR:-$HOME/memarena_spark_runs/test_5a10d/run}" \
      --models-file models_5a10d.tsv \
      --backends vanilla,oracle,inmem,baseline_simplerag,baseline_session \
      --trials s1,s2,s3 \
      --judge-preset local \
      --model-parallelism 5 \
      --answer-concurrency 32 \
      --eval-concurrency 32 \
      --out-dir out/accuracy_5a10d
EOF
}

die() {
  echo "[run-eval-matrix] ERROR: $*" >&2
  exit 2
}

write_models_template() {
  local path="$1"
  if [[ -e "$path" ]]; then
    die "models file already exists: $path"
  fi
  mkdir -p "$(dirname "$path")"
  cat >"$path" <<'EOF'
# model_tag|model_name|endpoint[|memobase_url|memos_url]
# Replace these rows with the reader models exposed by your OpenAI-compatible servers.
qwen3_8b|qwen3|http://localhost:16000
# llama3_8b|meta-llama/Llama-3.1-8B-Instruct|http://localhost:16001
# mistral_7b|mistralai/Mistral-7B-Instruct-v0.3|http://localhost:16002
EOF
  echo "[run-eval-matrix] wrote starter models file: $path"
}

trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

trial_seed() {
  local trial="$1"
  local idx="$2"
  if [[ "$trial" =~ ^s([0-9]+)$ ]]; then
    echo $((1000 + BASH_REMATCH[1]))
  else
    echo $((1000 + idx))
  fi
}

print_cmd() {
  local quoted=()
  local part
  for part in "$@"; do
    quoted+=("$(printf '%q' "$part")")
  done
  printf '%s\n' "${quoted[*]}"
}

safe_name() {
  printf '%s' "$1" | tr -c 'A-Za-z0-9_.-' '_'
}

write_status() {
  local path="$1"
  local tmp
  shift
  tmp="${path}.$$.tmp"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$@" >"$tmp"
  mv "$tmp" "$path"
}

progress_bar() {
  local done_count="$1"
  local total_count="$2"
  local width="${3:-24}"
  local filled
  local empty
  if [[ "$total_count" -le 0 ]]; then
    total_count=1
  fi
  filled=$((done_count * width / total_count))
  empty=$((width - filled))
  printf '|'
  if [[ "$filled" -gt 0 ]]; then
    printf '%*s' "$filled" '' | tr ' ' '#'
  fi
  if [[ "$empty" -gt 0 ]]; then
    printf '%*s' "$empty" '' | tr ' ' '-'
  fi
  printf '|'
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

RUN_DIR="${RUN_DIR:-}"
MODELS_FILE="${MODELS_FILE:-models.tsv}"
BACKENDS="${BACKENDS:-vanilla,oracle,inmem,baseline_simplerag,baseline_session}"
TRIALS="${TRIALS:-s1,s2,s3}"
JUDGE_PRESET="${JUDGE_PRESET:-local}"
OUT_DIR="${OUT_DIR:-out/accuracy_matrix}"
ANSWER_CONCURRENCY="${ANSWER_CONCURRENCY:-32}"
EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-512}"
MEMORY_CACHE_DIR="${MEMORY_CACHE_DIR:-}"
MEMORY_CACHE_CONFIG="${MEMORY_CACHE_CONFIG:-A_paired}"
MEMORY_CACHE_CONCURRENCY="${MEMORY_CACHE_CONCURRENCY:-32}"
MEMORY_CACHE_MAX_TOKENS="${MEMORY_CACHE_MAX_TOKENS:-}"
MEMORY_CACHE_EXTRACTOR_API_KEY="${MEMORY_CACHE_EXTRACTOR_API_KEY:-EMPTY}"
MODEL_PARALLELISM_SET=0
if [[ -n "${MODEL_PARALLELISM:-}" ]]; then
  MODEL_PARALLELISM_SET=1
fi
MODEL_PARALLELISM="${MODEL_PARALLELISM:-1}"
TEMPERATURE="${TEMPERATURE:-0.3}"
FORCE=0
PRINT_ONLY=0
INIT_MODELS_FILE=0
TEST_MODE=0
PRESET=""
MODELS_FILE_SET=0
BACKENDS_SET=0
TRIALS_SET=0
JUDGE_PRESET_SET=0

QA_LIMIT=""
MESSAGE_LIMIT=""
USER_LIMIT=""
DIMENSIONS=""
TOP_K=""
D6_ARM=""
EVAL_CONFIG_PATH=""
CONTEXT_LENGTH=""
MAX_TOKENS=""
MAX_PREVIOUS_CONTEXT=""
REMOTE_JUDGE_MODEL=""
REMOTE_JUDGE_ENDPOINT=""
LOCAL_JUDGE_MODEL=""
LOCAL_JUDGE_URL=""
JUDGE_MODEL=""
JUDGE_ENDPOINT=""
JUDGE_API_KEY=""
LOCAL_JUDGE_API_KEY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --paper-full|--preset-paper-full) PRESET="paper-full"; shift ;;
    --run-dir) RUN_DIR="${2:-}"; shift 2 ;;
    --models-file) MODELS_FILE="${2:-}"; MODELS_FILE_SET=1; shift 2 ;;
    --init-models-file) INIT_MODELS_FILE=1; shift ;;
    --backends) BACKENDS="${2:-}"; BACKENDS_SET=1; shift 2 ;;
    --trials) TRIALS="${2:-}"; TRIALS_SET=1; shift 2 ;;
    --judge-preset) JUDGE_PRESET="${2:-}"; JUDGE_PRESET_SET=1; shift 2 ;;
    --out-dir) OUT_DIR="${2:-}"; shift 2 ;;
    --answer-concurrency) ANSWER_CONCURRENCY="${2:-}"; shift 2 ;;
    --eval-concurrency) EVAL_CONCURRENCY="${2:-}"; shift 2 ;;
    --model-parallelism) MODEL_PARALLELISM="${2:-}"; MODEL_PARALLELISM_SET=1; shift 2 ;;
    --memory-cache-dir) MEMORY_CACHE_DIR="${2:-}"; shift 2 ;;
    --memory-cache-config) MEMORY_CACHE_CONFIG="${2:-}"; shift 2 ;;
    --memory-cache-concurrency) MEMORY_CACHE_CONCURRENCY="${2:-}"; shift 2 ;;
    --memory-cache-max-tokens) MEMORY_CACHE_MAX_TOKENS="${2:-}"; shift 2 ;;
    --temperature) TEMPERATURE="${2:-}"; shift 2 ;;
    --qa-limit) QA_LIMIT="${2:-}"; shift 2 ;;
    --message-limit) MESSAGE_LIMIT="${2:-}"; shift 2 ;;
    --user-limit) USER_LIMIT="${2:-}"; shift 2 ;;
    --dimensions) DIMENSIONS="${2:-}"; shift 2 ;;
    --top-k) TOP_K="${2:-}"; shift 2 ;;
    --d6-arm) D6_ARM="${2:-}"; shift 2 ;;
    --eval-config) EVAL_CONFIG_PATH="${2:-}"; shift 2 ;;
    --context-length) CONTEXT_LENGTH="${2:-}"; shift 2 ;;
    --max-tokens) MAX_TOKENS="${2:-}"; shift 2 ;;
    --max-previous-context) MAX_PREVIOUS_CONTEXT="${2:-}"; shift 2 ;;
    --remote-judge-model) REMOTE_JUDGE_MODEL="${2:-}"; shift 2 ;;
    --remote-judge-endpoint) REMOTE_JUDGE_ENDPOINT="${2:-}"; shift 2 ;;
    --local-judge-model) LOCAL_JUDGE_MODEL="${2:-}"; shift 2 ;;
    --local-judge-url) LOCAL_JUDGE_URL="${2:-}"; shift 2 ;;
    --judge-model) JUDGE_MODEL="${2:-}"; shift 2 ;;
    --judge-endpoint) JUDGE_ENDPOINT="${2:-}"; shift 2 ;;
    --judge-api-key) JUDGE_API_KEY="${2:-}"; shift 2 ;;
    --local-judge-api-key) LOCAL_JUDGE_API_KEY="${2:-}"; shift 2 ;;
    --test) TEST_MODE=1; shift ;;
    --force) FORCE=1; shift ;;
    --print-only) PRINT_ONLY=1; shift ;;
    *) die "unknown argument: $1" ;;
  esac
done

if [[ "$PRESET" == "paper-full" ]]; then
  [[ "$MODELS_FILE_SET" -eq 0 ]] && MODELS_FILE="$REPO_ROOT/config/eval_matrix_paper_models.tsv"
  [[ "$BACKENDS_SET" -eq 0 ]] && BACKENDS="vanilla,inmem,oracle,memobase,memos"
  [[ "$TRIALS_SET" -eq 0 ]] && TRIALS="s2,s3,s4"
  [[ "$JUDGE_PRESET_SET" -eq 0 ]] && JUDGE_PRESET="remote"
  [[ "$MODEL_PARALLELISM_SET" -eq 0 ]] && MODEL_PARALLELISM=5
fi

if [[ "$OUT_DIR" != /* ]]; then
  OUT_DIR="$REPO_ROOT/$OUT_DIR"
fi
if [[ -z "$MEMORY_CACHE_DIR" ]]; then
  MEMORY_CACHE_DIR="$OUT_DIR/memory_cache"
elif [[ "$MEMORY_CACHE_DIR" != /* ]]; then
  MEMORY_CACHE_DIR="$REPO_ROOT/$MEMORY_CACHE_DIR"
fi

if [[ "$INIT_MODELS_FILE" -eq 1 ]]; then
  write_models_template "$MODELS_FILE"
  exit 0
fi

[[ -n "$RUN_DIR" ]] || die "--run-dir is required"
[[ -d "$RUN_DIR" ]] || die "run directory does not exist: $RUN_DIR"

if [[ ! -e "$MODELS_FILE" ]]; then
  write_models_template "$MODELS_FILE"
  die "edit $MODELS_FILE with your model endpoints, then rerun this command"
fi
[[ -r "$MODELS_FILE" ]] || die "models file is not readable: $MODELS_FILE"

IFS=',' read -r -a BACKEND_ARR <<<"$BACKENDS"
IFS=',' read -r -a TRIAL_ARR <<<"$TRIALS"

MODEL_LINES=()
while IFS= read -r line || [[ -n "$line" ]]; do
  line="$(trim "$line")"
  [[ -z "$line" || "${line:0:1}" == "#" ]] && continue
  MODEL_LINES+=("$line")
done <"$MODELS_FILE"

[[ "${#MODEL_LINES[@]}" -gt 0 ]] || die "models file has no active rows: $MODELS_FILE"
[[ "${#BACKEND_ARR[@]}" -gt 0 ]] || die "no backends configured"
[[ "${#TRIAL_ARR[@]}" -gt 0 ]] || die "no trials configured"
[[ "$MODEL_PARALLELISM" =~ ^[0-9]+$ ]] || die "--model-parallelism must be a positive integer"
[[ "$MODEL_PARALLELISM" -ge 1 ]] || die "--model-parallelism must be >= 1"
case "$MEMORY_CACHE_CONFIG" in
  A_paired|B_remote) ;;
  *) die "--memory-cache-config must be A_paired or B_remote" ;;
esac

TOTAL=$(( ${#MODEL_LINES[@]} * ${#BACKEND_ARR[@]} * ${#TRIAL_ARR[@]} ))
PER_MODEL_TOTAL=$(( ${#BACKEND_ARR[@]} * ${#TRIAL_ARR[@]} ))
echo "[run-eval-matrix] repo: $REPO_ROOT"
echo "[run-eval-matrix] run dir: $RUN_DIR"
echo "[run-eval-matrix] output dir: $OUT_DIR"
echo "[run-eval-matrix] judge preset: $JUDGE_PRESET"
echo "[run-eval-matrix] preset: ${PRESET:-custom}"
echo "[run-eval-matrix] test mode: $TEST_MODE"
echo "[run-eval-matrix] model parallelism: $MODEL_PARALLELISM"
echo "[run-eval-matrix] memory cache dir: $MEMORY_CACHE_DIR"
echo "[run-eval-matrix] memory cache config: $MEMORY_CACHE_CONFIG"
echo "[run-eval-matrix] cells: ${#BACKEND_ARR[@]} backends x ${#MODEL_LINES[@]} models x ${#TRIAL_ARR[@]} trials = $TOTAL"
echo "[run-eval-matrix] models file: $MODELS_FILE"
if [[ "$PRINT_ONLY" -eq 1 ]]; then
  echo "[run-eval-matrix] print-only mode: commands will not be executed"
fi

mkdir -p "$OUT_DIR"
MATRIX_MANIFEST="$OUT_DIR/run_eval_matrix_manifest.tsv"
MANIFEST_HEADER='cell	total	backend	model_tag	model_name	endpoint	trial	seed	namespace	judge_preset	test_mode	returncode'
printf '%s\n' "$MANIFEST_HEADER" >"$MATRIX_MANIFEST"
echo "[run-eval-matrix] manifest: $MATRIX_MANIFEST"

parse_model_line() {
  local model_line="$1"
  local raw_model_tag
  local raw_model_name
  local raw_endpoint
  local raw_memobase_url
  local raw_memos_url
  local raw_extra
  IFS='|' read -r raw_model_tag raw_model_name raw_endpoint raw_memobase_url raw_memos_url raw_extra <<<"$model_line"
  [[ -z "${raw_extra:-}" ]] || die "invalid model row, expected three or five pipe-delimited fields: $model_line"
  if [[ -n "${raw_memobase_url:-}" && -z "${raw_memos_url:-}" ]] || [[ -z "${raw_memobase_url:-}" && -n "${raw_memos_url:-}" ]]; then
    die "invalid model row, memobase_url and memos_url must be provided together: $model_line"
  fi
  PARSED_MODEL_TAG="$(trim "$raw_model_tag")"
  PARSED_MODEL_NAME="$(trim "$raw_model_name")"
  PARSED_ENDPOINT="$(trim "$raw_endpoint")"
  PARSED_MEMOBASE_URL="$(trim "${raw_memobase_url:-}")"
  PARSED_MEMOS_URL="$(trim "${raw_memos_url:-}")"
  [[ -n "$PARSED_MODEL_TAG" && -n "$PARSED_MODEL_NAME" && -n "$PARSED_ENDPOINT" ]] || die "invalid model row: $model_line"
}

run_cell() {
  local cell="$1"
  local backend="$2"
  local model_tag="$3"
  local model_name="$4"
  local endpoint="$5"
  local trial="$6"
  local seed="$7"
  local memobase_url="$8"
  local memos_url="$9"
  local manifest_path="${10}"
  local output_mode="${11}"
  local log_path="${12:-}"
  local namespace
  local env_vars=()
  local cmd_system="$backend"
  local needs_memory_cache=0
  local cache_path=""
  local cache_dir=""
  local model_safe
  local trial_safe
  local backend_safe
  local build_cmd=()
  local rc
  namespace="$(safe_name "${backend}_${model_tag}_${trial}")"
  model_safe="$(safe_name "$model_tag")"
  trial_safe="$(safe_name "$trial")"
  backend_safe="$(safe_name "$backend")"

  # The "baseline_simplerag" cell is the BM25 in-memory retriever baseline.
  # Keep the public backend/namespace label for reporting, but execute the
  # eval pipeline through the inmem system so OpenClaw is never involved.
  if [[ "$backend" == "baseline_simplerag" ]]; then
    cmd_system="inmem"
  fi

  if [[ "$backend" == "memobase" && -n "$memobase_url" ]]; then
    env_vars+=("MEMOBASE_BASE_URL=$memobase_url" "MEMOBASE_API_TOKEN=${MEMOBASE_API_TOKEN:-secret}")
  elif [[ "$backend" == "memos" && -n "$memos_url" ]]; then
    env_vars+=("MEMOS_BASE_URL=$memos_url" "MEMOS_API_KEY=${MEMOS_API_KEY:-EMPTY}")
  fi

  if [[ "$TEST_MODE" -eq 0 && ( "$backend" == "memobase" || "$backend" == "memos" || "$backend" == "memsearch" ) ]]; then
    needs_memory_cache=1
    cmd_system="memory_cache"
    cache_dir="$MEMORY_CACHE_DIR/$backend_safe/$model_safe/$trial_safe"
    cache_path="$cache_dir/memcache_${backend_safe}_${MEMORY_CACHE_CONFIG}_${model_safe}_${trial_safe}.jsonl"
    build_cmd=(
      "$PYTHON_BIN" "$REPO_ROOT/scripts/reproduce/build_memory_cache.py"
      --system "$backend"
      --run-dir "$RUN_DIR"
      --extractor-model "$model_name"
      --extractor-endpoint "$endpoint"
      --extractor-api-key "$MEMORY_CACHE_EXTRACTOR_API_KEY"
      --extractor-config "$MEMORY_CACHE_CONFIG"
      --output "$cache_path"
      --trial-seed "$seed"
    )
    [[ -n "$TOP_K" ]] && build_cmd+=(--top-k "$TOP_K")
    [[ -n "$QA_LIMIT" ]] && build_cmd+=(--qa-limit "$QA_LIMIT")
    [[ -n "$USER_LIMIT" ]] && build_cmd+=(--user-limit "$USER_LIMIT")
    [[ -n "$MEMORY_CACHE_CONCURRENCY" ]] && build_cmd+=(--concurrency "$MEMORY_CACHE_CONCURRENCY")
    [[ -n "$MEMORY_CACHE_MAX_TOKENS" ]] && build_cmd+=(--max-tokens "$MEMORY_CACHE_MAX_TOKENS")
  fi

  local cmd=(
    "$PYTHON_BIN" "$REPO_ROOT/scripts/run_accuracy.py"
    --run-dir "$RUN_DIR"
    --system "$cmd_system"
    --judge-preset "$JUDGE_PRESET"
    --sglang-url "$endpoint"
    --model-name "$model_name"
    --model-tag "$model_tag"
    --trial-name "$trial"
    --seed "$seed"
    --namespace "$namespace"
    --out-dir "$OUT_DIR"
    --temperature "$TEMPERATURE"
  )
  if [[ "$needs_memory_cache" -eq 1 ]]; then
    cmd+=(
      --cache-path "$cache_path"
      --expected-extractor "$model_name"
      --expected-memory-system "$backend"
      --expected-config "$MEMORY_CACHE_CONFIG"
    )
  fi

  [[ -n "$ANSWER_CONCURRENCY" ]] && cmd+=(--answer-concurrency "$ANSWER_CONCURRENCY")
  [[ -n "$EVAL_CONCURRENCY" ]] && cmd+=(--eval-concurrency "$EVAL_CONCURRENCY")
  [[ -n "$QA_LIMIT" ]] && cmd+=(--qa-limit "$QA_LIMIT")
  [[ -n "$MESSAGE_LIMIT" ]] && cmd+=(--message-limit "$MESSAGE_LIMIT")
  [[ -n "$USER_LIMIT" ]] && cmd+=(--user-limit "$USER_LIMIT")
  [[ -n "$TOP_K" ]] && cmd+=(--top-k "$TOP_K")
  [[ -n "$D6_ARM" ]] && cmd+=(--d6-arm "$D6_ARM")
  [[ -n "$EVAL_CONFIG_PATH" ]] && cmd+=(--eval-config "$EVAL_CONFIG_PATH")
  [[ -n "$CONTEXT_LENGTH" ]] && cmd+=(--context-length "$CONTEXT_LENGTH")
  [[ -n "$MAX_TOKENS" ]] && cmd+=(--max-tokens "$MAX_TOKENS")
  [[ -n "$MAX_PREVIOUS_CONTEXT" ]] && cmd+=(--max-previous-context "$MAX_PREVIOUS_CONTEXT")
  [[ -n "$REMOTE_JUDGE_MODEL" ]] && cmd+=(--remote-judge-model "$REMOTE_JUDGE_MODEL")
  [[ -n "$REMOTE_JUDGE_ENDPOINT" ]] && cmd+=(--remote-judge-endpoint "$REMOTE_JUDGE_ENDPOINT")
  [[ -n "$LOCAL_JUDGE_MODEL" ]] && cmd+=(--local-judge-model "$LOCAL_JUDGE_MODEL")
  [[ -n "$LOCAL_JUDGE_URL" ]] && cmd+=(--local-judge-url "$LOCAL_JUDGE_URL")
  [[ -n "$JUDGE_MODEL" ]] && cmd+=(--judge-model "$JUDGE_MODEL")
  [[ -n "$JUDGE_ENDPOINT" ]] && cmd+=(--judge-endpoint "$JUDGE_ENDPOINT")
  [[ -n "$JUDGE_API_KEY" ]] && cmd+=(--judge-api-key "$JUDGE_API_KEY")
  [[ -n "$LOCAL_JUDGE_API_KEY" ]] && cmd+=(--local-judge-api-key "$LOCAL_JUDGE_API_KEY")
  [[ "$TEST_MODE" -eq 1 ]] && cmd+=(--test)
  [[ "$FORCE" -eq 1 ]] && cmd+=(--force)

  if [[ -n "$DIMENSIONS" ]]; then
    local dimension_arr=()
    local dim
    IFS=',' read -r -a dimension_arr <<<"$DIMENSIONS"
    cmd+=(--dimensions)
    for dim in "${dimension_arr[@]}"; do
      dim="$(trim "$dim")"
      [[ -n "$dim" ]] && cmd+=("$dim")
    done
  fi

  if [[ "$output_mode" == "print" ]]; then
    echo
    echo "[run-eval-matrix] [$cell/$TOTAL] backend=$backend model=$model_tag trial=$trial seed=$seed namespace=$namespace"
    if [[ "$needs_memory_cache" -eq 1 ]]; then
      echo "[run-eval-matrix] build memory cache:"
      if [[ "${#env_vars[@]}" -gt 0 ]]; then
        print_cmd env "${env_vars[@]}" "${build_cmd[@]}"
      else
        print_cmd "${build_cmd[@]}"
      fi
      echo "[run-eval-matrix] answer from memory cache:"
    fi
    if [[ "${#env_vars[@]}" -gt 0 ]]; then
      if [[ "$needs_memory_cache" -eq 1 ]]; then
        print_cmd "${cmd[@]}"
      else
        print_cmd env "${env_vars[@]}" "${cmd[@]}"
      fi
    else
      print_cmd "${cmd[@]}"
    fi
    rc=0
  elif [[ "$output_mode" == "live" ]]; then
    echo
    echo "[run-eval-matrix] [$cell/$TOTAL] backend=$backend model=$model_tag trial=$trial seed=$seed namespace=$namespace"
    if [[ "$needs_memory_cache" -eq 1 ]]; then
      mkdir -p "$cache_dir"
      echo "[run-eval-matrix] memory-cache=$cache_path"
      if [[ -f "$cache_path" && "$FORCE" -eq 0 ]]; then
        echo "[run-eval-matrix] reusing existing memory cache: $cache_path"
      else
        echo "[run-eval-matrix] build memory cache:"
        if [[ "${#env_vars[@]}" -gt 0 ]]; then
          print_cmd env "${env_vars[@]}" "${build_cmd[@]}"
          if (cd "$REPO_ROOT" && env "${env_vars[@]}" "${build_cmd[@]}"); then
            :
          else
            rc=$?
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
              "$cell" "$TOTAL" "$backend" "$model_tag" "$model_name" "$endpoint" \
              "$trial" "$seed" "$namespace" "$JUDGE_PRESET" "$TEST_MODE" "$rc" >>"$manifest_path"
            return "$rc"
          fi
        else
          print_cmd "${build_cmd[@]}"
          if (cd "$REPO_ROOT" && "${build_cmd[@]}"); then
            :
          else
            rc=$?
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
              "$cell" "$TOTAL" "$backend" "$model_tag" "$model_name" "$endpoint" \
              "$trial" "$seed" "$namespace" "$JUDGE_PRESET" "$TEST_MODE" "$rc" >>"$manifest_path"
            return "$rc"
          fi
        fi
      fi
      echo "[run-eval-matrix] answer from memory cache:"
    fi
    if [[ "${#env_vars[@]}" -gt 0 ]]; then
      if [[ "$needs_memory_cache" -eq 1 ]]; then
        print_cmd "${cmd[@]}"
      else
        print_cmd env "${env_vars[@]}" "${cmd[@]}"
      fi
    else
      print_cmd "${cmd[@]}"
    fi
    if [[ "${#env_vars[@]}" -gt 0 && "$needs_memory_cache" -eq 0 ]]; then
      if (cd "$REPO_ROOT" && env "${env_vars[@]}" "${cmd[@]}"); then
        rc=0
      else
        rc=$?
      fi
    elif (cd "$REPO_ROOT" && "${cmd[@]}"); then
      rc=0
    else
      rc=$?
    fi
  else
    mkdir -p "$(dirname "$log_path")"
    {
      echo "[run-eval-matrix] [$cell/$TOTAL] backend=$backend model=$model_tag trial=$trial seed=$seed namespace=$namespace"
      if [[ "$needs_memory_cache" -eq 1 ]]; then
        echo "[run-eval-matrix] memory-cache=$cache_path"
        echo "[run-eval-matrix] build memory cache:"
        if [[ "${#env_vars[@]}" -gt 0 ]]; then
          print_cmd env "${env_vars[@]}" "${build_cmd[@]}"
        else
          print_cmd "${build_cmd[@]}"
        fi
        echo
        echo "[run-eval-matrix] answer from memory cache:"
        print_cmd "${cmd[@]}"
      else
        echo "[run-eval-matrix] command:"
        if [[ "${#env_vars[@]}" -gt 0 ]]; then
          print_cmd env "${env_vars[@]}" "${cmd[@]}"
        else
          print_cmd "${cmd[@]}"
        fi
      fi
      echo
    } >"$log_path"
    if [[ "$needs_memory_cache" -eq 1 ]]; then
      mkdir -p "$cache_dir"
      if [[ -f "$cache_path" && "$FORCE" -eq 0 ]]; then
        echo "[run-eval-matrix] reusing existing memory cache: $cache_path" >>"$log_path"
      elif [[ "${#env_vars[@]}" -gt 0 ]]; then
        if (cd "$REPO_ROOT" && env "${env_vars[@]}" "${build_cmd[@]}") >>"$log_path" 2>&1; then
          :
        else
          rc=$?
          printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$cell" "$TOTAL" "$backend" "$model_tag" "$model_name" "$endpoint" \
            "$trial" "$seed" "$namespace" "$JUDGE_PRESET" "$TEST_MODE" "$rc" >>"$manifest_path"
          return "$rc"
        fi
      elif (cd "$REPO_ROOT" && "${build_cmd[@]}") >>"$log_path" 2>&1; then
        :
      else
        rc=$?
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
          "$cell" "$TOTAL" "$backend" "$model_tag" "$model_name" "$endpoint" \
          "$trial" "$seed" "$namespace" "$JUDGE_PRESET" "$TEST_MODE" "$rc" >>"$manifest_path"
        return "$rc"
      fi
    fi
    if [[ "${#env_vars[@]}" -gt 0 && "$needs_memory_cache" -eq 0 ]]; then
      if (cd "$REPO_ROOT" && env "${env_vars[@]}" "${cmd[@]}") >>"$log_path" 2>&1; then
        rc=0
      else
        rc=$?
      fi
    elif (cd "$REPO_ROOT" && "${cmd[@]}") >>"$log_path" 2>&1; then
      rc=0
    else
      rc=$?
    fi
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$cell" "$TOTAL" "$backend" "$model_tag" "$model_name" "$endpoint" \
    "$trial" "$seed" "$namespace" "$JUDGE_PRESET" "$TEST_MODE" "$rc" >>"$manifest_path"
  return "$rc"
}

run_sequential_matrix() {
  local cell=0
  local model_line
  local model_tag
  local model_name
  local endpoint
  local memobase_url
  local memos_url
  local backend
  local trial
  local trial_idx
  local seed
  local mode="live"
  local rc
  [[ "$PRINT_ONLY" -eq 1 ]] && mode="print"

  for model_line in "${MODEL_LINES[@]}"; do
    parse_model_line "$model_line"
    model_tag="$PARSED_MODEL_TAG"
    model_name="$PARSED_MODEL_NAME"
    endpoint="$PARSED_ENDPOINT"
    memobase_url="$PARSED_MEMOBASE_URL"
    memos_url="$PARSED_MEMOS_URL"

    for backend in "${BACKEND_ARR[@]}"; do
      backend="$(trim "$backend")"
      [[ -n "$backend" ]] || continue
      trial_idx=0
      for trial in "${TRIAL_ARR[@]}"; do
        trial="$(trim "$trial")"
        [[ -n "$trial" ]] || continue
        trial_idx=$((trial_idx + 1))
        cell=$((cell + 1))
        seed="$(trial_seed "$trial" "$trial_idx")"

        if run_cell "$cell" "$backend" "$model_tag" "$model_name" "$endpoint" "$trial" "$seed" "$memobase_url" "$memos_url" "$MATRIX_MANIFEST" "$mode"; then
          rc=0
        else
          rc=$?
        fi
        if [[ "$rc" -ne 0 ]]; then
          echo "[run-eval-matrix] cell failed with exit code $rc; manifest: $MATRIX_MANIFEST" >&2
          exit "$rc"
        fi
      done
    done
  done

  echo
  echo "[run-eval-matrix] completed $cell cell(s)"
  echo "[run-eval-matrix] manifest: $MATRIX_MANIFEST"
}

active_jobs_count() {
  jobs -pr | wc -l | tr -d '[:space:]'
}

progress_counts() {
  local done_count=0
  local failed_count=0
  local file
  local counts
  for file in "$STATE_DIR"/manifest_*.tsv; do
    [[ -f "$file" ]] || continue
    counts="$(awk -F '\t' 'NR > 1 {done += 1; if ($NF != "0") failed += 1} END {printf "%d %d", done + 0, failed + 0}' "$file")"
    done_count=$((done_count + ${counts%% *}))
    failed_count=$((failed_count + ${counts##* }))
  done
  printf '%s %s\n' "$done_count" "$failed_count"
}

render_parallel_progress() {
  local counts
  local done_count
  local failed_count
  local active_count
  local percent
  local model_line
  local model_tag
  local model_safe
  local status_file
  local status_model
  local status_done
  local status_total
  local status_backend
  local status_trial
  local status_state
  counts="$(progress_counts)"
  done_count="${counts%% *}"
  failed_count="${counts##* }"
  active_count="$(active_jobs_count)"
  percent="$(awk -v d="$done_count" -v t="$TOTAL" 'BEGIN {if (t <= 0) printf "100.0"; else printf "%.1f", d * 100 / t}')"

  echo
  printf '[run-eval-matrix] progress %s %s/%s %s%% active-models=%s failed-cells=%s\n' \
    "$(progress_bar "$done_count" "$TOTAL")" "$done_count" "$TOTAL" "$percent" "$active_count" "$failed_count"
  for model_line in "${MODEL_LINES[@]}"; do
    parse_model_line "$model_line"
    model_tag="$PARSED_MODEL_TAG"
    model_safe="$(safe_name "$model_tag")"
    status_file="$STATE_DIR/status_${model_safe}.txt"
    if [[ -s "$status_file" ]] && IFS=$'\t' read -r status_model status_done status_total status_backend status_trial status_state <"$status_file"; then
      printf '  %-14s [%2s/%2s] %-18s %-6s %s\n' \
        "$status_model" "$status_done" "$status_total" "$status_backend" "$status_trial" "$status_state"
    else
      printf '  %-14s [%2s/%2s] %-18s %-6s %s\n' \
        "$model_tag" "0" "$PER_MODEL_TOTAL" "-" "-" "pending"
    fi
  done
}

run_model_worker() {
  local model_idx="$1"
  local model_line="$2"
  local model_tag
  local model_name
  local endpoint
  local memobase_url
  local memos_url
  local model_safe
  local worker_manifest
  local status_file
  local log_dir
  local backend
  local backend_safe
  local trial
  local trial_safe
  local trial_idx
  local seed
  local local_cell=0
  local global_cell
  local log_path
  local rc

  parse_model_line "$model_line"
  model_tag="$PARSED_MODEL_TAG"
  model_name="$PARSED_MODEL_NAME"
  endpoint="$PARSED_ENDPOINT"
  memobase_url="$PARSED_MEMOBASE_URL"
  memos_url="$PARSED_MEMOS_URL"
  model_safe="$(safe_name "$model_tag")"
  worker_manifest="$STATE_DIR/manifest_${model_safe}.tsv"
  status_file="$STATE_DIR/status_${model_safe}.txt"
  log_dir="$LOG_ROOT/$model_safe"
  mkdir -p "$log_dir"
  printf '%s\n' "$MANIFEST_HEADER" >"$worker_manifest"
  write_status "$status_file" "$model_tag" "$local_cell" "$PER_MODEL_TOTAL" "-" "-" "starting"

  for backend in "${BACKEND_ARR[@]}"; do
    backend="$(trim "$backend")"
    [[ -n "$backend" ]] || continue
    backend_safe="$(safe_name "$backend")"
    trial_idx=0
    for trial in "${TRIAL_ARR[@]}"; do
      trial="$(trim "$trial")"
      [[ -n "$trial" ]] || continue
      trial_idx=$((trial_idx + 1))
      local_cell=$((local_cell + 1))
      global_cell=$((model_idx * PER_MODEL_TOTAL + local_cell))
      seed="$(trial_seed "$trial" "$trial_idx")"
      trial_safe="$(safe_name "$trial")"
      log_path="$log_dir/${backend_safe}_${trial_safe}.log"
      write_status "$status_file" "$model_tag" "$((local_cell - 1))" "$PER_MODEL_TOTAL" "$backend" "$trial" "running"
      if run_cell "$global_cell" "$backend" "$model_tag" "$model_name" "$endpoint" "$trial" "$seed" "$memobase_url" "$memos_url" "$worker_manifest" "logged" "$log_path"; then
        rc=0
      else
        rc=$?
      fi
      if [[ "$rc" -ne 0 ]]; then
        write_status "$status_file" "$model_tag" "$local_cell" "$PER_MODEL_TOTAL" "$backend" "$trial" "failed rc=$rc log=$log_path"
        return "$rc"
      fi
      write_status "$status_file" "$model_tag" "$local_cell" "$PER_MODEL_TOTAL" "$backend" "$trial" "done"
    done
  done
  write_status "$status_file" "$model_tag" "$local_cell" "$PER_MODEL_TOTAL" "-" "-" "complete"
}

merge_worker_manifests() {
  local model_line
  local model_tag
  local model_safe
  local worker_manifest
  printf '%s\n' "$MANIFEST_HEADER" >"$MATRIX_MANIFEST"
  for model_line in "${MODEL_LINES[@]}"; do
    parse_model_line "$model_line"
    model_tag="$PARSED_MODEL_TAG"
    model_safe="$(safe_name "$model_tag")"
    worker_manifest="$STATE_DIR/manifest_${model_safe}.tsv"
    [[ -f "$worker_manifest" ]] || continue
    awk 'NR > 1 {print}' "$worker_manifest" >>"$MATRIX_MANIFEST"
  done
}

manifest_failed_count() {
  awk -F '\t' 'NR > 1 && $NF != "0" {failed += 1} END {print failed + 0}' "$MATRIX_MANIFEST"
}

run_parallel_matrix() {
  local state_stamp
  local model_idx=0
  local model_line
  local pid
  local pids=()
  local wait_rc=0
  local failed_count
  local rc
  state_stamp="$(date +%Y%m%d_%H%M%S)"
  STATE_DIR="$OUT_DIR/.matrix_state_${state_stamp}"
  LOG_ROOT="$OUT_DIR/matrix_logs"
  mkdir -p "$STATE_DIR" "$LOG_ROOT"
  echo "[run-eval-matrix] parallel state: $STATE_DIR"
  echo "[run-eval-matrix] cell logs: $LOG_ROOT"

  for model_line in "${MODEL_LINES[@]}"; do
    while [[ "$(active_jobs_count)" -ge "$MODEL_PARALLELISM" ]]; do
      render_parallel_progress
      sleep 2
    done
    run_model_worker "$model_idx" "$model_line" &
    pid="$!"
    pids+=("$pid")
    model_idx=$((model_idx + 1))
    render_parallel_progress
  done

  while [[ "$(active_jobs_count)" -gt 0 ]]; do
    render_parallel_progress
    sleep 2
  done

  set +e
  for pid in "${pids[@]}"; do
    wait "$pid"
    rc=$?
    [[ "$rc" -eq 0 ]] || wait_rc="$rc"
  done
  set -e

  merge_worker_manifests
  render_parallel_progress
  failed_count="$(manifest_failed_count)"
  if [[ "$failed_count" -ne 0 ]]; then
    echo "[run-eval-matrix] $failed_count cell(s) failed; manifest: $MATRIX_MANIFEST" >&2
    echo "[run-eval-matrix] inspect logs under: $LOG_ROOT" >&2
    exit 1
  fi
  if [[ "$wait_rc" -ne 0 ]]; then
    echo "[run-eval-matrix] worker failed with exit code $wait_rc; inspect logs under: $LOG_ROOT" >&2
    exit "$wait_rc"
  fi
  echo
  echo "[run-eval-matrix] completed $TOTAL cell(s)"
  echo "[run-eval-matrix] manifest: $MATRIX_MANIFEST"
}

if [[ "$PRINT_ONLY" -eq 1 || "$MODEL_PARALLELISM" -eq 1 ]]; then
  run_sequential_matrix
else
  run_parallel_matrix
fi
