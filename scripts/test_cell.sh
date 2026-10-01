#!/usr/bin/env bash
# Generate the answers of one main-grid cell (reader x backend x seed) on one GPU.
# Answers only: judging is scripts/judge_cell.sh.

set -uo pipefail

# shellcheck source=scripts/grid_common.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/grid_common.sh"

usage() {
  cat <<'EOF'
Usage:
  scripts/test_cell.sh -model M -backend B -gpu G -seed S [-limit N] [-dataset DIR] [-out DIR]
                       [-force] [-redo-dims DIMS] [-dry-run] [-endpoint URL]
                       [-tag T [-top-k K] [-memory-from CELL] [-d6-access] [-dims DIMS]]

  -model     0_6b | llama3b | 7b | 8b | 32b
             (Qwen/Qwen3-0.6B, meta-llama/Llama-3.2-3B-Instruct,
              mistralai/Mistral-7B-Instruct-v0.3, Qwen/Qwen3-8B, Qwen/Qwen3-32B-AWQ)
  -backend   vanilla | rag | memobase | memsearch | oracle   (the main grid), or an ablation
             backend: dense_e5 | dense_bge_m3 | hybrid_rerank | temporal | text_sessions | omniscient
  -gpu       one GPU index; nothing else is used (CUDA_VISIBLE_DEVICES, docker --gpus device=G)
  -seed      s1 (T=0.0, trial seed 1001) | s2 s3 s4 (T=0.3, trial seeds 1002/1003/1004)
  -limit N   smoke test: first N items of every task file, written to <cell>/limitN/,
             which never counts as the finished cell
  -dataset   default: $MEMARENA_DATASET_DIR, else data/benchmark
             (python scripts/download_dataset.py fetches dereksodo/memarena-l)
  -out       output root, default out/runs
  -force     re-run a finished cell (and rebuild its Memobase memory)
  -tag T     a variant of the cell, kept apart: <out>/<backend>-T/<model>/<seed>, namespace
             <backend>_T_<model>_<seed>. Required with any of these variant options:
    -top-k K        retrieval depth of the retrieving backends (default: the pipeline's)
    -memory-from C  memobase: answer from the finished Memobase memory of cell directory C
                    (e.g. out/runs/memobase/32b/s2 for a Qwen3-32B-AWQ writer) instead of
                    building one with this reader
    -d6-access      inject [access:DENY]/[access:ALLOW] into the D6 prompts
    -d6-no-contacts / -d6-no-asker  drop the owner's contact list / the asker's name from
                    the D6 prompts (requester-identity ablation)
    -dims DIMS      answer only these paper dimensions (e.g. D6)
  -redo-dims re-answer only these paper dimensions (e.g. D6) of a finished cell,
             reusing its memory: the answers go to <cell>/redo_<dims>/ and then
             replace those rows of the cell's answer file (previous file kept as
             answer_results_<ns>.before_redo_<time>.json); the cell stays "done"
             throughout (progress.json redo_* fields). Judge again with
             judge_cell.sh -redo DIMS.
  -dry-run   print every command in order and run nothing
  -endpoint  use an OpenAI-compatible reader already serving the model at URL
             (e.g. http://127.0.0.1:30000/v1) instead of starting SGLang

Stages (each shown with its elapsed time, and tracked in progress.json):
  1/4 sglang   SGLang for the reader on GPU G (flags of start_sglang_servers.sh,
               TP=1 DP=1; Qwen3-32B-AWQ uses SGLang's native AWQ kernels)
  2/4 memory   memobase:  an isolated Memobase v0.0.42 stack for this reader x seed
                          (scripts/setup_memory_backends.sh), extractor = the reader,
                          nomic-embed-text on an Ollama pinned to GPU G; built with
                          scripts/reproduce/build_memory_cache.py; the stack is torn
                          down afterwards, its database kept in the cell directory
               memsearch: one reader-independent cache shared by every cell, built
                          once (lock file) under <out>/_shared/memsearch/
               vanilla, rag, oracle: nothing to prepare
  3/4 answer   python -m eval.cli, the paper's answer settings, D6 arm A,
               --fail-on-thinking
  4/4 teardown every container this script started is removed (also on error / Ctrl-C)

Output: <out>/<backend>/<model>/<seed>/[limitN/]
  progress.json  args.json  <system>/answer_results_<ns>.json  memory/  logs/*.log

Environment overrides (defaults in brackets):
  SGLANG_IMAGE [lmsysorg/sglang:latest]   MODEL_ROOT [~/models]   HF_CACHE_DIR [~/.cache/huggingface]
  SGLANG_CONTEXT_LENGTH [16384]  SGLANG_MEM_FRACTION_STATIC [0.85]  SGLANG_MAX_RUNNING_REQUESTS [128]
  SGLANG_PIP_INSTALL [protobuf sentencepiece]  SGLANG_EXTRA_ARGS []  SGLANG_READY_TIMEOUT [1800]
  OLLAMA_IMAGE [ollama/ollama:0.23.1]  OLLAMA_VOLUME [memarena-ollama-models]
  MEMARENA_PORT_BASE [20000]      ports used: BASE + 100*GPU + {0 sglang, 10 memobase api,
                                  20 postgres, 30 redis, 40 ollama}, first free in each block
  MEMARENA_MEMSEARCH_DIR [<out>/_shared/memsearch]   MEMARENA_SERVICES_CACHE [<out>/_shared/services]
  MEMARENA_ALLOWED_GPUS []        e.g. "3,4,6,7": refuse any other GPU
  MEMARENA_LOCK_TIMEOUT [21600]   seconds to wait for another cell's MemSearch build
  PYTHON                          interpreter (default .venv/bin/python, else python3)
EOF
}

# ---------------------------------------------------------------- arguments
MODEL="" BACKEND="" GPU="" SEED="" LIMIT=0 DATASET="" OUT="" FORCE=0 DRY=0 ENDPOINT="" REDO=""
TAG="" TOPK="" MEM_FROM="" D6_ACCESS=0 DIMS="" D6_NO_CONTACTS=0 D6_NO_ASKER=0
need_val() { [[ $# -ge 2 && -n "$2" ]] || { echo "test_cell: $1 needs a value" >&2; exit 2; }; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    -model|--model) need_val "$@"; MODEL="$2"; shift 2 ;;
    -backend|--backend) need_val "$@"; BACKEND="$2"; shift 2 ;;
    -gpu|--gpu) need_val "$@"; GPU="$2"; shift 2 ;;
    -seed|--seed) need_val "$@"; SEED="$2"; shift 2 ;;
    -limit|--limit) need_val "$@"; LIMIT="$2"; shift 2 ;;
    -dataset|--dataset) need_val "$@"; DATASET="$2"; shift 2 ;;
    -out|--out) need_val "$@"; OUT="$2"; shift 2 ;;
    -endpoint|--endpoint) need_val "$@"; ENDPOINT="$2"; shift 2 ;;
    -force|--force) FORCE=1; shift ;;
    -redo-dims|--redo-dims) need_val "$@"; REDO="$2"; shift 2 ;;
    -tag|--tag) need_val "$@"; TAG="$2"; shift 2 ;;
    -top-k|--top-k) need_val "$@"; TOPK="$2"; shift 2 ;;
    -memory-from|--memory-from) need_val "$@"; MEM_FROM="$2"; shift 2 ;;
    -d6-access|--d6-access) D6_ACCESS=1; shift ;;
    -d6-no-contacts|--d6-no-contacts) D6_NO_CONTACTS=1; shift ;;
    -d6-no-asker|--d6-no-asker) D6_NO_ASKER=1; shift ;;
    -dims|--dims) need_val "$@"; DIMS="$2"; shift 2 ;;
    -dry-run|--dry-run) DRY=1; shift ;;
    -h|-help|--help) usage; exit 0 ;;
    *) echo "test_cell: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

bad() { echo "test_cell: $*" >&2; exit 2; }
[[ -n "$MODEL" && -n "$BACKEND" && -n "$GPU" && -n "$SEED" ]] || { usage >&2; bad "-model, -backend, -gpu and -seed are required"; }
in_list "$MODEL" "$GRID_MODELS" || bad "unknown model '$MODEL' (valid: $GRID_MODELS)"
in_list "$BACKEND" "$GRID_ALL_BACKENDS" || bad "unknown backend '$BACKEND' (valid: $GRID_ALL_BACKENDS)"
in_list "$SEED" "$GRID_SEEDS" || bad "unknown seed '$SEED' (valid: $GRID_SEEDS)"
[[ "$GPU" =~ ^[0-9]+$ ]] || bad "-gpu takes a single GPU index, got '$GPU'"
[[ "$LIMIT" =~ ^[0-9]+$ ]] || bad "-limit takes a positive integer, got '$LIMIT'"
[[ -z "$REDO" || "$REDO" =~ ^([Dd][1-6][,[:space:]]*)+$ ]] || bad "-redo-dims takes paper dimensions D1..D6, e.g. D6"
[[ -z "$REDO" || "$FORCE" == 0 ]] || bad "-redo-dims and -force exclude each other"
[[ -z "$TAG" || "$TAG" =~ ^[A-Za-z0-9_]+$ ]] || bad "-tag takes letters, digits and _ only"
if [[ -n "$TOPK$MEM_FROM$DIMS" || "$D6_ACCESS$D6_NO_CONTACTS$D6_NO_ASKER" != 000 ]]; then
  [[ -n "$TAG" ]] || bad "-top-k, -memory-from, -d6-access and -dims need -tag (the variant must not overwrite the main cell)"
fi
[[ -z "$TOPK" || "$TOPK" =~ ^[1-9][0-9]*$ ]] || bad "-top-k takes a positive integer"
[[ -z "$MEM_FROM" || "$BACKEND" == memobase ]] || bad "-memory-from is for -backend memobase"
[[ -z "$DIMS" || "$DIMS" =~ ^([Dd][1-6][,[:space:]]*)+$ ]] || bad "-dims takes paper dimensions D1..D6, e.g. D6"
[[ -z "$DIMS" || -z "$REDO" ]] || bad "-dims and -redo-dims exclude each other"
if [[ -n "${MEMARENA_ALLOWED_GPUS:-}" ]] && ! in_list "$GPU" "${MEMARENA_ALLOWED_GPUS//,/ }"; then
  bad "GPU $GPU is not in MEMARENA_ALLOWED_GPUS=$MEMARENA_ALLOWED_GPUS"
fi

cd "$GRID_REPO_ROOT" || exit 2
PY="$(grid_python)"
HF_MODEL="$(model_hf "$MODEL")"
TEMP="$(seed_temperature "$SEED")"
TRIAL_SEED="$(seed_trial "$SEED")"
SYSTEM="$(backend_system "$BACKEND")"
NS="$(cell_namespace "$BACKEND" "$MODEL" "$SEED" "$TAG")"

OUT="$(abspath "${OUT:-out/runs}")"
DATASET_DEFAULT=0
if [[ -z "$DATASET" ]]; then
  if [[ -n "${MEMARENA_DATASET_DIR:-}" ]]; then DATASET="$MEMARENA_DATASET_DIR"; else DATASET="data/benchmark"; DATASET_DEFAULT=1; fi
fi
DATASET="$(abspath "$DATASET")"
CELL_DIR="$(cell_dir "$OUT" "$BACKEND" "$MODEL" "$SEED" "$LIMIT" "$TAG")"
LOG_DIR="$CELL_DIR/logs"
PROGRESS="$CELL_DIR/progress.json"
ANSWER_FILE="$CELL_DIR/$SYSTEM/answer_results_${NS}.json"
CELL_LABEL="$(backend_dirname "$BACKEND" "$TAG")/$MODEL/$SEED"
[[ "$LIMIT" != 0 ]] && CELL_LABEL="$CELL_LABEL/limit$LIMIT"
CELL_LABEL="$CELL_LABEL gpu$GPU"

# one name prefix per cell directory: containers, compose project, Ollama
cell_hash="$(printf '%s' "$CELL_DIR" | cksum | awk '{printf "%06x", $1 % 16777216}')"
PREFIX="ma-$(backend_dirname "$BACKEND" "$TAG" | tr '_' '-')-${MODEL//_/-}-${SEED}"
[[ "$LIMIT" != 0 ]] && PREFIX="${PREFIX}-l${LIMIT}"
PREFIX="${PREFIX}-${cell_hash}"

SGLANG_IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang:latest}"
MODEL_ROOT="${MODEL_ROOT:-$HOME/models}"; MODEL_ROOT="${MODEL_ROOT/#\~/$HOME}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"; HF_CACHE_DIR="${HF_CACHE_DIR/#\~/$HOME}"
SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-16384}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.85}"
SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-128}"
if [[ -z "${SGLANG_PIP_INSTALL+x}" ]]; then SGLANG_PIP_INSTALL="protobuf sentencepiece"; fi
SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS:-}"
SGLANG_READY_TIMEOUT="${SGLANG_READY_TIMEOUT:-1800}"
OLLAMA_IMAGE="${OLLAMA_IMAGE:-ollama/ollama:0.23.1}"
OLLAMA_VOLUME="${OLLAMA_VOLUME:-memarena-ollama-models}"
EMBED_MODEL="nomic-embed-text:latest"
PORT_BASE=$(( ${MEMARENA_PORT_BASE:-20000} + 100 * GPU ))
SHARED_DIR="$OUT/_shared"
MEMSEARCH_ROOT="$(abspath "${MEMARENA_MEMSEARCH_DIR:-$SHARED_DIR/memsearch}")"
SERVICES_CACHE="$(abspath "${MEMARENA_SERVICES_CACHE:-$SHARED_DIR/services}")"
LOCK_TIMEOUT="${MEMARENA_LOCK_TIMEOUT:-21600}"
GPU_LOCK="/tmp/memarena-locks-$(id -u)/gpu$GPU"
CELL_LOCK="$CELL_DIR/.lock"

export CUDA_VISIBLE_DEVICES="$GPU"
# the reader needs no key; keep judge keys out of everything this cell starts
unset OPENROUTER_API_KEY OPENAI_API_KEY QWEN235B_API_KEY

# ---------------------------------------------------------------- output helpers
T0=$SECONDS
say() {
  local line
  line="[cell $CELL_LABEL $(date +%H:%M:%S)] $*"
  echo "$line"
  if [[ "$DRY" == 0 && -d "$LOG_DIR" ]]; then echo "$line" >>"$LOG_DIR/test_cell.log"; fi
}
progress() { [[ "$DRY" == 1 || "$RUN_STARTED" == 0 ]] || helper progress-set "$PROGRESS" "$@"; }

STAGE_T0=$SECONDS STAGE_N=0 STAGE_NAME=""
stage_begin() {
  STAGE_N="$1" STAGE_NAME="$2" STAGE_T0=$SECONDS
  say "== stage $1/4 $2 =="
  [[ "$DRY" == 1 ]] || helper stage "$PROGRESS" "$1" "$2"
}
stage_end() { say "stage $STAGE_N/4 $STAGE_NAME done in $(fmt_secs $((SECONDS - STAGE_T0)))"; }

# run: print in dry-run, else execute with output appended to a log file
run() {  # run LOG cmd...
  local log="$1"; shift
  if [[ "$DRY" == 1 ]]; then
    echo "+ $(quote_cmd "$@") >>$(printf '%q' "$log") 2>&1"
    return 0
  fi
  "$@" >>"$log" 2>&1
}

# ---------------------------------------------------------------- cleanup
CONTAINERS=() FOLLOWERS=() CHILDREN=() LOCKS=()
MB_STACK_UP=0 MB_ENV=() CLEANED=0 INTERRUPTED=0 FINAL_STATUS="" RUN_STARTED=0

memobase_logs_and_down() {
  [[ "$MB_STACK_UP" == 1 ]] || return 0
  local proj="${PREFIX}-memobase" c
  for c in $(docker ps -a --filter "label=com.docker.compose.project=$proj" --format '{{.Names}}' 2>/dev/null); do
    { echo "===== $c"; docker logs --tail 2000 "$c" 2>&1; } >>"$LOG_DIR/memobase_stack.log"
  done
  env "${MB_ENV[@]}" bash "$GRID_SCRIPT_DIR/setup_memory_backends.sh" stop >>"$LOG_DIR/memobase_stack.log" 2>&1 \
    || say "WARNING: memobase compose down failed; check: docker compose -p $proj ps"
  MB_STACK_UP=0
}

cleanup() {
  [[ "$CLEANED" == 1 ]] && return 0
  CLEANED=1
  [[ "$DRY" == 1 ]] && return 0
  local pid c i
  for pid in ${CHILDREN[@]+"${CHILDREN[@]}"}; do kill -TERM "$pid" 2>/dev/null; done
  for i in 1 2 3 4 5 6 7 8 9 10; do
    local alive=0
    for pid in ${CHILDREN[@]+"${CHILDREN[@]}"}; do kill -0 "$pid" 2>/dev/null && alive=1; done
    [[ "$alive" == 0 ]] && break
    sleep 1
  done
  for pid in ${CHILDREN[@]+"${CHILDREN[@]}"}; do kill -KILL "$pid" 2>/dev/null; done
  memobase_logs_and_down
  for c in ${CONTAINERS[@]+"${CONTAINERS[@]}"}; do
    if docker inspect "$c" >/dev/null 2>&1; then
      docker rm -f "$c" >/dev/null 2>&1 && say "removed container $c"
    fi
  done
  for pid in ${FOLLOWERS[@]+"${FOLLOWERS[@]}"}; do kill "$pid" 2>/dev/null; done
  for c in ${LOCKS[@]+"${LOCKS[@]}"}; do lock_release "$c"; done
}

fail() {
  say "FAILED: $*"
  FINAL_STATUS=failed
  if [[ -n "$REDO" ]]; then  # the cell's own answers are untouched: it stays "done"
    progress redo_status=failed "redo_message=$*" "redo_failed_stage=$STAGE_N/4 $STAGE_NAME" "redo_ended_at=$(now_iso)"
  else
    progress status=failed "message=$*" "failed_stage=$STAGE_N/4 $STAGE_NAME" "ended_at=$(now_iso)"
  fi
  exit 1
}

on_exit() {
  local rc=$?
  cleanup
  if [[ "$DRY" == 0 && "$RUN_STARTED" == 1 && -z "$FINAL_STATUS" && -f "$PROGRESS" ]]; then
    local why="exit code $rc"
    [[ "$INTERRUPTED" == 1 ]] && why="interrupted"
    if [[ -n "$REDO" ]]; then
      progress redo_status=failed "redo_message=$why" "redo_failed_stage=$STAGE_N/4 $STAGE_NAME" "redo_ended_at=$(now_iso)"
    else
      progress status=failed "message=$why" "failed_stage=$STAGE_N/4 $STAGE_NAME" "ended_at=$(now_iso)"
    fi
    FINAL_STATUS=failed
  fi
  if [[ -n "${MEMARENA_CELL_RESULT_FILE:-}" ]]; then
    local r="${FINAL_STATUS:-failed}"; [[ "$DRY" == 1 && -z "$FINAL_STATUS" ]] && r=dry-run
    echo "$r" >"$MEMARENA_CELL_RESULT_FILE"
  fi
}
trap on_exit EXIT
trap 'INTERRUPTED=1; say "interrupted, stopping services"; exit 130' INT TERM HUP

# wait for a background child while showing its progress
watch_child() {  # watch_child PID LOG KIND(answer|build)
  local pid="$1" log="$2" kind="$3" parsed last_print=-1000 last_line="" line
  while kill -0 "$pid" 2>/dev/null; do
    parsed="$(helper parse-log "$log" "$kind")"
    if [[ -n "$parsed" ]]; then
      set -- $parsed
      local unit="$3"
      case "$unit" in eval_search) unit=searched ;; eval_answer) unit=answered ;; days) unit="memory build day" ;; esac
      line="$(helper items "$PROGRESS" "$1" "$2" "$unit")"
      if [[ "$line" != "$last_line" && $((SECONDS - last_print)) -ge 30 ]] || [[ $((SECONDS - last_print)) -ge 300 ]]; then
        say "$line"
        last_print=$SECONDS last_line="$line"
      fi
    fi
    sleep 5 & wait $!
  done
  wait "$pid"
}

start_bg() {  # start_bg LOG cmd...  -> BG_PID
  local log="$1"; shift
  "$@" >>"$log" 2>&1 &
  BG_PID=$!
  CHILDREN+=("$BG_PID")
}

pick_port() {  # pick_port OFFSET   (caller: X="$(pick_port N)" || fail ...)
  helper free-port $((PORT_BASE + $1)) 10
}

wait_http() {  # wait_http URL TIMEOUT LABEL [CONTAINER]
  local url="$1" timeout="$2" label="$3" ctr="${4:-}" t0=$SECONDS next=30
  [[ "$DRY" == 1 ]] && { echo "+ wait until $url answers (timeout ${timeout}s)"; return 0; }
  until curl -fsS -m 5 "$url" >/dev/null 2>&1; do
    if [[ "$ctr" == compose:* ]]; then
      if [[ -n "$(docker ps -a -q --filter "label=com.docker.compose.project=${ctr#compose:}" --filter status=exited 2>/dev/null)" ]]; then
        fail "a $label container exited before becoming ready (see logs/memobase_stack.log)"
      fi
    elif [[ -n "$ctr" && "$(docker inspect -f '{{.State.Running}}' "$ctr" 2>/dev/null)" != "true" ]]; then
      fail "$label exited before becoming ready (see logs)"
    fi
    (( SECONDS - t0 > timeout )) && fail "$label not ready after ${timeout}s"
    if (( SECONDS - t0 >= next )); then say "waiting for $label ($(fmt_secs $((SECONDS - t0))))"; next=$((next + 60)); fi
    sleep 5 & wait $!
  done
  say "$label ready after $(fmt_secs $((SECONDS - t0)))"
}

follow_logs() {  # follow_logs CONTAINER LOGFILE
  [[ "$DRY" == 1 ]] && return 0
  docker logs -f "$1" >>"$2" 2>&1 &
  FOLLOWERS+=("$!")
}

# ---------------------------------------------------------------- skip / locks
if [[ -n "$REDO" ]] && ! [[ -f "$PROGRESS" && -f "$ANSWER_FILE" && "$(helper progress-get "$PROGRESS" status)" == "done" ]]; then
  bad "$CELL_LABEL has no finished answers to redo dimensions of ($ANSWER_FILE); run it first without -redo-dims"
fi
if [[ -z "$REDO" && "$FORCE" == 0 && -f "$PROGRESS" && -f "$ANSWER_FILE" && "$(helper progress-get "$PROGRESS" status)" == "done" ]]; then
  echo "[skip] $CELL_LABEL is done ($ANSWER_FILE); use -force to re-run"
  FINAL_STATUS=skipped
  exit 0
fi

if [[ ! -d "$DATASET/eval_instances" ]]; then
  if [[ "$DATASET_DEFAULT" == 1 ]]; then
    if [[ "$DRY" == 1 ]]; then
      echo "+ $(quote_cmd "$PY" scripts/download_dataset.py)   # $DATASET is missing"
    else
      echo "[cell] dataset missing at $DATASET; downloading dereksodo/memarena-l"
      "$PY" scripts/download_dataset.py || bad "dataset download failed"
    fi
  else
    bad "no dataset at $DATASET (expected eval_instances/); run: python scripts/download_dataset.py"
  fi
fi

if [[ "$DRY" == 0 ]]; then
  command -v docker >/dev/null 2>&1 || [[ -n "$ENDPOINT" && "$BACKEND" != memobase && "$BACKEND" != memsearch ]] \
    || bad "docker is not on PATH"
  mkdir -p "$LOG_DIR"
  lock_try "$CELL_LOCK" || bad "cell $CELL_LABEL is already running (pid $(lock_owner "$CELL_LOCK"))"
  LOCKS+=("$CELL_LOCK")
  if [[ -z "$ENDPOINT" || "$BACKEND" == memobase || "$BACKEND" == memsearch ]]; then
    lock_try "$GPU_LOCK" || bad "GPU $GPU is busy with another cell (pid $(lock_owner "$GPU_LOCK"))"
    LOCKS+=("$GPU_LOCK")
  fi
fi

# dataset actually answered: the full one, or a per-task-file prefix for -limit
DATA="$DATASET"
if [[ "$LIMIT" != 0 ]]; then
  DATA="$SHARED_DIR/datasets/limit$LIMIT"
  if [[ ! -d "$DATA" ]]; then
    if [[ "$DRY" == 1 ]]; then
      echo "+ $(quote_cmd python3 "$GRID_HELPER" make-limit-dataset "$DATASET" "$DATA" "$LIMIT")"
    else
      lk="$SHARED_DIR/datasets/.limit$LIMIT.lock"
      until lock_try "$lk"; do sleep 1; done
      [[ -d "$DATA" ]] || helper make-limit-dataset "$DATASET" "$DATA" "$LIMIT" || { lock_release "$lk"; bad "could not build $DATA"; }
      lock_release "$lk"
    fi
  fi
fi
REDO_TASKS=() REDO_DIR="" REDO_ANSWER=""
if [[ -n "$REDO" ]]; then
  read -r ITEMS_TOTAL REDO_TASKS_STR <<<"$(helper redo-dims "$DATA" "$REDO")" || bad "cannot resolve -redo-dims $REDO"
  read -r -a REDO_TASKS <<<"$REDO_TASKS_STR"
  REDO_DIR="$CELL_DIR/redo_$(printf '%s' "$REDO" | tr -c 'A-Za-z0-9' '_' | sed 's/_*$//')"
  REDO_ANSWER="$REDO_DIR/$SYSTEM/answer_results_${NS}.json"
elif [[ -n "$DIMS" ]]; then
  read -r ITEMS_TOTAL REDO_TASKS_STR <<<"$(helper redo-dims "$DATA" "$DIMS")" || bad "cannot resolve -dims $DIMS"
  read -r -a REDO_TASKS <<<"$REDO_TASKS_STR"
elif [[ -d "$DATA/eval_instances" ]]; then
  ITEMS_TOTAL="$(helper count-items "$DATA")"
else
  ITEMS_TOTAL="$(helper count-items "$DATASET" "$LIMIT" 2>/dev/null || echo '?')"
fi

if [[ "$DRY" == 1 ]]; then
  echo "# dry run: $CELL_LABEL  (reader $HF_MODEL, T=$TEMP, trial seed $TRIAL_SEED, $ITEMS_TOTAL items)"
  echo "# cell dir: $CELL_DIR   container prefix: $PREFIX"
elif [[ -n "$REDO" ]]; then
  RUN_STARTED=1   # keep the finished cell's progress.json; the redo has its own fields
  progress redo_status=running "redo_dims=$REDO" "redo_started_at=$(now_iso)" "redo_items=$ITEMS_TOTAL" \
    redo_message=null "redo_pid=$$"
  say "re-answering $REDO (${REDO_TASKS[*]}: $ITEMS_TOTAL items) of the finished cell, reader $HF_MODEL, T=$TEMP"
else
  # a fresh progress.json for this attempt
  rm -f "$PROGRESS"
  RUN_STARTED=1
  progress status=running "backend=$BACKEND" "model=$MODEL" "model_name=$HF_MODEL" "seed=$SEED" \
    "temperature=$TEMP" "trial_seed=$TRIAL_SEED" "gpu=$GPU" "limit=$LIMIT" "namespace=$NS" \
    "cell_dir=$CELL_DIR" "started_at=$(now_iso)" stage_total=4 "answers_total=$ITEMS_TOTAL" "pid=$$"
  say "reader $HF_MODEL, T=$TEMP, trial seed $TRIAL_SEED, $ITEMS_TOTAL items, out $CELL_DIR"
fi

# ---------------------------------------------------------------- stage 1: sglang
stage_begin 1 sglang
if [[ -n "$ENDPOINT" ]]; then
  READER_URL="${ENDPOINT%/}"
  [[ "$READER_URL" == */v1 ]] || READER_URL="$READER_URL/v1"
  SPORT="$(printf '%s' "$READER_URL" | sed -E 's#^[a-z]+://[^:/]+:([0-9]+).*#\1#')"
  say "using the reader already at $READER_URL (no SGLang started)"
  wait_http "$READER_URL/models" 60 "reader endpoint"
else
  SPORT="$(pick_port 0)" || fail "no free port in $PORT_BASE..$((PORT_BASE + 9))"
  READER_URL="http://127.0.0.1:$SPORT/v1"
  SGLANG_CTR="${PREFIX}-sglang"
  if [[ -d "$MODEL_ROOT/$MODEL" ]]; then
    model_mount=(-v "$MODEL_ROOT:/models:ro"); model_path="/models/$MODEL"
  else
    model_mount=(); model_path="$HF_MODEL"   # downloaded into the HF cache on first use
  fi
  env_args=()
  [[ -n "${HF_TOKEN:-}" ]] && env_args+=(--env HF_TOKEN)                        # value taken from the environment,
  [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]] && env_args+=(--env HUGGING_FACE_HUB_TOKEN)  # never written on the command line
  # shellcheck disable=SC2206
  extra_args=($SGLANG_EXTRA_ARGS)
  sglang_cmd=(docker run -d --name "$SGLANG_CTR" --gpus "\"device=$GPU\"" --ipc=host --shm-size 32g --restart no
    -p "$SPORT:$SPORT" ${model_mount[@]+"${model_mount[@]}"} -v "$HF_CACHE_DIR:/root/.cache/huggingface"
    --env "SGLANG_PIP_INSTALL=$SGLANG_PIP_INSTALL" ${env_args[@]+"${env_args[@]}"}
    "$SGLANG_IMAGE" bash -lc 'set -euo pipefail
if [[ -n "${SGLANG_PIP_INSTALL:-}" ]]; then python3 -m pip install --no-cache-dir ${SGLANG_PIP_INSTALL}; fi
exec "$@"' bash
    python3 -m sglang.launch_server --model-path "$model_path" --served-model-name "$HF_MODEL"
    --host 0.0.0.0 --port "$SPORT" --tp 1 --dp 1 --context-length "$SGLANG_CONTEXT_LENGTH"
    --mem-fraction-static "$SGLANG_MEM_FRACTION_STATIC" --max-running-requests "$SGLANG_MAX_RUNNING_REQUESTS"
    ${extra_args[@]+"${extra_args[@]}"})
  if [[ "$DRY" == 0 ]]; then
    docker rm -f "$SGLANG_CTR" >/dev/null 2>&1   # leftover of a killed run of this same cell
    CONTAINERS+=("$SGLANG_CTR")
    progress "sglang_container=$SGLANG_CTR" "sglang_port=$SPORT"
  fi
  run "$LOG_DIR/sglang.log" "${sglang_cmd[@]}" || fail "docker run for SGLang failed (logs/sglang.log)"
  follow_logs "$SGLANG_CTR" "$LOG_DIR/sglang.log"
  say "SGLang $HF_MODEL on GPU $GPU, port $SPORT (container $SGLANG_CTR)"
  wait_http "http://127.0.0.1:$SPORT/v1/models" "$SGLANG_READY_TIMEOUT" "SGLang" "$SGLANG_CTR"
fi
stage_end

# ---------------------------------------------------------------- stage 2: memory
# host address of the reader as seen from inside a container (Memobase extractor)
READER_URL_DOCKER="$(printf '%s' "$READER_URL" | sed -E 's#://(127\.0\.0\.1|localhost)([:/])#://host.docker.internal\2#')"

start_ollama() {  # start_ollama NUM_PARALLEL -> OPORT, OLLAMA_CTR
  OPORT="$(pick_port 40)" || fail "no free port for Ollama near $((PORT_BASE + 40))"
  OLLAMA_CTR="${PREFIX}-ollama"
  OLLAMA_ENV=(OLLAMA_CONTAINER_NAME="$OLLAMA_CTR" OLLAMA_IMAGE="$OLLAMA_IMAGE" OLLAMA_GPU="$GPU"
    OLLAMA_HOST_PORT="$OPORT" OLLAMA_VOLUME="$OLLAMA_VOLUME" OLLAMA_NUM_PARALLEL="$1"
    OLLAMA_KEEP_ALIVE=24h OLLAMA_RESTART_POLICY=no MEMOS_EMBEDDER_MODEL="$EMBED_MODEL")
  if [[ "$DRY" == 0 ]]; then
    docker rm -f "$OLLAMA_CTR" >/dev/null 2>&1
    CONTAINERS+=("$OLLAMA_CTR")
    # the model volume is shared: one pull at a time
    local lk="/tmp/memarena-locks-$(id -u)/ollama-pull" t0=$SECONDS
    until lock_try "$lk"; do (( SECONDS - t0 > 1800 )) && fail "timed out waiting for $lk"; sleep 2; done
    LOCKS+=("$lk")
  fi
  run "$LOG_DIR/ollama.log" env "${OLLAMA_ENV[@]}" bash "$GRID_SCRIPT_DIR/setup_memory_backends.sh" ollama
  local rc=$?
  [[ "$DRY" == 0 ]] && lock_release "/tmp/memarena-locks-$(id -u)/ollama-pull"
  [[ $rc == 0 ]] || fail "Ollama did not start (logs/ollama.log)"
  follow_logs "$OLLAMA_CTR" "$LOG_DIR/ollama.log"
  # a cold runner at day 0 has dropped embeddings before: warm it and check it answers
  run "$LOG_DIR/ollama.log" curl -fsS -m 120 "http://127.0.0.1:$OPORT/api/embed" \
    -d "{\"model\":\"$EMBED_MODEL\",\"input\":\"warm\"}" -o /dev/null || fail "Ollama embedding warm-up failed"
  say "Ollama ($EMBED_MODEL, NUM_PARALLEL=$1) on GPU $GPU, port $OPORT"
}

build_cache() {  # build_cache LOG ENV... -- ARGS...   (runs build_memory_cache.py, watched)
  local log="$1"; shift
  local -a envs=()
  while [[ $# -gt 0 && "$1" != "--" ]]; do envs+=("$1"); shift; done
  shift
  local cmd=(env ${envs[@]+"${envs[@]}"} "$PY" scripts/reproduce/build_memory_cache.py "$@")
  if [[ "$DRY" == 1 ]]; then
    echo "+ $(quote_cmd "${cmd[@]}") >>$(printf '%q' "$log") 2>&1"
    return 0
  fi
  start_bg "$log" "${cmd[@]}"
  watch_child "$BG_PID" "$log" build
}

CACHE_PATH=""
stage_begin 2 memory
EXTRACTOR_HF="$HF_MODEL"
case "$BACKEND" in
  vanilla|rag|oracle|dense_e5|dense_bge_m3|hybrid_rerank|temporal|text_sessions|omniscient)
    say "no memory service for $BACKEND"
    ;;

  memobase)
    if [[ -n "$MEM_FROM" ]]; then
      MEM_FROM="$(abspath "$MEM_FROM")"
      CACHE_PATH="$(ls "$MEM_FROM"/memory/memcache_memobase_A_paired_*.jsonl 2>/dev/null | head -1)"
      [[ -f "$MEM_FROM/memory/DONE.json" && -n "$CACHE_PATH" ]] || fail "-memory-from $MEM_FROM has no finished Memobase memory"
      EXTRACTOR_HF="$(helper progress-get "$MEM_FROM/memory/DONE.json" extractor)"
      say "answering from the Memobase memory of $MEM_FROM (extractor $EXTRACTOR_HF)"
    else
    MEM_DIR="$CELL_DIR/memory"
    CACHE_PATH="$MEM_DIR/memcache_memobase_A_paired_${MODEL}_${SEED}.jsonl"
    MEM_DONE="$MEM_DIR/DONE.json"
    if [[ "$FORCE" == 0 && -f "$MEM_DONE" && -f "$CACHE_PATH" ]]; then
      say "reusing the finished Memobase memory of this cell ($CACHE_PATH)"
    else
      [[ "$DRY" == 0 ]] && mkdir -p "$MEM_DIR" && rm -f "$MEM_DONE"
      MB_DIR="$MEM_DIR/memobase_service"          # checkout + database (kept after teardown)
      MB_SERVER="$MB_DIR/src/server"
      # a local mirror, so parallel cells clone v0.0.42 without refetching GitHub
      MB_REPO="${MEMOBASE_REPO:-}"
      if [[ -z "$MB_REPO" ]]; then
        mirror="$SERVICES_CACHE/memobase.git"
        if [[ ! -d "$mirror" ]]; then
          if [[ "$DRY" == 1 ]]; then
            echo "+ $(quote_cmd git clone --mirror https://github.com/memodb-io/memobase.git "$mirror")"
          else
            lk="$SERVICES_CACHE/.memobase-mirror.lock"
            until lock_try "$lk"; do sleep 2; done
            LOCKS+=("$lk")
            [[ -d "$mirror" ]] || run "$LOG_DIR/memobase_setup.log" git clone --mirror https://github.com/memodb-io/memobase.git "$mirror" \
              || { rm -rf "$mirror"; fail "could not mirror the Memobase repository"; }
            lock_release "$lk"
          fi
        fi
        MB_REPO="file://$mirror"
      fi
      # a new build starts from an empty store: keep any old database aside
      if [[ -d "$MB_SERVER/db" ]]; then
        stale="$MB_SERVER/db.stale.$(date +%Y%m%d-%H%M%S)"
        if [[ "$DRY" == 1 ]]; then echo "+ mv $MB_SERVER/db $stale"; else mv "$MB_SERVER/db" "$stale" || fail "cannot move old database aside"; say "old database moved to $stale"; fi
      fi
      if [[ "$DRY" == 0 && -f "$CACHE_PATH" ]]; then mv "$CACHE_PATH" "$CACHE_PATH.stale.$(date +%s)"; fi

      start_ollama 2
      API_PORT="$(pick_port 10)" && DB_PORT="$(pick_port 20)" && REDIS_PORT="$(pick_port 30)" \
        || fail "no free ports for Memobase near $((PORT_BASE + 10))"
      MB_ENV=(MEMOBASE_DIR="$MB_DIR" MEMOBASE_REPO="$MB_REPO" MEMORY_BACKENDS_SKIP_MEMOS=1
        MEMORY_READER_TAG="$MODEL" MEMORY_LLM_BASE_URL="$READER_URL_DOCKER" MEMORY_LLM_MODEL="$HF_MODEL"
        MEMORY_LLM_API_KEY=EMPTY OLLAMA_API_BASE="http://host.docker.internal:$OPORT"
        MEMOBASE_PORT="$API_PORT" MEMOBASE_DB_PORT="$DB_PORT" MEMOBASE_REDIS_PORT="$REDIS_PORT"
        MEMOBASE_COMPOSE_PROJECT="${PREFIX}-memobase" OVERWRITE_MEMORY_CONFIG=1
        ${OLLAMA_ENV[@]+"${OLLAMA_ENV[@]}"})
      if [[ "$DRY" == 0 ]]; then
        MB_STACK_UP=1   # from here on, teardown runs compose down
        progress "memobase_project=${PREFIX}-memobase" "memobase_port=$API_PORT" "ollama_port=$OPORT"
      fi
      say "Memobase v0.0.42 stack ${PREFIX}-memobase: api $API_PORT, postgres $DB_PORT, redis $REDIS_PORT; extractor $HF_MODEL"
      run "$LOG_DIR/memobase_setup.log" env "${MB_ENV[@]}" bash "$GRID_SCRIPT_DIR/setup_memory_backends.sh" start \
        || fail "Memobase stack did not start (logs/memobase_setup.log)"
      wait_http "http://127.0.0.1:$API_PORT/api/v1/healthcheck" 900 "Memobase API" "compose:${PREFIX}-memobase"

      build_cache "$LOG_DIR/memory_build.log" MEMOBASE_BASE_URL="http://localhost:$API_PORT" MEMOBASE_API_TOKEN=secret -- \
        --system memobase --run-dir "$DATA" --extractor-model "$HF_MODEL" \
        --extractor-endpoint "${READER_URL%/v1}" --extractor-api-key EMPTY --extractor-config A_paired \
        --output "$CACHE_PATH" --trial-seed "$TRIAL_SEED" --concurrency "${MEMOBASE_BUILD_CONCURRENCY:-32}"
      brc=$?
      if [[ "$DRY" == 0 ]]; then
        [[ $brc == 0 ]] || fail "Memobase build exited $brc (logs/memory_build.log)"
        msg="$(helper cache-check "${CACHE_PATH%.jsonl}.summary.json")" || fail "Memobase build incomplete: $msg"
        say "Memobase memory: $msg"
        helper write-args "$MEM_DONE" "system=memobase" "extractor=$HF_MODEL" "dataset=$DATA" "cache=$CACHE_PATH" "summary=$msg"
      fi
      say "tearing the Memobase stack down (database kept in $MB_SERVER/db)"
      [[ "$DRY" == 1 ]] && echo "+ $(quote_cmd env "${MB_ENV[@]}" bash "$GRID_SCRIPT_DIR/setup_memory_backends.sh" stop)"
      memobase_logs_and_down
      if [[ "$DRY" == 0 ]]; then docker rm -f "$OLLAMA_CTR" >/dev/null 2>&1; else echo "+ docker rm -f $OLLAMA_CTR"; fi
    fi
    fi
    ;;

  memsearch)
    # One cache for every reader and seed, as in the paper (extractor_model="shared").
    if [[ "$LIMIT" != 0 ]]; then MS_DIR="$MEMSEARCH_ROOT/limit$LIMIT"; else MS_DIR="$MEMSEARCH_ROOT/full"; fi
    CACHE_PATH="$MS_DIR/memcache_memsearch_A_paired_shared_s2.jsonl"
    MS_DONE="$MS_DIR/DONE.json" MS_LOCK="$MS_DIR/.build.lock"
    ms_ready() { [[ -f "$MS_DONE" && -f "$CACHE_PATH" ]]; }
    if ms_ready; then
      say "reusing the shared MemSearch cache $CACHE_PATH"
    elif [[ "$DRY" == 1 ]]; then
      echo "# shared MemSearch cache missing: this cell builds it (others wait on $MS_LOCK)"
      start_ollama 16
      build_cache "$LOG_DIR/memory_build.log" -u MEMSEARCH_EMBED_PROVIDER -u MEMSEARCH_EMBED_MODEL -u MEMSEARCH_EMBED_BASE_URL \
        -u MEMSEARCH_ROOT_DIR -u MEMSEARCH_MILVUS_URI OLLAMA_HOST="http://127.0.0.1:$OPORT" MEMSEARCH_ROOT_DIR_BASE="$MS_DIR/store" -- \
        --system memsearch --run-dir "$DATA" --extractor-model shared --extractor-endpoint "$READER_URL" \
        --extractor-api-key EMPTY --extractor-config A_paired --output "$CACHE_PATH" --trial-seed 1002 \
        --concurrency "${MEMSEARCH_BUILD_CONCURRENCY:-4}"
      echo "+ docker rm -f ${PREFIX}-ollama"
    else
      mkdir -p "$MS_DIR"
      t0=$SECONDS next=0
      until lock_try "$MS_LOCK"; do
        if (( SECONDS - t0 >= next )); then
          say "waiting for the shared MemSearch build (pid $(lock_owner "$MS_LOCK"), $(fmt_secs $((SECONDS - t0))))"
          next=$((next + 120))
        fi
        (( SECONDS - t0 > LOCK_TIMEOUT )) && fail "gave up waiting for $MS_LOCK after ${LOCK_TIMEOUT}s"
        sleep 10 & wait $!
      done
      LOCKS+=("$MS_LOCK")
      if ms_ready; then
        say "the shared MemSearch cache was built by another cell; reusing it"
      else
        say "building the shared MemSearch cache in $MS_DIR (one build for all readers and seeds)"
        for old in "$MS_DIR/store" "$CACHE_PATH"; do [[ -e "$old" ]] && mv "$old" "$old.stale.$(date +%s)"; done
        start_ollama 16
        build_cache "$LOG_DIR/memory_build.log" -u MEMSEARCH_EMBED_PROVIDER -u MEMSEARCH_EMBED_MODEL -u MEMSEARCH_EMBED_BASE_URL \
          -u MEMSEARCH_ROOT_DIR -u MEMSEARCH_MILVUS_URI OLLAMA_HOST="http://127.0.0.1:$OPORT" MEMSEARCH_ROOT_DIR_BASE="$MS_DIR/store" -- \
          --system memsearch --run-dir "$DATA" --extractor-model shared --extractor-endpoint "$READER_URL" \
          --extractor-api-key EMPTY --extractor-config A_paired --output "$CACHE_PATH" --trial-seed 1002 \
          --concurrency "${MEMSEARCH_BUILD_CONCURRENCY:-4}"
        brc=$?
        cp "$LOG_DIR/memory_build.log" "$MS_DIR/build.log" 2>/dev/null
        docker rm -f "$OLLAMA_CTR" >/dev/null 2>&1
        [[ $brc == 0 ]] || fail "MemSearch build exited $brc (logs/memory_build.log)"
        msg="$(helper cache-check "${CACHE_PATH%.jsonl}.summary.json")" || fail "MemSearch build incomplete: $msg"
        helper write-args "$MS_DONE" "system=memsearch" "extractor=shared" "dataset=$DATA" "cache=$CACHE_PATH" \
          "built_by=$CELL_DIR" "summary=$msg"
        say "shared MemSearch memory: $msg"
      fi
      lock_release "$MS_LOCK"
    fi
    if [[ "$DRY" == 0 ]]; then
      built_for="$(helper progress-get "$MS_DONE" dataset)"
      [[ "$built_for" == "$DATA" ]] || fail "shared MemSearch cache was built from $built_for, not $DATA (set MEMARENA_MEMSEARCH_DIR)"
      mkdir -p "$CELL_DIR/memory"
      ln -sfn "$CACHE_PATH" "$CELL_DIR/memory/$(basename "$CACHE_PATH")"
    fi
    ;;
esac
[[ "$DRY" == 0 && -n "$CACHE_PATH" ]] && progress "memory_cache=$CACHE_PATH"
stage_end

# ---------------------------------------------------------------- stage 3: answer
stage_begin 3 answer
answer_cmd=("$PY" -m eval.cli --run-dir "$DATA" --system "$SYSTEM" --output-dir "$CELL_DIR" --namespace "$NS"
  --model "$HF_MODEL" --endpoint "$READER_URL" --api-key EMPTY
  --trial-seed "$TRIAL_SEED" --temperature "$TEMP" --eval-concurrency 512 --force --log-every 50
  --d6-arm A --fail-on-thinking)
[[ -n "$TOPK" ]] && answer_cmd+=(--top-k "$TOPK")
[[ "$D6_ACCESS" == 1 ]] && answer_cmd+=(--d6-inject-access)
[[ "$D6_NO_CONTACTS" == 1 ]] && answer_cmd+=(--no-d6-owner-contacts)
[[ "$D6_NO_ASKER" == 1 ]] && answer_cmd+=(--no-d6-show-asker)
[[ -n "$DIMS" ]] && answer_cmd+=(--dimensions "${REDO_TASKS[@]}")
if [[ -n "$REDO" ]]; then
  # only these task files, into their own directory (the cell's answer file is spliced afterwards)
  for i in "${!answer_cmd[@]}"; do [[ "${answer_cmd[$i]}" == "$CELL_DIR" ]] && answer_cmd[$i]="$REDO_DIR"; done
  answer_cmd+=(--dimensions "${REDO_TASKS[@]}")
fi
case "$BACKEND" in
  # Vanilla / Oracle use the TEXT_SESSIONS prompt path, which never adds the playbook;
  # the flag only makes that explicit in args.json.
  vanilla) answer_cmd+=(--config eval/config/pipeline_vanilla512.yaml --stages answer --answer-concurrency 64
      --openclaw-soul-path __no_policy_playbook__) ;;
  oracle|omniscient)  answer_cmd+=(--config eval/config/pipeline.yaml --stages answer --answer-concurrency 64
      --openclaw-soul-path __no_policy_playbook__) ;;
  rag|dense_e5|dense_bge_m3|hybrid_rerank|temporal|text_sessions)
           answer_cmd+=(--config eval/config/pipeline.yaml --stages add search answer --answer-concurrency 64) ;;
  memobase)
    answer_cmd+=(--config eval/config/pipeline.yaml --stages search answer --answer-concurrency 32
      --cache-path "$CACHE_PATH" --expected-extractor "$EXTRACTOR_HF" --expected-memory-system memobase
      --expected-config A_paired --openclaw-soul-path __no_policy_playbook__) ;;
  memsearch)
    answer_cmd+=(--config eval/config/pipeline.yaml --stages search answer --answer-concurrency 32
      --cache-path "$CACHE_PATH" --no-cache-strict --expected-memory-system memsearch
      --expected-config A_paired --openclaw-soul-path __no_policy_playbook__) ;;
esac

ARGS_FILE="$CELL_DIR/args.json" ANSWER_LOG="$LOG_DIR/answer.log" ANSWER_OUT="$ANSWER_FILE"
if [[ -n "$REDO" ]]; then
  ARGS_FILE="$REDO_DIR/args.json" ANSWER_LOG="$LOG_DIR/answer_$(basename "$REDO_DIR").log" ANSWER_OUT="$REDO_ANSWER"
fi
if [[ "$DRY" == 0 ]]; then
  mkdir -p "$(dirname "$ARGS_FILE")"
  git_rev="$(git -C "$GRID_REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
  git_dirty="$( [[ -n "$(git -C "$GRID_REPO_ROOT" status --porcelain --untracked-files=no 2>/dev/null)" ]] && echo true || echo false)"
  helper write-args "$ARGS_FILE" "backend=$BACKEND" "system=$SYSTEM" "model=$MODEL" "model_name=$HF_MODEL" \
    "seed=$SEED" "temperature=$TEMP" "trial_seed=$TRIAL_SEED" "gpu=$GPU" "limit=$LIMIT" "namespace=$NS" \
    "dataset=$DATASET" "answered_dataset=$DATA" "items_total=$ITEMS_TOTAL" "cell_dir=$CELL_DIR" \
    "reader_endpoint=$READER_URL" "reader_api_key=EMPTY" "sglang_image=$SGLANG_IMAGE" \
    "sglang_flags=--tp 1 --dp 1 --context-length $SGLANG_CONTEXT_LENGTH --mem-fraction-static $SGLANG_MEM_FRACTION_STATIC --max-running-requests $SGLANG_MAX_RUNNING_REQUESTS $SGLANG_EXTRA_ARGS" \
    "sglang_pip_install=$SGLANG_PIP_INSTALL" "external_endpoint=${ENDPOINT:-null}" \
    "memory_cache=${CACHE_PATH:-null}" "d6_arm=A" "fail_on_thinking=true" \
    "tag=${TAG:-null}" "top_k=${TOPK:-null}" "memory_from=${MEM_FROM:-null}" "extractor=$EXTRACTOR_HF" \
    "d6_inject_access=$D6_ACCESS" "d6_no_contacts=$D6_NO_CONTACTS" "d6_no_asker=$D6_NO_ASKER" "dims=${DIMS:-null}" \
    "answer_command=$(quote_cmd "${answer_cmd[@]}")" "git_commit=$git_rev" "git_dirty=$git_dirty"
  say "eval.cli $SYSTEM ($ITEMS_TOTAL items, namespace $NS)"
  start_bg "$ANSWER_LOG" "${answer_cmd[@]}"
  watch_child "$BG_PID" "$ANSWER_LOG" answer
  arc=$?
  if [[ $arc != 0 ]]; then
    if grep -q "ThinkingEnabledError" "$ANSWER_LOG" 2>/dev/null; then
      fail "the reader returned <think> reasoning although thinking is off (--fail-on-thinking); see logs/$(basename "$ANSWER_LOG")"
    fi
    fail "eval.cli exited $arc (logs/$(basename "$ANSWER_LOG"))"
  fi
  msg="$(helper answers-check "$ANSWER_OUT" "$ITEMS_TOTAL")" || fail "answer file check: $msg ($ANSWER_OUT)"
  helper items "$PROGRESS" "$ITEMS_TOTAL" "$ITEMS_TOTAL" answered >/dev/null
  say "$msg"
  if [[ -n "$REDO" ]]; then
    msg="$(helper splice-answers "$ANSWER_FILE" "$REDO_ANSWER")" || fail "splicing the re-answered rows failed: $msg"
    progress "answers_redone=$REDO" "answers_redone_at=$(now_iso)"
    say "$msg"
  else
    progress "answer_file=$ANSWER_FILE" "answers_done=$ITEMS_TOTAL"
  fi
else
  echo "+ $(quote_cmd "${answer_cmd[@]}") >>$(printf '%q' "$ANSWER_LOG") 2>&1"
  [[ -n "$REDO" ]] && echo "+ grid_helper.py splice-answers $(printf '%q' "$ANSWER_FILE") $(printf '%q' "$REDO_ANSWER")"
fi
stage_end

# ---------------------------------------------------------------- stage 4: teardown
stage_begin 4 teardown
if [[ "$DRY" == 1 ]]; then
  [[ -n "$ENDPOINT" ]] || echo "+ docker rm -f ${PREFIX}-sglang"
else
  cleanup
fi
stage_end
if [[ "$DRY" == 0 ]]; then
  helper stage "$PROGRESS" 4 -
  FINAL_STATUS=done
  if [[ -n "$REDO" ]]; then
    progress redo_status=done "redo_ended_at=$(now_iso)" "redo_elapsed_s=$((SECONDS - T0))"
  else
    progress status=done "ended_at=$(now_iso)" "elapsed_s=$((SECONDS - T0))" message=null
  fi
  say "done in $(fmt_secs $((SECONDS - T0))): $ANSWER_FILE"
fi
exit 0
