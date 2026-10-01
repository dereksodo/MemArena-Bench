#!/usr/bin/env bash
# Judge one answered cell exactly as the paper did, with a chosen judge (-judge):
#   1. scripts/llmjudge.py, the judge via OpenRouter (--judge-preset remote,
#      concurrency 512)
#   2. scripts/rerun_d6_judge.py relabels every D6 (d4_permission) record with
#      the 5-label rubric, same judge.
# Judges (scripts/grid_common.sh, memarena/judges.py):
#   gpt4omini (default, primary)  openai/gpt-4o-mini-2024-07-18  *_judge_remote.json
#   deepseek  (secondary)         deepseek/deepseek-v4.1-flash   *_judge_deepseek.json
# The OpenRouter key comes from the environment or .env and is never put on a
# command line, printed or written anywhere by this script.

set -uo pipefail

# shellcheck source=scripts/grid_common.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/grid_common.sh"

usage() {
  cat <<'EOF'
Usage:
  scripts/judge_cell.sh -model M -backend B -seed S [-judge TAG] [-limit N] [-out DIR]
                        [-dataset DIR] [-concurrency N] [-workers N] [-force] [-redo DIMS] [-tag T] [-dry-run]

  Judges <out>/<backend>/<model>/<seed>/[limitN/] (default out/runs), which
  scripts/test_cell.sh must have finished, with judge TAG:
    gpt4omini  openai/gpt-4o-mini-2024-07-18  (default; the primary judge)
    deepseek   deepseek/deepseek-v4.1-flash   (secondary; agreement / kappa)
  Writes, next to the answers (each judge its own files; judges never touch
  each other's):
    TAG=gpt4omini: <system>/evaluation_results_<ns>_judge_remote.json   (the historical name)
                   logs/judge.log, logs/llmjudge_manifest.json
    TAG=deepseek:  <system>/evaluation_results_<ns>_judge_deepseek.json
                   logs/judge_deepseek.log, logs/llmjudge_manifest_deepseek.json
    (+ *_legacy.json next to the eval file: the file before the D6 relabel)
  and updates progress.json judges.TAG {status, stage, judged, total, accuracy,
  model, eval_file, ...}; gpt4omini also keeps the flat judge_* fields.
  A cell this judge already judged is skipped unless -force. No GPU is used.
  -redo DIMS (paper dimensions, e.g. D6 or D2,D3,D6) re-judges only those
  dimensions of an already judged cell: their records are marked as not judged
  (scripts/grid_helper.py judge-mark), then step 1 judges just the marked D1-D5
  records and step 2 relabels just the cleared D6 records; everything else in
  the file is kept. An interrupted redo resumes from the marks without -redo.

  Key: OPENROUTER_API_KEY in the environment or in .env at the repository root.
  -dataset defaults to the dataset recorded in the cell's args.json.
  -concurrency defaults to 512 (the paper's llmjudge setting): requests in flight in step 1.
  -workers defaults to 128: requests in flight in the D6 relabel (step 2).
  MEMARENA_JUDGE_ENDPOINT overrides https://openrouter.ai/api/v1 (only for testing
  against a local stub; the paper's numbers need OpenRouter).
EOF
}

MODEL="" BACKEND="" SEED="" LIMIT=0 OUT="" DATASET="" CONC=512 WORKERS=128 FORCE=0 DRY=0 REDO="" TAG=""
JUDGE="$GRID_DEFAULT_JUDGE"
need_val() { [[ $# -ge 2 && -n "$2" ]] || { echo "judge_cell: $1 needs a value" >&2; exit 2; }; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    -model|--model) need_val "$@"; MODEL="$2"; shift 2 ;;
    -backend|--backend) need_val "$@"; BACKEND="$2"; shift 2 ;;
    -seed|--seed) need_val "$@"; SEED="$2"; shift 2 ;;
    -limit|--limit) need_val "$@"; LIMIT="$2"; shift 2 ;;
    -out|--out) need_val "$@"; OUT="$2"; shift 2 ;;
    -dataset|--dataset) need_val "$@"; DATASET="$2"; shift 2 ;;
    -concurrency|--concurrency) need_val "$@"; CONC="$2"; shift 2 ;;
    -workers|--workers) need_val "$@"; WORKERS="$2"; shift 2 ;;
    -judge|--judge) need_val "$@"; JUDGE="$2"; shift 2 ;;
    -force|--force) FORCE=1; shift ;;
    -redo|--redo) need_val "$@"; REDO="$2"; shift 2 ;;
    -tag|--tag) need_val "$@"; TAG="$2"; shift 2 ;;
    -dry-run|--dry-run) DRY=1; shift ;;
    -h|-help|--help) usage; exit 0 ;;
    *) echo "judge_cell: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
bad() { echo "judge_cell: $*" >&2; exit 2; }
[[ -n "$MODEL" && -n "$BACKEND" && -n "$SEED" ]] || { usage >&2; bad "-model, -backend and -seed are required"; }
in_list "$MODEL" "$GRID_MODELS" || bad "unknown model '$MODEL' (valid: $GRID_MODELS)"
in_list "$BACKEND" "$GRID_ALL_BACKENDS" || bad "unknown backend '$BACKEND' (valid: $GRID_ALL_BACKENDS)"
in_list "$SEED" "$GRID_SEEDS" || bad "unknown seed '$SEED' (valid: $GRID_SEEDS)"
[[ "$LIMIT" =~ ^[0-9]+$ ]] || bad "-limit takes a positive integer"
[[ "$CONC" =~ ^[0-9]+$ ]] || bad "-concurrency takes a positive integer"
[[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || bad "-workers takes a positive integer"
in_list "$JUDGE" "$GRID_JUDGES" || bad "unknown judge '$JUDGE' (valid: $GRID_JUDGES)"
[[ -z "$REDO" || "$REDO" =~ ^([Dd][1-6][,[:space:]]*)+$ ]] || bad "-redo takes paper dimensions D1..D6, e.g. D6 or D2,D6"
[[ -z "$REDO" || "$FORCE" == 0 ]] || bad "-redo and -force exclude each other (-force re-judges everything)"
JUDGE_MODEL="$(judge_model "$JUDGE")"
JUDGE_FILE_TAG="$(judge_file_tag "$JUDGE")"
JSFX="$(judge_suffix "$JUDGE")"

cd "$GRID_REPO_ROOT" || exit 2
PY="$(grid_python)"
OUT="$(abspath "${OUT:-out/runs}")"
SYSTEM="$(backend_system "$BACKEND")"
NS="$(cell_namespace "$BACKEND" "$MODEL" "$SEED" "$TAG")"
CELL_DIR="$(cell_dir "$OUT" "$BACKEND" "$MODEL" "$SEED" "$LIMIT" "$TAG")"
LOG_DIR="$CELL_DIR/logs"
PROGRESS="$CELL_DIR/progress.json"
ANSWER_FILE="$CELL_DIR/$SYSTEM/answer_results_${NS}.json"
EVAL_FILE="$CELL_DIR/$SYSTEM/evaluation_results_${NS}_judge_${JUDGE_FILE_TAG}.json"
JUDGE_LOG="$LOG_DIR/judge${JSFX}.log"
JUDGE_LOCK="$CELL_DIR/.judge${JSFX}.lock"
LEGACY_FILE="${EVAL_FILE%.json}_legacy.json"
JUDGE_ENDPOINT="${MEMARENA_JUDGE_ENDPOINT:-https://openrouter.ai/api/v1}"
LABEL="$(backend_dirname "$BACKEND" "$TAG")/$MODEL/$SEED"; [[ "$LIMIT" != 0 ]] && LABEL="$LABEL/limit$LIMIT"
LABEL="$LABEL [$JUDGE]"

say() {
  local line="[judge $LABEL $(date +%H:%M:%S)] $*"
  echo "$line"
  if [[ "$DRY" == 0 && -d "$LOG_DIR" ]]; then echo "$line" >>"$JUDGE_LOG"; fi
}
# this judge's state in progress.json (judges.<tag>); other judges are never touched
progress() { [[ "$DRY" == 1 ]] || helper judge-set "$PROGRESS" "$JUDGE" "$@"; }
jget() { helper judge-get "$PROGRESS" "$JUDGE" "$1" 2>/dev/null; }
FINAL=""
fail() {
  say "FAILED: $*"
  FINAL=failed
  progress status=failed "message=$*" "ended_at=$(now_iso)"
  exit 1
}

# ---------------------------------------------------------------- preconditions
answers_status="$(helper progress-get "$PROGRESS" status 2>/dev/null)"
if [[ ! -f "$ANSWER_FILE" || "$answers_status" != "done" ]]; then
  if [[ "$DRY" == 1 ]]; then
    echo "# note: $LABEL has no finished answers yet ($ANSWER_FILE); showing the commands anyway"
  else
    bad "$LABEL has no finished answers (status '${answers_status:-none}', $ANSWER_FILE); run scripts/test_cell.sh first"
  fi
fi
pending_marks() { [[ -f "$EVAL_FILE" ]] && helper judge-pending "$EVAL_FILE" || echo 0; }
if [[ -n "$REDO" && ( ! -f "$EVAL_FILE" || "$(jget step1_done)" != "true" ) ]]; then
  bad "$LABEL has no finished judging by $JUDGE to redo dimensions of; judge it first (without -redo)"
fi
if [[ "$FORCE" == 0 && -z "$REDO" && -f "$EVAL_FILE" && "$(jget status)" == "done" && "$(pending_marks)" == 0 ]]; then
  echo "[skip] $LABEL is already judged by $JUDGE ($EVAL_FILE); use -force to re-judge"
  FINAL=skipped
  [[ -n "${MEMARENA_CELL_RESULT_FILE:-}" ]] && echo skipped >"$MEMARENA_CELL_RESULT_FILE"
  exit 0
fi

if [[ -z "$DATASET" ]]; then
  DATASET="$(helper progress-get "$CELL_DIR/args.json" answered_dataset 2>/dev/null)"
  [[ -n "$DATASET" ]] || DATASET="${MEMARENA_DATASET_DIR:-data/benchmark}"
fi
DATASET="$(abspath "$DATASET")"
[[ "$DRY" == 1 || -d "$DATASET/eval_instances" ]] || bad "no dataset at $DATASET (pass -dataset)"

has_key() {
  [[ -n "${OPENROUTER_API_KEY:-}" ]] && return 0
  [[ -f "$GRID_REPO_ROOT/.env" ]] && grep -Eq '^[[:space:]]*(export[[:space:]]+)?OPENROUTER_API_KEY=[^[:space:]]' "$GRID_REPO_ROOT/.env"
}
if [[ "$DRY" == 0 ]]; then
  has_key || bad "no OPENROUTER_API_KEY in the environment or in $GRID_REPO_ROOT/.env"
  mkdir -p "$LOG_DIR"
  lock_try "$JUDGE_LOCK" || bad "$LABEL is already being judged by $JUDGE (pid $(lock_owner "$JUDGE_LOCK"))"
fi
on_exit() {
  local rc=$?
  lock_release "$JUDGE_LOCK"
  if [[ "$DRY" == 0 && -z "$FINAL" && -f "$PROGRESS" ]]; then
    progress status=failed "message=exit code $rc" "ended_at=$(now_iso)"
    FINAL=failed
  fi
  if [[ -n "${MEMARENA_CELL_RESULT_FILE:-}" ]]; then
    local r="${FINAL:-failed}"; [[ "$DRY" == 1 && -z "$FINAL" ]] && r=dry-run
    echo "$r" >"$MEMARENA_CELL_RESULT_FILE"
  fi
}
trap on_exit EXIT
trap 'say "interrupted"; [[ -n "${JPID:-}" ]] && kill -TERM "$JPID" 2>/dev/null; exit 130' INT TERM HUP

run_logged() {  # run_logged cmd...   (log: $JUDGE_LOG; waits, reporting elapsed time)
  if [[ "$DRY" == 1 ]]; then
    echo "+ $(quote_cmd "$@") >>$(printf '%q' "$JUDGE_LOG") 2>&1"
    return 0
  fi
  "$@" >>"$JUDGE_LOG" 2>&1 &
  JPID=$!
  local t0=$SECONDS next=60
  while kill -0 "$JPID" 2>/dev/null; do
    if (( SECONDS - t0 >= next )); then say "still judging ($(fmt_secs $((SECONDS - t0))))"; next=$((next + 60)); fi
    sleep 1 & wait $!
  done
  wait "$JPID"
  local rc=$?
  JPID=""
  return $rc
}

if [[ -n "$REDO" ]]; then
  if [[ "$DRY" == 1 ]]; then
    echo "+ grid_helper.py judge-mark $(printf '%q' "$EVAL_FILE") $REDO"
  else
    msg="$(helper judge-mark "$EVAL_FILE" "$REDO")" || fail "judge-mark failed: $msg"
    say "redo $REDO: $msg"
  fi
fi

T0=$SECONDS
TOTAL="$( [[ -f "$ANSWER_FILE" ]] && python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$ANSWER_FILE" || echo '?')"
say "judging $TOTAL answers of $LABEL with $JUDGE_MODEL -> $(basename "$EVAL_FILE")"
progress status=running "started_at=$(now_iso)" "model=$JUDGE_MODEL" "eval_file=$EVAL_FILE" \
  "stage=1/2 llmjudge" judged=0 "total=$TOTAL" message=null

# ---------------------------------------------------------------- 1/2 llmjudge
step1_done="$(jget step1_done)"
pending="$(pending_marks)"
if [[ "$FORCE" == 0 && "$pending" != 0 ]]; then
  # only the records marked by judge-mark; the rest of the file is kept
  step1=("$PY" scripts/llmjudge.py --run-dir "$DATASET" --answer-path "$ANSWER_FILE"
    --judge-preset remote --judge-model "$JUDGE_MODEL" --judge-tag "$JUDGE_FILE_TAG"
    --judge-endpoint "$JUDGE_ENDPOINT" --concurrency "$CONC" --only-marked
    --manifest-out "$LOG_DIR/llmjudge_manifest${JSFX}.json")
  say "step 1/2: scripts/llmjudge.py --only-marked ($pending marked records)"
  run_logged "${step1[@]}" || fail "llmjudge.py failed (logs/$(basename "$JUDGE_LOG"))"
  [[ "$DRY" == 1 || "$(pending_marks)" == 0 ]] || fail "llmjudge.py left $(pending_marks) marked records"
  say "step 1/2 done: the $pending marked records re-judged"
elif [[ "$FORCE" == 0 && "$step1_done" == "true" && -f "$EVAL_FILE" ]]; then
  say "step 1/2 (llmjudge) already done; going on to the D6 relabel"
else
  if [[ "$FORCE" == 1 && -f "$LEGACY_FILE" ]]; then
    # the relabel backs up the file it first touches; a re-judge needs a fresh backup
    if [[ "$DRY" == 1 ]]; then echo "+ rm -f $(printf '%q' "$LEGACY_FILE")"; else rm -f "$LEGACY_FILE"; fi
  fi
  step1=("$PY" scripts/llmjudge.py --run-dir "$DATASET" --answer-path "$ANSWER_FILE"
    --judge-preset remote --judge-model "$JUDGE_MODEL" --judge-tag "$JUDGE_FILE_TAG"
    --judge-endpoint "$JUDGE_ENDPOINT" --concurrency "$CONC"
    --manifest-out "$LOG_DIR/llmjudge_manifest${JSFX}.json")
  [[ "$FORCE" == 1 ]] && step1+=(--force)
  # cleared first, so a step 1 that fails here is redone by the next run
  # instead of leaving the previous judging run's file marked done
  progress step1_done=false
  say "step 1/2: scripts/llmjudge.py"
  run_logged "${step1[@]}" || fail "llmjudge.py failed (logs/$(basename "$JUDGE_LOG"))"
  if [[ "$DRY" == 0 ]]; then
    [[ -f "$EVAL_FILE" ]] || fail "llmjudge.py wrote no $EVAL_FILE"
    judged="$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1])).get("details") or []))' "$EVAL_FILE")"
    progress step1_done=true "judged=$judged"
    say "step 1/2 done: $judged/$TOTAL records scored"
  fi
fi

# ---------------------------------------------------------------- 2/2 D6 relabel
progress "stage=2/2 d6 relabel"
step2=(env "OPENROUTER_BASE_URL=$JUDGE_ENDPOINT" "$PY" scripts/rerun_d6_judge.py --eval-path "$EVAL_FILE"
  --judge-model "$JUDGE_MODEL" --workers "$WORKERS" --dataset "$DATASET")
say "step 2/2: scripts/rerun_d6_judge.py (5-label D6 rubric)"
attempt=1
while :; do
  run_logged "${step2[@]}" || fail "rerun_d6_judge.py failed (logs/$(basename "$JUDGE_LOG"))"
  [[ "$DRY" == 1 ]] && break
  errs="$(grep -Eo 'records_error=[0-9]+' "$JUDGE_LOG" | tail -1 | cut -d= -f2)"
  [[ "${errs:-0}" == 0 ]] && break
  (( attempt >= 3 )) && fail "D6 relabel still has $errs PARSE_ERROR records after $attempt attempts"
  say "D6 relabel left $errs PARSE_ERROR records; retrying (it is idempotent)"
  attempt=$((attempt + 1))
done

# ---------------------------------------------------------------- check
if [[ "$DRY" == 0 ]]; then
  msg="$(helper judge-check "$EVAL_FILE" "$ANSWER_FILE")" || fail "judge check: $msg"
  judged="$(printf '%s' "$msg" | sed -E 's#^judged ([0-9]+)/.*#\1#')"
  acc="$(python3 -c 'import json,sys; print((json.load(open(sys.argv[1])).get("summary") or {}).get("accuracy"))' "$EVAL_FILE")"
  FINAL=done
  progress status=done "judged=$judged" "total=$TOTAL" "stage=2/2 d6 relabel" \
    "eval_file=$EVAL_FILE" "accuracy=$acc" "ended_at=$(now_iso)" "elapsed_s=$((SECONDS - T0))"
  say "$msg"
  say "done in $(fmt_secs $((SECONDS - T0))): $EVAL_FILE"
fi
exit 0
