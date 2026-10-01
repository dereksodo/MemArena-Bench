#!/usr/bin/env bash
# Judge every answered cell of one or more seeds: models (outer) x backends
# (inner), each through scripts/judge_cell.sh, up to -parallel cells at a time.
# A failed cell is logged and skipped over.

set -uo pipefail

# shellcheck source=scripts/grid_common.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/grid_common.sh"

usage() {
  cat <<'EOF'
Usage:
  scripts/judge.sh -seed S [-judge TAG] [-parallel N] [-limit N] [-models "0_6b 8b"]
                   [-backends "vanilla rag"] [-out DIR] [-concurrency N] [-workers N]
                   [-force] [-redo DIMS] [-dry-run]

  Runs scripts/judge_cell.sh for every seed x model (0_6b llama3b 7b 8b 32b) x
  backend (vanilla rag memobase memsearch oracle), continues past failures, and
  prints a summary table. Cells without finished answers are reported as "no-answers".
  -seed takes one seed or a list ("s2 s3 s4" or s2,s3,s4).
  -judge TAG: gpt4omini (default, openai/gpt-4o-mini-2024-07-18, *_judge_remote.json)
              or deepseek (deepseek/deepseek-v4.1-flash, *_judge_deepseek.json).
  -parallel N judges up to N cells at once (default 4). With N > 1 each cell's
  console output goes to <out>/_grid_logs/judge_<tag>_<seeds>_<time>/<seed>_<model>_<backend>.out
  (the cell's own logs/ and progress.json are written as usual).
  -concurrency (default 512) and -workers (default 128) are per cell, so the
  requests in flight are up to N x that.
  -redo DIMS re-judges only those paper dimensions (e.g. D6 or D2,D3,D6) of
  already judged cells and keeps every other verdict (see scripts/judge_cell.sh).
  Needs OPENROUTER_API_KEY in the environment or .env; no GPU.
EOF
}

SEEDS="" MODELS="$GRID_MODELS" BACKENDS="$GRID_BACKENDS" OUT="out/runs" LIMIT=0 DRY=0 PAR=4 TAG=""
JUDGE="$GRID_DEFAULT_JUDGE"
PASS=()
need_val() { [[ $# -ge 2 && -n "$2" ]] || { echo "judge.sh: $1 needs a value" >&2; exit 2; }; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    -seed|--seed|-seeds|--seeds) need_val "$@"; SEEDS="${2//,/ }"; shift 2 ;;
    -models|--models) need_val "$@"; MODELS="${2//,/ }"; shift 2 ;;
    -backends|--backends) need_val "$@"; BACKENDS="${2//,/ }"; shift 2 ;;
    -judge|--judge) need_val "$@"; JUDGE="$2"; shift 2 ;;
    -parallel|--parallel) need_val "$@"; PAR="$2"; shift 2 ;;
    -out|--out) need_val "$@"; OUT="$2"; PASS+=("$1" "$2"); shift 2 ;;
    -limit|--limit) need_val "$@"; LIMIT="$2"; PASS+=("$1" "$2"); shift 2 ;;
    -concurrency|--concurrency|-workers|--workers) need_val "$@"; PASS+=("$1" "$2"); shift 2 ;;
    -dry-run|--dry-run) DRY=1; PASS+=("$1"); shift ;;
    -force|--force) PASS+=("$1"); shift ;;
    -redo|--redo) need_val "$@"; PASS+=("$1" "$2"); shift 2 ;;
    -tag|--tag) need_val "$@"; TAG="$2"; PASS+=("$1" "$2"); shift 2 ;;
    -h|-help|--help) usage; exit 0 ;;
    *) echo "judge.sh: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -n "${SEEDS// /}" ]] || { usage >&2; echo "judge.sh: -seed is required" >&2; exit 2; }
for s in $SEEDS; do in_list "$s" "$GRID_SEEDS" || { echo "judge.sh: unknown seed $s" >&2; exit 2; }; done
for m in $MODELS; do in_list "$m" "$GRID_MODELS" || { echo "judge.sh: unknown model $m" >&2; exit 2; }; done
for b in $BACKENDS; do in_list "$b" "$GRID_ALL_BACKENDS" || { echo "judge.sh: unknown backend $b" >&2; exit 2; }; done
in_list "$JUDGE" "$GRID_JUDGES" || { echo "judge.sh: unknown judge $JUDGE (valid: $GRID_JUDGES)" >&2; exit 2; }
[[ "$PAR" =~ ^[1-9][0-9]*$ ]] || { echo "judge.sh: -parallel takes a positive integer" >&2; exit 2; }
PASS+=(-judge "$JUDGE")

cd "$GRID_REPO_ROOT" || exit 2
OUT="$(abspath "$OUT")"
SEED_TAG="$(echo $SEEDS | tr ' ' '-')"
STAMP="$(date +%Y%m%d-%H%M%S)"
SUMMARY_LOG="$OUT/_grid_logs/judge_${JUDGE}_${SEED_TAG}_${STAMP}.log"
CELL_OUT_DIR="$OUT/_grid_logs/judge_${JUDGE}_${SEED_TAG}_${STAMP}"
[[ "$DRY" == 1 ]] || mkdir -p "$OUT/_grid_logs"
note() { echo "$*"; [[ "$DRY" == 1 ]] || echo "$*" >>"$SUMMARY_LOG"; }
# a dry run prints commands only, so it runs serially and inline
[[ "$DRY" == 1 ]] && PAR=1

RESULT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/memarena-judge.XXXXXX")"
PIDS=() SLOTS=()
cleanup() { rm -rf "$RESULT_DIR"; }
interrupted() {
  echo "judge.sh: interrupted"
  local p
  for p in ${PIDS[@]+"${PIDS[@]}"}; do kill -TERM "$p" 2>/dev/null; done
  wait 2>/dev/null
  exit 130
}
trap cleanup EXIT
trap interrupted INT TERM

# cells in order: seed, model, backend
CELLS=()
for s in $SEEDS; do for m in $MODELS; do for b in $BACKENDS; do CELLS+=("$s $m $b"); done; done; done
N=${#CELLS[@]}
declare -a RES T_START T_END
n_ok=0 n_skip=0 n_fail=0 n_missing=0
T0=$SECONDS

cell_dir_of() { set -- $1; cell_dir "$OUT" "$3" "$2" "$1" "$LIMIT" "$TAG"; }

start_cell() {  # start_cell INDEX   (background when PAR > 1)
  local i="$1" s m b
  set -- ${CELLS[$i]}; s="$1" m="$2" b="$3"
  T_START[$i]=$SECONDS
  : >"$RESULT_DIR/$i"
  note "[judge.sh] ---- $m x $b ($s) [$JUDGE]"
  if [[ "$PAR" == 1 ]]; then
    MEMARENA_CELL_RESULT_FILE="$RESULT_DIR/$i" bash "$GRID_SCRIPT_DIR/judge_cell.sh" \
      -model "$m" -backend "$b" -seed "$s" ${PASS[@]+"${PASS[@]}"}
    echo $? >"$RESULT_DIR/$i.rc"
  else
    mkdir -p "$CELL_OUT_DIR"
    MEMARENA_CELL_RESULT_FILE="$RESULT_DIR/$i" bash "$GRID_SCRIPT_DIR/judge_cell.sh" \
      -model "$m" -backend "$b" -seed "$s" ${PASS[@]+"${PASS[@]}"} \
      >"$CELL_OUT_DIR/${s}_${m}_${b}.out" 2>&1 &
    PIDS+=($!)
    SLOTS+=("$i")
  fi
}

finish_cell() {  # finish_cell INDEX RC
  local i="$1" rc="$2" res s m b dir
  set -- ${CELLS[$i]}; s="$1" m="$2" b="$3"
  dir="$(cell_dir_of "${CELLS[$i]}")"
  T_END[$i]=$SECONDS
  [[ $rc == 130 ]] && { note "[judge.sh] interrupted during $m x $b ($s)"; exit 130; }
  res="$(cat "$RESULT_DIR/$i" 2>/dev/null)"; [[ -n "$res" ]] || res=failed
  [[ $rc != 0 ]] && res=failed
  case "$res" in done|dry-run) n_ok=$((n_ok + 1)) ;; skipped) n_skip=$((n_skip + 1)) ;; *) n_fail=$((n_fail + 1)) ;; esac
  RES[$i]="$res"
  if [[ "$res" == failed ]]; then
    note "[judge.sh] FAILED: $m x $b ($s); see $dir/logs/judge$(judge_suffix "$JUDGE").log"
  elif [[ "$PAR" != 1 ]]; then
    note "[judge.sh] $res: $m x $b ($s) in $(fmt_secs $((T_END[$i] - T_START[$i])))"
  fi
}

reap() {  # wait until a running cell finishes; record it
  local k pid keep_p=() keep_s=() reaped=0
  while :; do
    keep_p=() keep_s=()
    for k in "${!PIDS[@]}"; do
      pid="${PIDS[$k]}"
      if kill -0 "$pid" 2>/dev/null; then
        keep_p+=("$pid"); keep_s+=("${SLOTS[$k]}")
      else
        wait "$pid"; finish_cell "${SLOTS[$k]}" $?
        reaped=1
      fi
    done
    PIDS=(${keep_p[@]+"${keep_p[@]}"}); SLOTS=(${keep_s[@]+"${keep_s[@]}"})
    [[ $reaped == 1 || ${#PIDS[@]} == 0 ]] && return 0
    sleep 1 & wait $!
  done
}

for ((i = 0; i < N; i++)); do
  dir="$(cell_dir_of "${CELLS[$i]}")"
  if [[ "$DRY" == 0 && "$(helper progress-get "$dir/progress.json" status 2>/dev/null)" != "done" ]]; then
    RES[$i]=no-answers; T_START[$i]=$SECONDS; T_END[$i]=$SECONDS
    n_missing=$((n_missing + 1))
    continue
  fi
  if [[ "$PAR" == 1 ]]; then
    start_cell "$i"
    finish_cell "$i" "$(cat "$RESULT_DIR/$i.rc")"
  else
    while (( ${#PIDS[@]} >= PAR )); do reap; done
    start_cell "$i"
  fi
done
while (( ${#PIDS[@]} > 0 )); do reap; done

note ""
note "==== judge.sh summary: judge $JUDGE, seed(s) $SEEDS, $(fmt_secs $((SECONDS - T0))), parallel $PAR ===="
note "$(printf '%-4s %-8s %-10s %-11s %8s  %-8s %s' seed model backend result time accuracy cell_dir)"
for ((i = 0; i < N; i++)); do
  set -- ${CELLS[$i]}
  dir="$(cell_dir_of "${CELLS[$i]}")"
  acc="$(helper judge-get "$dir/progress.json" "$JUDGE" accuracy 2>/dev/null)"
  note "$(printf '%-4s %-8s %-10s %-11s %8s  %-8s %s' "$1" "$2" "$3" "${RES[$i]:-?}" \
    "$(fmt_secs $(( ${T_END[$i]:-0} - ${T_START[$i]:-0} )))" "${acc:0:6}" "$dir")"
done
note "judged $n_ok, skipped $n_skip, failed $n_fail, without answers $n_missing"
[[ "$DRY" == 1 ]] || note "(this summary: $SUMMARY_LOG)"
[[ "$PAR" != 1 && "$DRY" == 0 ]] && note "(per-cell console output: $CELL_OUT_DIR)"
[[ $n_fail == 0 ]]
