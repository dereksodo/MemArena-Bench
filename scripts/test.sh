#!/usr/bin/env bash
# All answer cells of one seed on one GPU: models (outer) x backends (inner),
# each through scripts/test_cell.sh. A failed cell is logged and skipped over.

set -uo pipefail

# shellcheck source=scripts/grid_common.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/grid_common.sh"

usage() {
  cat <<'EOF'
Usage:
  scripts/test.sh -seed S -gpu G [-limit N] [-models "0_6b 8b"] [-backends "vanilla rag"]
                  [-dataset DIR] [-out DIR] [-force] [-redo-dims DIMS] [-dry-run]

  Runs scripts/test_cell.sh for every model (outer loop: 0_6b llama3b 7b 8b 32b) and
  backend (inner loop: vanilla rag memobase memsearch oracle) on GPU G, continues past
  failed cells, and prints a summary table at the end. Finished cells are skipped
  unless -force. -limit, -dataset, -out, -force, -redo-dims and -dry-run are passed to every cell
  (-endpoint URL too: every cell then uses that already-running reader; testing only).

  Run one test.sh per GPU to use several GPUs, e.g.
    scripts/test.sh -seed s2 -gpu 0 &  scripts/test.sh -seed s3 -gpu 1 &
  The shared MemSearch cache is built by whichever cell gets there first; the others wait.
EOF
}

SEED="" GPU="" MODELS="$GRID_MODELS" BACKENDS="$GRID_BACKENDS" TAG=""
PASS=()
need_val() { [[ $# -ge 2 && -n "$2" ]] || { echo "test.sh: $1 needs a value" >&2; exit 2; }; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    -seed|--seed) need_val "$@"; SEED="$2"; shift 2 ;;
    -gpu|--gpu) need_val "$@"; GPU="$2"; shift 2 ;;
    -models|--models) need_val "$@"; MODELS="${2//,/ }"; shift 2 ;;
    -backends|--backends) need_val "$@"; BACKENDS="${2//,/ }"; shift 2 ;;
    -limit|--limit|-dataset|--dataset|-out|--out|-endpoint|--endpoint|-redo-dims|--redo-dims|-tag|--tag|-top-k|--top-k|-memory-from|--memory-from|-dims|--dims) need_val "$@"; PASS+=("$1" "$2"); shift 2 ;;
    -force|--force|-dry-run|--dry-run|-d6-access|--d6-access|-d6-no-contacts|--d6-no-contacts|-d6-no-asker|--d6-no-asker) PASS+=("$1"); shift ;;
    -h|-help|--help) usage; exit 0 ;;
    *) echo "test.sh: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -n "$SEED" && -n "$GPU" ]] || { usage >&2; echo "test.sh: -seed and -gpu are required" >&2; exit 2; }
in_list "$SEED" "$GRID_SEEDS" || { echo "test.sh: unknown seed $SEED" >&2; exit 2; }
for m in $MODELS; do in_list "$m" "$GRID_MODELS" || { echo "test.sh: unknown model $m" >&2; exit 2; }; done
for b in $BACKENDS; do in_list "$b" "$GRID_ALL_BACKENDS" || { echo "test.sh: unknown backend $b" >&2; exit 2; }; done

OUT="out/runs" DRY=0 LIMIT=0
set -- ${PASS[@]+"${PASS[@]}"}
while [[ $# -gt 0 ]]; do
  case "$1" in -out|--out) OUT="$2"; shift ;; -limit|--limit) LIMIT="$2"; shift ;; -tag|--tag) TAG="$2"; shift ;; -dry-run|--dry-run) DRY=1 ;; esac
  shift
done
cd "$GRID_REPO_ROOT" || exit 2
OUT="$(abspath "$OUT")"
STAMP="$(date +%Y%m%d-%H%M%S)"
GRID_LOG_DIR="$OUT/_grid_logs"
SUMMARY_LOG="$GRID_LOG_DIR/test_${SEED}_gpu${GPU}_${STAMP}.log"
[[ "$DRY" == 1 ]] || mkdir -p "$GRID_LOG_DIR"
note() { echo "$*"; [[ "$DRY" == 1 ]] || echo "$*" >>"$SUMMARY_LOG"; }

RESULT_FILE="$(mktemp "${TMPDIR:-/tmp}/memarena-cell.XXXXXX")"
trap 'rm -f "$RESULT_FILE"' EXIT
trap 'echo "test.sh: interrupted"; exit 130' INT TERM

ROWS=() n_done=0 n_skip=0 n_fail=0
T0=$SECONDS
note "[test.sh] seed $SEED on GPU $GPU; models: $MODELS; backends: $BACKENDS"
for m in $MODELS; do
  for b in $BACKENDS; do
    t=$SECONDS
    : >"$RESULT_FILE"
    note "[test.sh] ---- $m x $b ($SEED, GPU $GPU)"
    MEMARENA_CELL_RESULT_FILE="$RESULT_FILE" bash "$GRID_SCRIPT_DIR/test_cell.sh" \
      -model "$m" -backend "$b" -seed "$SEED" -gpu "$GPU" ${PASS[@]+"${PASS[@]}"}
    rc=$?
    [[ $rc == 130 ]] && { note "[test.sh] interrupted during $m x $b"; exit 130; }
    res="$(cat "$RESULT_FILE" 2>/dev/null)"; [[ -n "$res" ]] || res=failed
    [[ $rc != 0 ]] && res=failed
    case "$res" in done|dry-run) n_done=$((n_done + 1)) ;; skipped) n_skip=$((n_skip + 1)) ;; *) n_fail=$((n_fail + 1)) ;; esac
    ROWS+=("$(printf '%-8s %-10s %-8s %8s  %s' "$m" "$b" "$res" "$(fmt_secs $((SECONDS - t)))" "$(cell_dir "$OUT" "$b" "$m" "$SEED" "$LIMIT" "$TAG")")")
    [[ "$res" == failed ]] && note "[test.sh] FAILED: $m x $b (rc $rc); see $(cell_dir "$OUT" "$b" "$m" "$SEED" "$LIMIT" "$TAG")/logs/"
  done
done

note ""
note "==== test.sh summary: seed $SEED, GPU $GPU, $(fmt_secs $((SECONDS - T0))) ===="
note "$(printf '%-8s %-10s %-8s %8s  %s' model backend result time cell_dir)"
for r in "${ROWS[@]}"; do note "$r"; done
note "done $n_done, skipped $n_skip, failed $n_fail"
[[ "$DRY" == 1 ]] || note "(this summary: $SUMMARY_LOG)"
[[ $n_fail == 0 ]]
