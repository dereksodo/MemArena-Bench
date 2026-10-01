# shellcheck shell=bash
# Shared definitions for the grid runners: scripts/test_cell.sh, test.sh,
# judge_cell.sh and judge.sh. Sourced, never executed. Works with bash 3.2
# (macOS) as well as bash 4/5, so no associative arrays.

GRID_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRID_REPO_ROOT="$(cd "$GRID_SCRIPT_DIR/.." && pwd)"
GRID_HELPER="$GRID_SCRIPT_DIR/grid_helper.py"

GRID_MODELS="0_6b llama3b 7b 8b 32b"
GRID_BACKENDS="vanilla rag memobase memsearch oracle"
# Ablation backends: runnable cell by cell (and in test.sh / judge.sh with -backends),
# never part of the default main-grid loops.
#   dense_e5 / dense_bge_m3   dense retrieval (intfloat/e5-large-v2, BAAI/bge-m3)
#   hybrid_rerank             BM25 top-20 + BAAI/bge-reranker-v2-m3
#   temporal                  time-aware retrieval
#   text_sessions             BM25 hits rendered as TEXT_SESSIONS (prompt-format control)
#   omniscient                every session of the world, not just the ego's
GRID_ABLATION_BACKENDS="dense_e5 dense_bge_m3 hybrid_rerank temporal text_sessions omniscient"
GRID_ALL_BACKENDS="$GRID_BACKENDS $GRID_ABLATION_BACKENDS"
GRID_SEEDS="s1 s2 s3 s4"

# Judges (scripts/llmjudge.py --judge-preset remote, then scripts/rerun_d6_judge.py
# for D6), by tag. Keep in sync with memarena/judges.py (tests check it).
#   gpt4omini  openai/gpt-4o-mini-2024-07-18  primary     *_judge_remote.json (historical name)
#   deepseek   deepseek/deepseek-v4.1-flash   secondary   *_judge_deepseek.json
GRID_JUDGES="deepseek gpt4omini"
GRID_DEFAULT_JUDGE="gpt4omini"

judge_model() {
  case "$1" in
    deepseek) echo "deepseek/deepseek-v4.1-flash" ;;
    gpt4omini) echo "openai/gpt-4o-mini-2024-07-18" ;;
    *) return 1 ;;
  esac
}
# tag in evaluation_results_<ns>_judge_<file tag>.json
judge_file_tag() { case "$1" in gpt4omini) echo remote ;; deepseek) echo deepseek ;; *) return 1 ;; esac; }
# gpt4omini keeps the file names of the first runners (judge.log, .judge.lock, ...)
judge_suffix() { case "$1" in gpt4omini) echo "" ;; *) echo "_$1" ;; esac; }

# Python: $PYTHON, else the repo venv, else python3.
grid_python() {
  if [[ -n "${PYTHON:-}" ]]; then
    printf '%s' "$PYTHON"
  elif [[ -x "$GRID_REPO_ROOT/.venv/bin/python" ]]; then
    printf '%s' "$GRID_REPO_ROOT/.venv/bin/python"
  else
    printf '%s' "python3"
  fi
}

helper() { python3 "$GRID_HELPER" "$@"; }

in_list() {  # in_list WORD "LIST"
  local w
  for w in $2; do [[ "$w" == "$1" ]] && return 0; done
  return 1
}

model_hf() {
  case "$1" in
    0_6b) echo "Qwen/Qwen3-0.6B" ;;
    llama3b) echo "meta-llama/Llama-3.2-3B-Instruct" ;;
    7b) echo "mistralai/Mistral-7B-Instruct-v0.3" ;;
    8b) echo "Qwen/Qwen3-8B" ;;
    32b) echo "Qwen/Qwen3-32B-AWQ" ;;
    *) return 1 ;;
  esac
}

# s1 is the deterministic pass (T=0.0); s2-s4 are the stochastic seeds
# (T=0.3) behind the paper's mean +- std. The trial seed is provenance only.
seed_temperature() { case "$1" in s1) echo "0.0" ;; s2|s3|s4) echo "0.3" ;; *) return 1 ;; esac; }
seed_trial() { case "$1" in s1) echo 1001 ;; s2) echo 1002 ;; s3) echo 1003 ;; s4) echo 1004 ;; *) return 1 ;; esac; }

# The paper calls the BM25 baseline "inmem" in file names and "RAG" in tables.
backend_paper() { case "$1" in rag) echo "inmem" ;; *) echo "$1" ;; esac; }
backend_system() {
  case "$1" in
    vanilla) echo vanilla ;;
    oracle) echo oracle ;;
    rag) echo inmem ;;
    memobase|memsearch) echo memory_cache ;;
    dense_e5|dense_bge_m3|temporal|omniscient) echo "$1" ;;
    hybrid_rerank) echo hybrid_bm25rerank ;;
    text_sessions) echo inmem_text_sessions ;;
    *) return 1 ;;
  esac
}

cell_namespace() {  # backend model seed [tag]
  if [[ -n "${4:-}" ]]; then echo "$(backend_paper "$1")_${4}_$2_$3"; else echo "$(backend_paper "$1")_$2_$3"; fi
}

# a variant run (-tag T) lives next to the main cell: <out>/<backend>-<T>/<model>/<seed>
backend_dirname() { if [[ -n "${2:-}" ]]; then echo "$1-$2"; else echo "$1"; fi; }   # backend [tag]

cell_dir() {  # out backend model seed limit [tag]
  local d="$1/$(backend_dirname "$2" "${6:-}")/$3/$4"
  if [[ -n "${5:-}" && "${5:-0}" != "0" ]]; then d="$d/limit$5"; fi
  printf '%s' "$d"
}

abspath() {  # absolute path, whether or not it exists yet
  case "$1" in
    /*) printf '%s' "$1" ;;
    *) printf '%s/%s' "$PWD" "$1" ;;
  esac
}

now_iso() { date -u +%Y-%m-%dT%H:%M:%SZ; }

fmt_secs() {
  local s="$1"
  if (( s >= 3600 )); then printf '%dh%02dm' $((s / 3600)) $((s % 3600 / 60))
  elif (( s >= 60 )); then printf '%dm%02ds' $((s / 60)) $((s % 60))
  else printf '%ds' "$s"; fi
}

# print a command the way a shell would read it back
quote_cmd() {
  local out="" part
  for part in "$@"; do out="$out $(printf '%q' "$part")"; done
  printf '%s' "${out# }"
}

# mkdir-based lock (portable; flock is not on macOS). The owner's PID is
# stored inside, so a lock left behind by a killed process is taken over.
lock_try() {  # lock_try DIR
  local dir="$1" pid
  mkdir -p "$(dirname "$dir")"
  if mkdir "$dir" 2>/dev/null; then
    echo "$$" >"$dir/pid"
    return 0
  fi
  pid="$(cat "$dir/pid" 2>/dev/null || true)"
  if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
    rm -rf "$dir"
    if mkdir "$dir" 2>/dev/null; then
      echo "$$" >"$dir/pid"
      return 0
    fi
  fi
  return 1
}

lock_owner() { cat "$1/pid" 2>/dev/null || true; }

lock_release() {  # only if we own it
  local dir="$1"
  if [[ -d "$dir" && "$(lock_owner "$dir")" == "$$" ]]; then
    rm -rf "$dir"
  fi
}
