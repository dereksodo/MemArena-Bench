#!/usr/bin/env bash
# Per-cell isolated memobase + memos stack manager.
#
# Each (reader-model, trial) pair gets its own dedicated docker compose
# project so concurrent build_memory_cache cells do not share a single
# postgres / redis / qdrant / neo4j (which fails under load with
# `psycopg2.OperationalError`). One ollama is reused across all stacks
# (embedding traffic is not the bottleneck).
#
# Design constraints:
#   - up to 5 reader models (0_6b llama3b 7b 8b 32b) × 4 trials (s1..s4)
#     = 20 stacks live at once. All ports fall in deterministic ranges.
#   - memobase config / MemOS env in the source dir tell each container's
#     LLM/embedding endpoints. We DO NOT clone the source per stack —
#     the data dirs (postgres / redis) are isolated, and named docker
#     volumes for qdrant/neo4j are isolated automatically because each
#     compose project gets its own namespace.
#
# Port plan (per (model, trial) slot = model_idx*10 + trial_idx, slot 1..54):
#   memobase api    18200 + slot
#   memobase db     19200 + slot
#   memobase redis  19250 + slot
#   memos api       18300 + slot
#
# Ollama (shared): 11434.
#
# Usage:
#   scripts/spin_per_trial_stacks.sh up llama3b s2,s3,s4
#   scripts/spin_per_trial_stacks.sh down llama3b s2,s3,s4
#   scripts/spin_per_trial_stacks.sh urls llama3b s2
#   scripts/spin_per_trial_stacks.sh status

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SERVICES_ROOT="${MEMARENA_MEMORY_SERVICES_ROOT:-$HOME/memarena-memory-services}"

# ---- index helpers ----------------------------------------------------
trial_idx() {
  case "$1" in
    s1) echo 1 ;; s2) echo 2 ;; s3) echo 3 ;; s4) echo 4 ;;
    *) echo "[spin] ERROR: unknown trial $1" >&2; return 2 ;;
  esac
}
model_idx() {
  case "$1" in
    0_6b)    echo 0 ;;
    llama3b) echo 1 ;;
    7b)      echo 2 ;;
    8b)      echo 3 ;;
    32b)     echo 4 ;;
    *) echo "[spin] ERROR: unknown model $1" >&2; return 2 ;;
  esac
}
cell_ports() {
  # echoes "MB_API MB_DB MB_REDIS MOS_API"
  local mi ti slot
  mi="$(model_idx "$1")"; ti="$(trial_idx "$2")"
  slot=$((mi * 10 + ti))
  printf '%s %s %s %s\n' "$((18200 + slot))" "$((19200 + slot))" "$((19250 + slot))" "$((18300 + slot))"
}

cell_paths() {
  # echoes "MEMOBASE_DIR MEMOS_DIR" — sources are per-model since LLM
  # endpoint is baked into per-model config.yaml + .env.
  local model="$1"
  local base="$SERVICES_ROOT/$model"
  printf '%s %s\n' "$base/memobase" "$base/MemOS"
}

require_dirs() {
  local model="$1"
  read -r MB_DIR MOS_DIR <<<"$(cell_paths "$model")"
  [[ -d "$MB_DIR/src/server" ]] || { echo "[spin] missing $MB_DIR/src/server" >&2; exit 2; }
  [[ -d "$MOS_DIR/docker" ]]    || { echo "[spin] missing $MOS_DIR/docker" >&2; exit 2; }
}

# ---- per-cell compose actions ----------------------------------------
up_cell() {
  local model="$1" trial="$2"
  require_dirs "$model"
  read -r MB_API MB_DB MB_REDIS MOS_API <<<"$(cell_ports "$model" "$trial")"
  local mb_proj="memarena_${model}_${trial}_memobase"
  local mos_proj="memarena_${model}_${trial}_memos"

  echo "[spin] up   model=$model trial=$trial → memobase api=$MB_API db=$MB_DB redis=$MB_REDIS, memos api=$MOS_API"

  # memobase: per-cell postgres/redis dir (host-mounted, isolated)
  ( cd "$MB_DIR/src/server" && \
    COMPOSE_PROJECT_NAME="$mb_proj" \
    DATABASE_NAME="memobase" DATABASE_USER="memobase" DATABASE_PASSWORD="memobase" \
    DATABASE_LOCATION="./db_${trial}/data" \
    REDIS_PASSWORD="memobase"  REDIS_LOCATION="./db_${trial}/redis" \
    DATABASE_EXPORT_PORT="$MB_DB" REDIS_EXPORT_PORT="$MB_REDIS" \
    API_EXPORT_PORT="$MB_API" \
    API_HOSTS="http://0.0.0.0:${MB_API},http://localhost:${MB_API}" \
    USE_CORS="false" PROJECT_ID="memobase_dev" \
    ACCESS_TOKEN="${MEMOBASE_API_TOKEN:-secret}" \
    docker compose up -d
  ) | tail -3

  # MemOS: per-cell api port. qdrant/neo4j are docker named volumes,
  # automatically isolated by COMPOSE_PROJECT_NAME.
  ( cd "$MOS_DIR/docker" && \
    COMPOSE_PROJECT_NAME="$mos_proj" \
    MEMOS_EXPORT_PORT="$MOS_API" \
    docker compose up -d
  ) | tail -3
}

down_cell() {
  local model="$1" trial="$2"
  require_dirs "$model" 2>/dev/null || return 0
  read -r MB_API MB_DB MB_REDIS MOS_API <<<"$(cell_ports "$model" "$trial")"

  ( cd "$MB_DIR/src/server" && \
    COMPOSE_PROJECT_NAME="memarena_${model}_${trial}_memobase" \
    DATABASE_LOCATION="./db_${trial}/data" REDIS_LOCATION="./db_${trial}/redis" \
    DATABASE_EXPORT_PORT="$MB_DB" REDIS_EXPORT_PORT="$MB_REDIS" \
    API_EXPORT_PORT="$MB_API" \
    docker compose down 2>/dev/null
  ) | tail -2 || true
  ( cd "$MOS_DIR/docker" && \
    COMPOSE_PROJECT_NAME="memarena_${model}_${trial}_memos" \
    MEMOS_EXPORT_PORT="$MOS_API" \
    docker compose down 2>/dev/null
  ) | tail -2 || true
}

urls_cell() {
  local model="$1" trial="$2"
  read -r MB_API MB_DB MB_REDIS MOS_API <<<"$(cell_ports "$model" "$trial")"
  cat <<EOF
MODEL=$model TRIAL=$trial
MEMOBASE_BASE_URL=http://localhost:${MB_API}
MEMOBASE_API_TOKEN=secret
MEMOS_BASE_URL=http://localhost:${MOS_API}
MEMOS_API_KEY=EMPTY
EOF
}

status_all() {
  docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' | \
    grep -E "memarena_.*_(memobase|memos)" || echo "(no per-cell stacks running)"
}

# ---- CLI -------------------------------------------------------------
action="${1:-}"; shift || true
model="${1:-}"; shift || true
trials_csv="${1:-s2,s3,s4}"

if [[ "$action" == "status" ]]; then
  status_all; exit 0
fi
[[ -z "$model" ]] && { echo "Usage: $0 {up|down|urls|status} MODEL [TRIALS_CSV]"; exit 2; }

IFS=',' read -r -a TRIAL_ARR <<<"$trials_csv"

case "$action" in
  up)   for t in "${TRIAL_ARR[@]}"; do up_cell   "$model" "$t"; done; status_all ;;
  down) for t in "${TRIAL_ARR[@]}"; do down_cell "$model" "$t"; done ;;
  urls) for t in "${TRIAL_ARR[@]}"; do urls_cell "$model" "$t"; echo; done ;;
  *) echo "Usage: $0 {up|down|urls|status} MODEL [TRIALS_CSV]"; exit 2 ;;
esac
