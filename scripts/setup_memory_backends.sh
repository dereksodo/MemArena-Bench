#!/usr/bin/env bash
# Set up the official Memobase and MemOS services used by MemArena.
#
# The batch/concurrent ingest logic lives in MemArena's adapters. These
# services are pinned to the official versions we used for reproduction and
# are configured to avoid the reader-model SGLang host ports. MemOS and
# Memobase still listen on their upstream container ports internally.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MEMOBASE_REPO="${MEMOBASE_REPO:-https://github.com/memodb-io/memobase.git}"
MEMOBASE_REF="${MEMOBASE_REF:-v0.0.42}"
MEMOS_REPO="${MEMOS_REPO:-https://github.com/MemTensor/MemOS.git}"
MEMOS_REF="${MEMOS_REF:-v2.0.13}"

SERVICES_DIR="${MEMARENA_MEMORY_SERVICES_DIR:-$HOME/memarena-memory-services}"
ALL_SERVICES_ROOT="${MEMARENA_MEMORY_SERVICES_ROOT:-${MEMARENA_MEMORY_SERVICES_DIR:-$HOME/memarena-memory-services}}"
MEMOBASE_DIR="${MEMOBASE_DIR:-$SERVICES_DIR/memobase}"
MEMOS_DIR="${MEMOS_DIR:-$SERVICES_DIR/MemOS}"

MEMOBASE_PORT="${MEMOBASE_PORT:-8019}"
MEMOBASE_DB_PORT="${MEMOBASE_DB_PORT:-15432}"
MEMOBASE_REDIS_PORT="${MEMOBASE_REDIS_PORT:-16379}"
MEMOBASE_API_TOKEN="${MEMOBASE_API_TOKEN:-secret}"
# Postgres max_connections for the Memobase stack. The stock value (100) is
# below what the Memobase server's connection pool opens under the concurrency
# MemArena builds use (pool 125), so Postgres refused clients ("too many
# clients") and Memobase answered HTTP 200 + {"errno": 500}: inserts silently
# lost. New stacks get headroom above the pool.
MEMOBASE_PG_MAX_CONNECTIONS="${MEMOBASE_PG_MAX_CONNECTIONS:-300}"

MEMOS_PORT="${MEMOS_PORT:-8020}"
MEMOS_API_KEY="${MEMOS_API_KEY:-EMPTY}"

MEMORY_COMPOSE_PROJECT_PREFIX="${MEMORY_COMPOSE_PROJECT_PREFIX:-}"
MEMOBASE_COMPOSE_PROJECT="${MEMOBASE_COMPOSE_PROJECT:-}"
MEMOS_COMPOSE_PROJECT="${MEMOS_COMPOSE_PROJECT:-}"
if [[ -n "$MEMORY_COMPOSE_PROJECT_PREFIX" ]]; then
  MEMOBASE_COMPOSE_PROJECT="${MEMOBASE_COMPOSE_PROJECT:-${MEMORY_COMPOSE_PROJECT_PREFIX}_memobase}"
  MEMOS_COMPOSE_PROJECT="${MEMOS_COMPOSE_PROJECT:-${MEMORY_COMPOSE_PROJECT_PREFIX}_memos}"
fi

MEMORY_READER_TAG="${MEMORY_READER_TAG:-0_6b}"
MEMORY_LLM_BASE_URL="${MEMORY_LLM_BASE_URL:-}"
MEMORY_LLM_MODEL="${MEMORY_LLM_MODEL:-}"
MEMORY_LLM_API_KEY="${MEMORY_LLM_API_KEY:-EMPTY}"

MEMORY_READER_SPECS=(
  "0_6b|Qwen/Qwen3-0.6B|http://host.docker.internal:16000/v1"
  "llama3b|meta-llama/Llama-3.2-3B-Instruct|http://host.docker.internal:16001/v1"
  "7b|mistralai/Mistral-7B-Instruct-v0.3|http://host.docker.internal:16002/v1"
  "8b|Qwen/Qwen3-8B|http://host.docker.internal:16003/v1"
  "32b|Qwen/Qwen3-32B-AWQ|http://host.docker.internal:16004/v1"
)
MEMORY_READER_TAGS=(0_6b llama3b 7b 8b 32b)

OLLAMA_API_BASE="${OLLAMA_API_BASE:-http://host.docker.internal:11434}"
MEMOS_EMBEDDER_MODEL="${MEMOS_EMBEDDER_MODEL:-nomic-embed-text:latest}"
MEMOS_EMBEDDING_DIMENSION="${MEMOS_EMBEDDING_DIMENSION:-768}"

OVERWRITE_MEMORY_CONFIG="${OVERWRITE_MEMORY_CONFIG:-0}"
ALLOW_DIRTY="${MEMORY_BACKENDS_ALLOW_DIRTY:-0}"
UPDATE_BACKENDS="${MEMORY_BACKENDS_UPDATE:-0}"
BUILD_IMAGES="${MEMORY_BACKENDS_BUILD:-0}"
ISOLATED_CHILD="${MEMORY_BACKENDS_ISOLATED_CHILD:-0}"
PARALLEL_ISOLATED="${MEMORY_BACKENDS_PARALLEL:-1}"
# 1 = Memobase only: do not clone, configure, start or stop MemOS. The grid
# runners (scripts/test_cell.sh) use this; MemOS is not part of the main grid.
SKIP_MEMOS="${MEMORY_BACKENDS_SKIP_MEMOS:-0}"

DOCKER_COMPOSE=()
COMPOSE_UP_FLAGS=(-d)

usage() {
  cat <<EOF
Usage:
  scripts/setup_memory_backends.sh [setup|start|restart] [READER_TAG ...]
  scripts/setup_memory_backends.sh [setup-all|start-all|restart-all|stop-all|status|env|versions|versions-all]

Pinned official service versions:
  Memobase  ${MEMOBASE_REPO}  ${MEMOBASE_REF}
  MemOS     ${MEMOS_REPO}  ${MEMOS_REF}

Default local host endpoints:
  Memobase  http://localhost:${MEMOBASE_PORT}
  MemOS     http://localhost:${MEMOS_PORT}

Actions:
  setup     Clone/update the pinned service repos and write local config files.
            With reader tags, set up isolated per-reader stacks on paper ports.
  start     Run setup, then start both services with docker compose.
            With reader tags, start isolated per-reader stacks on paper ports.
  stop      Stop both docker compose projects if present.
  status    Show matching docker containers.
  env       Print exports needed by MemArena evaluation.
  versions  Print pinned repos, refs, and local ports.

Common overrides:
  MEMARENA_MEMORY_SERVICES_DIR   Parent directory. Default: ~/memarena-memory-services
  MEMOBASE_DIR, MEMOS_DIR        Explicit checkout directories.
  MEMOBASE_PORT                  Host port. Default: 8019
  MEMOBASE_DB_PORT               Host Postgres port. Default: 15432
  MEMOBASE_REDIS_PORT            Host Redis port. Default: 16379
  MEMOBASE_PG_MAX_CONNECTIONS    Postgres max_connections of the Memobase
                                  stack (must exceed the server's connection
                                  pool, 125). Default: 300
  MEMOS_PORT                     Host port mapped to container 8000. Default: 8020
  MEMORY_COMPOSE_PROJECT_PREFIX  Optional prefix for isolated docker compose
                                  projects, useful when running one memory stack
                                  per reader model.
  MEMARENA_MEMORY_SERVICES_ROOT  Parent directory for setup-all/start-all/
                                  restart-all. Default: ~/memarena-memory-services
  MEMOBASE_COMPOSE_PROJECT       Explicit Memobase compose project name.
  MEMOS_COMPOSE_PROJECT          Explicit MemOS compose project name.
  MEMORY_READER_TAG              Extractor reader tag. Default: 0_6b
                                  Valid: 0_6b, llama3b, 7b, 8b, 32b
  MEMORY_LLM_BASE_URL            Manual extractor endpoint override.
  MEMORY_LLM_MODEL               Manual extractor model override.
  OLLAMA_API_BASE                Default: http://host.docker.internal:11434
  OVERWRITE_MEMORY_CONFIG=1      Regenerate service config/env files.
  MEMORY_BACKENDS_UPDATE=1       Fetch pinned repos even when local refs exist.
                                  Default: 0, reuse local checkouts.
  MEMORY_BACKENDS_BUILD=1        Force docker compose --build.
                                  Default: 0, reuse local images when possible.
  MEMORY_BACKENDS_PARALLEL=0     Run multi-reader setup/start/restart/stop
                                  serially. Default: 1, parallel.
  MEMORY_BACKENDS_SKIP_MEMOS=1   Memobase only: leave MemOS alone.
  OLLAMA_CONTAINER_NAME, OLLAMA_IMAGE, OLLAMA_GPU, OLLAMA_HOST_PORT,
  OLLAMA_VOLUME, OLLAMA_NUM_PARALLEL, OLLAMA_KEEP_ALIVE,
  OLLAMA_RESTART_POLICY           Embedder container used by the 'ollama' and
                                  'start' actions. Defaults: ollama-memos,
                                  ollama/ollama:latest, GPU 0, 11434,
                                  ollama_data, 2, Ollama default,
                                  unless-stopped.

Examples:
  scripts/setup_memory_backends.sh start
  MEMORY_READER_TAG=7b scripts/setup_memory_backends.sh restart
  scripts/setup_memory_backends.sh restart llama3b
  scripts/setup_memory_backends.sh restart 0_6b llama3b 7b 8b 32b
  scripts/setup_memory_backends.sh restart-all
EOF
}

log() {
  echo "[memory-backends] $*"
}

die() {
  echo "[memory-backends] ERROR: $*" >&2
  exit 2
}

trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "$1 is not installed or not on PATH"
}

select_memory_reader() {
  local requested="${1:-$MEMORY_READER_TAG}"
  local spec tag model endpoint
  local found=0

  for spec in "${MEMORY_READER_SPECS[@]}"; do
    IFS='|' read -r tag model endpoint <<<"$spec"
    if [[ "$tag" == "$requested" ]]; then
      MEMORY_READER_TAG="$tag"
      if [[ -z "$MEMORY_LLM_MODEL" ]]; then
        MEMORY_LLM_MODEL="$model"
      fi
      if [[ -z "$MEMORY_LLM_BASE_URL" ]]; then
        MEMORY_LLM_BASE_URL="$endpoint"
      fi
      found=1
      break
    fi
  done

  if [[ "$found" -ne 1 ]]; then
    die "unknown reader tag for memory extractor: $requested (valid: 0_6b, llama3b, 7b, 8b, 32b)"
  fi
}

set_all_stack_ports() {
  local tag="$1"
  case "$tag" in
    0_6b)
      ALL_MEMOBASE_PORT=18100
      ALL_MEMOS_PORT=18101
      ALL_MEMOBASE_DB_PORT=19100
      ALL_MEMOBASE_REDIS_PORT=19101
      ;;
    llama3b)
      ALL_MEMOBASE_PORT=18102
      ALL_MEMOS_PORT=18103
      ALL_MEMOBASE_DB_PORT=19102
      ALL_MEMOBASE_REDIS_PORT=19103
      ;;
    7b)
      ALL_MEMOBASE_PORT=18104
      ALL_MEMOS_PORT=18105
      ALL_MEMOBASE_DB_PORT=19104
      ALL_MEMOBASE_REDIS_PORT=19105
      ;;
    8b)
      ALL_MEMOBASE_PORT=18106
      ALL_MEMOS_PORT=18107
      ALL_MEMOBASE_DB_PORT=19106
      ALL_MEMOBASE_REDIS_PORT=19107
      ;;
    32b)
      ALL_MEMOBASE_PORT=18108
      ALL_MEMOS_PORT=18109
      ALL_MEMOBASE_DB_PORT=19108
      ALL_MEMOBASE_REDIS_PORT=19109
      ;;
    *)
      die "unknown reader tag for all-stack ports: $tag"
      ;;
  esac
}

REQUESTED_READER_TAGS=()

expand_reader_tags() {
  local raw
  local part
  local -a pieces
  REQUESTED_READER_TAGS=()
  for raw in "$@"; do
    IFS=',' read -r -a pieces <<<"$raw"
    for part in "${pieces[@]}"; do
      part="$(trim "$part")"
      [[ -n "$part" ]] && REQUESTED_READER_TAGS+=("$part")
    done
  done
}

detect_docker_compose() {
  require_cmd docker
  if docker compose version >/dev/null 2>&1; then
    DOCKER_COMPOSE=(docker compose)
  elif command -v docker-compose >/dev/null 2>&1; then
    DOCKER_COMPOSE=(docker-compose)
  else
    die "docker compose is not available"
  fi
}

configure_compose_up_flags() {
  local build_mode
  build_mode="$(printf '%s' "$BUILD_IMAGES" | tr '[:upper:]' '[:lower:]')"
  COMPOSE_UP_FLAGS=(-d)
  case "$build_mode" in
    1|true|yes|always)
      COMPOSE_UP_FLAGS+=(--build)
      ;;
    0|false|no|auto|"")
      ;;
    never)
      COMPOSE_UP_FLAGS+=(--no-build)
      ;;
    *)
      die "unsupported MEMORY_BACKENDS_BUILD=$BUILD_IMAGES (use 0, 1, or never)"
      ;;
  esac
}

run_compose() {
  local project="$1"
  shift
  if [[ -n "$project" ]]; then
    COMPOSE_PROJECT_NAME="$project" "$@"
  else
    "$@"
  fi
}

run_memobase_compose() {
  local project="$1"
  shift
  (
    export DATABASE_NAME="memobase"
    export DATABASE_USER="memobase"
    export DATABASE_PASSWORD="memobase"
    export DATABASE_LOCATION="./db/data"
    export REDIS_PASSWORD="memobase"
    export REDIS_LOCATION="./db/redis/data"
    export DATABASE_EXPORT_PORT="$MEMOBASE_DB_PORT"
    export REDIS_EXPORT_PORT="$MEMOBASE_REDIS_PORT"
    export API_EXPORT_PORT="$MEMOBASE_PORT"
    export API_HOSTS="http://0.0.0.0:${MEMOBASE_PORT},http://localhost:${MEMOBASE_PORT}"
    export USE_CORS="false"
    export PROJECT_ID="memobase_dev"
    export ACCESS_TOKEN="$MEMOBASE_API_TOKEN"
    export TIKTOKEN_CACHE_DIR_HOST="${TIKTOKEN_CACHE_DIR_HOST:-$HOME/.cache/tiktoken}"
    run_compose "$project" "$@"
  )
}

ensure_clean_or_allowed() {
  local dir="$1"
  if [[ "$ALLOW_DIRTY" == "1" ]]; then
    return 0
  fi
  if [[ -n "$(git -C "$dir" status --porcelain)" ]]; then
    die "refusing to change dirty checkout: $dir (set MEMORY_BACKENDS_ALLOW_DIRTY=1 to override)"
  fi
}

checkout_repo() {
  local dir="$1"
  local repo="$2"
  local ref="$3"
  local current
  local target
  local stale_dir

  require_cmd git
  if [[ ! -d "$dir/.git" ]]; then
    mkdir -p "$(dirname "$dir")"
    if [[ -e "$dir" ]]; then
      stale_dir="${dir}.stale.$(date +%Y%m%d%H%M%S).$$"
      log "moving non-git checkout out of the way: $dir -> $stale_dir"
      mv "$dir" "$stale_dir"
    fi
    log "cloning $repo -> $dir"
    git clone --branch "$ref" --depth 1 "$repo" "$dir"
  fi

  if [[ "$UPDATE_BACKENDS" == "1" ]]; then
    log "updating pinned refs in $dir"
    git -C "$dir" fetch --tags origin
  elif git -C "$dir" rev-parse "${ref}^{commit}" >/dev/null 2>&1; then
    log "using cached repo checkout: $dir"
  else
    log "local ref $ref not found in $dir; fetching once"
    git -C "$dir" fetch --tags origin
  fi

  current="$(git -C "$dir" rev-parse HEAD)"
  target="$(git -C "$dir" rev-parse "${ref}^{commit}")"
  if [[ "$current" == "$target" ]]; then
    log "already at $ref: $dir"
    return 0
  fi

  ensure_clean_or_allowed "$dir"
  log "checking out $dir @ $ref"
  git -C "$dir" checkout -q "$ref"
}

set_env_kv() {
  local file="$1"
  local key="$2"
  local value="$3"
  local tmp="${file}.tmp.$$"

  if [[ -f "$file" ]] && grep -q "^${key}=" "$file"; then
    awk -v key="$key" -v value="$value" '
      BEGIN { prefix = key "=" }
      index($0, prefix) == 1 { $0 = prefix value }
      { print }
    ' "$file" >"$tmp"
  else
    if [[ -f "$file" ]]; then
      cp "$file" "$tmp"
    else
      : >"$tmp"
    fi
    printf '%s=%s\n' "$key" "$value" >>"$tmp"
  fi
  mv "$tmp" "$file"
}

write_memobase_config() {
  local config="$MEMOBASE_DIR/src/server/api/config.yaml"
  if [[ -f "$config" && "$OVERWRITE_MEMORY_CONFIG" != "1" ]]; then
    log "keeping existing Memobase config: $config"
    return 0
  fi

  log "writing Memobase config: $config"
  cat >"$config" <<EOF
llm_api_key: ${MEMORY_LLM_API_KEY}
llm_base_url: ${MEMORY_LLM_BASE_URL}
best_llm_model: ${MEMORY_LLM_MODEL}
llm_style: openai
enable_event_embedding: true
embedding_provider: ollama
embedding_base_url: ${OLLAMA_API_BASE}
embedding_api_key: EMPTY
embedding_model: ${MEMOS_EMBEDDER_MODEL}
embedding_dim: ${MEMOS_EMBEDDING_DIMENSION}
profile_validate_mode: true
profile_strict_mode: true
persistent_chat_blobs: true
max_chat_blob_buffer_token_size: 4096
max_pre_profile_token_size: 512
language: en
EOF
}

patch_memobase_compose() {
  local compose="$MEMOBASE_DIR/src/server/docker-compose.yml"
  [[ -f "$compose" ]] || die "Memobase docker compose file not found: $compose"

  if grep -qE '^[[:space:]]+platform:[[:space:]]+linux/amd64[[:space:]]*$' "$compose"; then
    log "removing fixed Memobase linux/amd64 platform in $compose"
    perl -0pi -e 's/^[ \t]*platform:\s*linux\/amd64\s*\n//mg' "$compose"
  fi

  if ! grep -q 'TIKTOKEN_CACHE_DIR' "$compose"; then
    log "adding Memobase tiktoken cache mount in $compose"
    perl -0pi -e 's/(  memobase-server-api:\n(?:(?!  [A-Za-z0-9_-]+:).)*?    environment:\n)/$1      - TIKTOKEN_CACHE_DIR=\/root\/.cache\/tiktoken\n/s; s|(      - \./api/config\.yaml:/app/config\.yaml\n)|$1      - \${TIKTOKEN_CACHE_DIR_HOST:-/tmp/tiktoken-cache}:/root/.cache/tiktoken:ro\n|' "$compose"
  fi

  if grep -qE '^(name:|[[:space:]]+container_name:)' "$compose"; then
    log "patching fixed Memobase compose names in $compose"
    perl -0pi -e 's/^name:\s*memobase-server\s*\n//mg; s/^[ \t]*container_name:\s*[^\n]+\n//mg' "$compose"
  fi

  patch_memobase_postgres_connections "$compose"
}

# Give the Memobase Postgres service `command: postgres -c max_connections=N`
# (see MEMOBASE_PG_MAX_CONNECTIONS). Takes effect when the db container is
# (re)created; a running stack keeps its old limit until restarted.
patch_memobase_postgres_connections() {
  local compose="$1"
  if grep -q 'max_connections' "$compose"; then
    return 0
  fi
  log "setting Memobase Postgres max_connections=${MEMOBASE_PG_MAX_CONNECTIONS} in $compose"
  if ! MEMOBASE_PG_MAX_CONNECTIONS="$MEMOBASE_PG_MAX_CONNECTIONS" python3 - "$compose" <<'PY'
import os
import re
import sys

path = sys.argv[1]
n = int(os.environ["MEMOBASE_PG_MAX_CONNECTIONS"])
lines = open(path, encoding="utf-8").read().split("\n")
headers = [i for i, line in enumerate(lines) if re.match(r"^  [A-Za-z0-9_.-]+:\s*$", line)]
for k, start in enumerate(headers):
    end = headers[k + 1] if k + 1 < len(headers) else len(lines)
    block = range(start + 1, end)
    image = next((i for i in block if re.match(r"^\s+image:\s*\S*(postgres|pgvector)", lines[i])), None)
    if image is None:
        continue
    if any(re.match(r"^\s+command:", lines[i]) for i in block):
        sys.exit(f"{path}: the Postgres service already has a command; add -c max_connections={n} by hand")
    indent = re.match(r"^(\s+)", lines[image]).group(1)
    lines.insert(image + 1, f'{indent}command: ["postgres", "-c", "max_connections={n}"]')
    open(path, "w", encoding="utf-8").write("\n".join(lines))
    sys.exit(0)
sys.exit(f"{path}: no Postgres service (image postgres/pgvector) found")
PY
  then
    log "WARNING: could not set Postgres max_connections for Memobase; inserts may fail with errno 500 under load"
  fi
}

configure_memobase() {
  local server_dir="$MEMOBASE_DIR/src/server"
  local env_file="$server_dir/.env"

  [[ -d "$server_dir" ]] || die "Memobase server directory not found: $server_dir"
  patch_memobase_compose
  if [[ ! -f "$env_file" && -f "$server_dir/.env.example" ]]; then
    cp "$server_dir/.env.example" "$env_file"
  fi

  set_env_kv "$env_file" "DATABASE_NAME" "memobase"
  set_env_kv "$env_file" "DATABASE_USER" "memobase"
  set_env_kv "$env_file" "DATABASE_PASSWORD" "memobase"
  set_env_kv "$env_file" "DATABASE_LOCATION" "./db/data"
  set_env_kv "$env_file" "REDIS_PASSWORD" "memobase"
  set_env_kv "$env_file" "REDIS_LOCATION" "./db/redis/data"
  set_env_kv "$env_file" "DATABASE_EXPORT_PORT" "$MEMOBASE_DB_PORT"
  set_env_kv "$env_file" "REDIS_EXPORT_PORT" "$MEMOBASE_REDIS_PORT"
  set_env_kv "$env_file" "API_EXPORT_PORT" "$MEMOBASE_PORT"
  set_env_kv "$env_file" "API_HOSTS" "http://0.0.0.0:${MEMOBASE_PORT},http://localhost:${MEMOBASE_PORT}"
  set_env_kv "$env_file" "USE_CORS" "false"
  set_env_kv "$env_file" "PROJECT_ID" "memobase_dev"
  set_env_kv "$env_file" "ACCESS_TOKEN" "$MEMOBASE_API_TOKEN"

  write_memobase_config
}

patch_memos_compose() {
  local compose="$MEMOS_DIR/docker/docker-compose.yml"
  [[ -f "$compose" ]] || die "MemOS docker compose file not found: $compose"

  if grep -qE '^(name:|[[:space:]]+container_name:)' "$compose"; then
    log "patching fixed MemOS compose names in $compose"
    perl -0pi -e 's/^name:\s*memos-dev\s*\n//mg; s/^[ \t]*container_name:\s*[^\n]+\n//mg' "$compose"
  fi

  if grep -q '"8000:8000"' "$compose"; then
    log "patching MemOS host API port in $compose (container stays on 8000)"
    perl -0pi -e 's/"8000:8000"/"\${MEMOS_EXPORT_PORT:-8020}:8000"/g' "$compose"
  fi

  if grep -q 'qdrant-docker\|neo4j-docker' "$compose"; then
    log "patching MemOS service hostnames in $compose"
    perl -0pi -e 's/qdrant-docker/qdrant/g; s/neo4j-docker/neo4j/g' "$compose"
  fi

  if grep -qE '"7474:7474"|"7687:7687"|"6333:6333"|"6334:6334"' "$compose"; then
    log "removing MemOS internal service host ports in $compose"
    perl -0pi -e 's/\n    ports:\n      - "7474:7474"[^\n]*\n      - "7687:7687"[^\n]*//g; s/\n    ports:\n      - "6333:6333"[^\n]*\n      - "6334:6334"[^\n]*//g' "$compose"
  fi

  if ! grep -q 'host.docker.internal:host-gateway' "$compose"; then
    log "adding host.docker.internal access to $compose"
    perl -0pi -e 's/(    env_file:\n      - \.\.\/\.env\n)/$1    extra_hosts:\n      - "host.docker.internal:host-gateway"\n/' "$compose"
  fi
}

write_memos_env() {
  local env_file="$MEMOS_DIR/.env"
  if [[ -f "$env_file" && "$OVERWRITE_MEMORY_CONFIG" != "1" ]]; then
    log "keeping existing MemOS env: $env_file"
    return 0
  fi

  log "writing MemOS env: $env_file"
  cat >"$env_file" <<EOF
TZ=Asia/Shanghai
MOS_CUBE_PATH=/tmp/data_test
MEMOS_BASE_PATH=.
MOS_ENABLE_DEFAULT_CUBE_CONFIG=true
MOS_ENABLE_REORGANIZE=false
MOS_TEXT_MEM_TYPE=general_text
ASYNC_MODE=sync

MOS_TOP_K=10
MOS_CHAT_MODEL=${MEMORY_LLM_MODEL}
MOS_CHAT_TEMPERATURE=0.1
MOS_MAX_TOKENS=4096
MOS_TOP_P=0.9
MOS_CHAT_MODEL_PROVIDER=openai
OPENAI_API_KEY=${MEMORY_LLM_API_KEY}
OPENAI_API_BASE=${MEMORY_LLM_BASE_URL}

MEMRADER_MODEL=${MEMORY_LLM_MODEL}
MEMRADER_API_KEY=${MEMORY_LLM_API_KEY}
MEMRADER_API_BASE=${MEMORY_LLM_BASE_URL}
MEMRADER_MAX_TOKENS=4096

EMBEDDING_DIMENSION=${MEMOS_EMBEDDING_DIMENSION}
MOS_EMBEDDER_BACKEND=ollama
MOS_EMBEDDER_MODEL=${MEMOS_EMBEDDER_MODEL}
OLLAMA_API_BASE=${OLLAMA_API_BASE}

MOS_RERANKER_BACKEND=cosine_local
ENABLE_INTERNET=false
ENABLE_PREFERENCE_MEMORY=false

MEM_READER_BACKEND=simple_struct
MEM_READER_CHAT_CHUNK_TYPE=default
MEM_READER_CHAT_CHUNK_TOKEN_SIZE=1600
MEM_READER_CHAT_CHUNK_SESS_SIZE=10
MEM_READER_CHAT_CHUNK_OVERLAP=2

MOS_ENABLE_SCHEDULER=false
API_SCHEDULER_ON=false

NEO4J_BACKEND=neo4j-community
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=12345678
NEO4J_DB_NAME=neo4j
MOS_NEO4J_SHARED_DB=false
QDRANT_HOST=qdrant
QDRANT_PORT=6333

CHAT_MODEL_LIST=[{"backend":"openai","api_base":"${MEMORY_LLM_BASE_URL}","api_key":"${MEMORY_LLM_API_KEY}","model_name_or_path":"${MEMORY_LLM_MODEL}","support_models":["${MEMORY_LLM_MODEL}"]}]
EOF
}

configure_memos() {
  [[ -d "$MEMOS_DIR/docker" ]] || die "MemOS docker directory not found: $MEMOS_DIR/docker"
  patch_memos_compose
  write_memos_env
}

# Bring up the shared Ollama embedder used by both Memobase and MemOS.
# - Memobase server's embedding_provider=ollama hits this for event embeddings
# - MemOS server's MOS_EMBEDDER_BACKEND=ollama hits this for memory store
# Without this, Memobase add returns errno 503 on event embedding sanity check
# and MemOS add returns 200 with data:[] (silently persists nothing).
ensure_ollama() {
  require_cmd docker
  local image="${OLLAMA_IMAGE:-ollama/ollama:latest}"
  local name="${OLLAMA_CONTAINER_NAME:-ollama-memos}"
  local volume="${OLLAMA_VOLUME:-ollama_data}"
  local port="${OLLAMA_HOST_PORT:-11434}"
  local gpu="${OLLAMA_GPU:-0}"
  local parallel="${OLLAMA_NUM_PARALLEL:-2}"
  local model="${MEMOS_EMBEDDER_MODEL:-nomic-embed-text:latest}"
  local ready_timeout="${OLLAMA_READY_TIMEOUT:-60}"
  local pull_timeout="${OLLAMA_PULL_TIMEOUT:-600}"
  local restart_policy="${OLLAMA_RESTART_POLICY:-unless-stopped}"
  local -a keep_alive_env=()
  if [[ -n "${OLLAMA_KEEP_ALIVE:-}" ]]; then
    keep_alive_env=(-e "OLLAMA_KEEP_ALIVE=${OLLAMA_KEEP_ALIVE}")
  fi

  local state
  state="$(docker inspect -f '{{.State.Status}}' "$name" 2>/dev/null || true)"
  case "$state" in
    running) log "ollama: $name already running" ;;
    "")
      log "ollama: starting $name (image=$image, gpu=$gpu, port=$port, parallel=$parallel)"
      docker image inspect "$image" >/dev/null 2>&1 || docker pull "$image" >/dev/null
      docker run -d --name "$name" --restart "$restart_policy" \
        --gpus "\"device=${gpu}\"" \
        -e OLLAMA_NUM_PARALLEL="$parallel" \
        ${keep_alive_env[@]+"${keep_alive_env[@]}"} \
        -p "${port}:11434" \
        -v "${volume}:/root/.ollama" \
        "$image" >/dev/null
      ;;
    *) log "ollama: $name in state $state, starting"; docker start "$name" >/dev/null ;;
  esac

  local deadline=$((SECONDS + ready_timeout))
  while ! curl -fsS "http://127.0.0.1:${port}/" >/dev/null 2>&1; do
    [[ "$SECONDS" -ge "$deadline" ]] && die "ollama: timeout waiting for $name on :$port"
    sleep 1
  done
  log "ollama: ready on http://localhost:${port}"

  if docker exec "$name" ollama list 2>/dev/null | awk 'NR>1 {print $1}' | grep -qx "$model"; then
    log "ollama: model $model present"
  else
    log "ollama: pulling $model (timeout ${pull_timeout}s)"
    timeout "$pull_timeout" docker exec "$name" ollama pull "$model" || die "ollama: pull $model failed"
  fi
}

stop_ollama() {
  require_cmd docker
  local name="${OLLAMA_CONTAINER_NAME:-ollama-memos}"
  if [[ "$(docker inspect -f '{{.State.Status}}' "$name" 2>/dev/null || true)" == "running" ]]; then
    log "ollama: stopping $name"; docker stop "$name" >/dev/null
  fi
}

setup_all() {
  local reader_tag="${1:-}"
  select_memory_reader "$reader_tag"
  if [[ -n "$reader_tag" ]]; then
    OVERWRITE_MEMORY_CONFIG=1
  fi
  checkout_repo "$MEMOBASE_DIR" "$MEMOBASE_REPO" "$MEMOBASE_REF"
  if [[ "$SKIP_MEMOS" != "1" ]]; then
    checkout_repo "$MEMOS_DIR" "$MEMOS_REPO" "$MEMOS_REF"
  fi
  configure_memobase
  if [[ "$SKIP_MEMOS" != "1" ]]; then
    configure_memos
  fi
  log "setup complete"
  print_env
}

start_all() {
  local reader_tag="${1:-}"
  setup_all "$reader_tag"
  detect_docker_compose
  configure_compose_up_flags

  ensure_ollama

  log "memory extractor: ${MEMORY_READER_TAG} -> ${MEMORY_LLM_MODEL} @ ${MEMORY_LLM_BASE_URL}"
  log "starting Memobase on http://localhost:${MEMOBASE_PORT} (compose project: ${MEMOBASE_COMPOSE_PROJECT:-default})"
  (cd "$MEMOBASE_DIR/src/server" && run_memobase_compose "$MEMOBASE_COMPOSE_PROJECT" "${DOCKER_COMPOSE[@]}" up "${COMPOSE_UP_FLAGS[@]}")

  if [[ "$SKIP_MEMOS" != "1" ]]; then
    log "starting MemOS on http://localhost:${MEMOS_PORT} (compose project: ${MEMOS_COMPOSE_PROJECT:-default})"
    (cd "$MEMOS_DIR/docker" && export MEMOS_EXPORT_PORT="$MEMOS_PORT" && run_compose "$MEMOS_COMPOSE_PROJECT" "${DOCKER_COMPOSE[@]}" up "${COMPOSE_UP_FLAGS[@]}")
  fi

  log "services requested"
  print_env
}

restart_all() {
  local reader_tag="${1:-}"
  stop_all
  OVERWRITE_MEMORY_CONFIG=1 start_all "$reader_tag"
}

stop_all() {
  detect_docker_compose
  if [[ -d "$MEMOBASE_DIR/src/server" ]]; then
    log "stopping Memobase (compose project: ${MEMOBASE_COMPOSE_PROJECT:-default})"
    (cd "$MEMOBASE_DIR/src/server" && run_memobase_compose "$MEMOBASE_COMPOSE_PROJECT" "${DOCKER_COMPOSE[@]}" down)
  fi
  if [[ "$SKIP_MEMOS" != "1" && -d "$MEMOS_DIR/docker" ]]; then
    log "stopping MemOS (compose project: ${MEMOS_COMPOSE_PROJECT:-default})"
    (cd "$MEMOS_DIR/docker" && export MEMOS_EXPORT_PORT="$MEMOS_PORT" && run_compose "$MEMOS_COMPOSE_PROJECT" "${DOCKER_COMPOSE[@]}" down)
  fi
}

status_all() {
  require_cmd docker
  docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Ports}}\t{{.Status}}' \
    | grep -Ei 'memobase|memos|qdrant|neo4j|postgres|redis' || true
}

print_env() {
  printf 'export MEMOBASE_BASE_URL=%q\n' "http://localhost:${MEMOBASE_PORT}"
  printf 'export MEMOBASE_API_TOKEN=%q\n' "$MEMOBASE_API_TOKEN"
  printf 'export MEMOS_BASE_URL=%q\n' "http://localhost:${MEMOS_PORT}"
  printf 'export MEMOS_API_KEY=%q\n' "$MEMOS_API_KEY"
}

print_versions() {
  local reader_tag="${1:-}"
  select_memory_reader "$reader_tag"
  cat <<EOF
Memobase repo: ${MEMOBASE_REPO}
Memobase ref:  ${MEMOBASE_REF}
Memobase dir:  ${MEMOBASE_DIR}
Memobase port: ${MEMOBASE_PORT}
Memobase compose project: ${MEMOBASE_COMPOSE_PROJECT:-default}

MemOS repo:    ${MEMOS_REPO}
MemOS ref:     ${MEMOS_REF}
MemOS dir:     ${MEMOS_DIR}
MemOS port:    ${MEMOS_PORT}
MemOS compose project: ${MEMOS_COMPOSE_PROJECT:-default}

Memory extractor reader tag: ${MEMORY_READER_TAG}
Memory extractor model:      ${MEMORY_LLM_MODEL}
Memory extractor endpoint:   ${MEMORY_LLM_BASE_URL}
Reuse local repo cache:       $([[ "$UPDATE_BACKENDS" == "1" ]] && printf 'no' || printf 'yes')
Force docker compose build:   $([[ "$BUILD_IMAGES" == "1" ]] && printf 'yes' || printf 'no')
EOF
}

run_isolated_reader_stacks() {
  local action="$1"
  shift
  local tag
  local script="$SCRIPT_DIR/setup_memory_backends.sh"
  local -a pids=()
  local -a labels=()
  local i failed=0
  local parallel=0

  if [[ "$#" -eq 0 ]]; then
    set -- "${MEMORY_READER_TAGS[@]}"
  fi

  case "$action" in
    setup|start|restart|stop)
      [[ "$PARALLEL_ISOLATED" != "0" ]] && parallel=1
      ;;
  esac

  for tag in "$@"; do
    set_all_stack_ports "$tag"
    log "[$tag] ${action} isolated stack: Memobase http://localhost:${ALL_MEMOBASE_PORT}, MemOS http://localhost:${ALL_MEMOS_PORT}"
    if [[ "$parallel" == "1" ]]; then
      (
        MEMARENA_MEMORY_SERVICES_DIR="${ALL_SERVICES_ROOT}/${tag}" \
        MEMORY_COMPOSE_PROJECT_PREFIX="memarena_${tag}" \
        MEMOBASE_COMPOSE_PROJECT= \
        MEMOS_COMPOSE_PROJECT= \
        MEMOBASE_DIR= \
        MEMOS_DIR= \
        MEMORY_LLM_BASE_URL= \
        MEMORY_LLM_MODEL= \
        MEMOBASE_PORT="$ALL_MEMOBASE_PORT" \
        MEMOBASE_DB_PORT="$ALL_MEMOBASE_DB_PORT" \
        MEMOBASE_REDIS_PORT="$ALL_MEMOBASE_REDIS_PORT" \
        MEMOS_PORT="$ALL_MEMOS_PORT" \
        MEMORY_BACKENDS_ISOLATED_CHILD=1 \
        "$script" "$action" "$tag"
      ) &
      pids+=("$!")
      labels+=("$tag")
    else
      MEMARENA_MEMORY_SERVICES_DIR="${ALL_SERVICES_ROOT}/${tag}" \
      MEMORY_COMPOSE_PROJECT_PREFIX="memarena_${tag}" \
      MEMOBASE_COMPOSE_PROJECT= \
      MEMOS_COMPOSE_PROJECT= \
      MEMOBASE_DIR= \
      MEMOS_DIR= \
      MEMORY_LLM_BASE_URL= \
      MEMORY_LLM_MODEL= \
      MEMOBASE_PORT="$ALL_MEMOBASE_PORT" \
      MEMOBASE_DB_PORT="$ALL_MEMOBASE_DB_PORT" \
      MEMOBASE_REDIS_PORT="$ALL_MEMOBASE_REDIS_PORT" \
      MEMOS_PORT="$ALL_MEMOS_PORT" \
      MEMORY_BACKENDS_ISOLATED_CHILD=1 \
      "$script" "$action" "$tag"
    fi
  done

  if [[ "$parallel" == "1" ]]; then
    for i in "${!pids[@]}"; do
      if ! wait "${pids[$i]}"; then
        log "ERROR: [${labels[$i]}] ${action} failed"
        failed=1
      fi
    done
    return "$failed"
  fi
}

action="${1:-setup}"
expand_reader_tags "${@:2}"
case "$action" in
  setup)
    if [[ "$ISOLATED_CHILD" != "1" && "${#REQUESTED_READER_TAGS[@]}" -gt 0 ]]; then
      run_isolated_reader_stacks setup "${REQUESTED_READER_TAGS[@]}"
    else
      setup_all "${REQUESTED_READER_TAGS[0]:-}"
    fi
    ;;
  start)
    if [[ "$ISOLATED_CHILD" != "1" && "${#REQUESTED_READER_TAGS[@]}" -gt 0 ]]; then
      run_isolated_reader_stacks start "${REQUESTED_READER_TAGS[@]}"
    else
      start_all "${REQUESTED_READER_TAGS[0]:-}"
    fi
    ;;
  restart)
    if [[ "$ISOLATED_CHILD" != "1" && "${#REQUESTED_READER_TAGS[@]}" -gt 0 ]]; then
      run_isolated_reader_stacks restart "${REQUESTED_READER_TAGS[@]}"
    else
      restart_all "${REQUESTED_READER_TAGS[0]:-}"
    fi
    ;;
  setup-all)
    run_isolated_reader_stacks setup
    ;;
  start-all)
    run_isolated_reader_stacks start
    ;;
  restart-all)
    run_isolated_reader_stacks restart
    ;;
  stop-all)
    run_isolated_reader_stacks stop
    ;;
  stop)
    if [[ "$ISOLATED_CHILD" != "1" && "${#REQUESTED_READER_TAGS[@]}" -gt 0 ]]; then
      run_isolated_reader_stacks stop "${REQUESTED_READER_TAGS[@]}"
    else
      stop_all
    fi
    ;;
  status)
    status_all
    ;;
  ollama)
    ensure_ollama
    ;;
  ollama-stop)
    stop_ollama
    ;;
  env)
    if [[ "$ISOLATED_CHILD" != "1" && "${#REQUESTED_READER_TAGS[@]}" -gt 0 ]]; then
      run_isolated_reader_stacks env "${REQUESTED_READER_TAGS[@]}"
    else
      print_env
    fi
    ;;
  versions)
    if [[ "$ISOLATED_CHILD" != "1" && "${#REQUESTED_READER_TAGS[@]}" -gt 0 ]]; then
      run_isolated_reader_stacks versions "${REQUESTED_READER_TAGS[@]}"
    else
      print_versions "${REQUESTED_READER_TAGS[0]:-}"
    fi
    ;;
  versions-all)
    run_isolated_reader_stacks versions
    ;;
  help|--help|-h)
    usage
    ;;
  *)
    die "unknown action: $action"
    ;;
esac
