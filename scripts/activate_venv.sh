#!/usr/bin/env bash
# Source this file to activate MemArena's repository-local virtualenv.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Run this with source so it can update your current shell:"
  echo "  source scripts/activate_venv.sh"
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "[activate-venv] missing venv: $VENV_DIR"
  echo "[activate-venv] create it first: scripts/setup_venv.sh"
  return 2
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
echo "[activate-venv] activated $VIRTUAL_ENV"
