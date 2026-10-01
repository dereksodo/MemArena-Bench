#!/usr/bin/env bash
# Create/update the MemArena repository-local Python virtual environment.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "[setup-venv] creating venv: $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

echo "[setup-venv] upgrading pip/setuptools/wheel"
"$VENV_DIR/bin/python" -m pip install --upgrade pip wheel "setuptools>=68,<81"

echo "[setup-venv] installing MemArena editable dev environment"
(cd "$REPO_ROOT" && "$VENV_DIR/bin/python" -m pip install -e ".[dev,memsearch]")

echo
echo "[setup-venv] ready"
echo "  source \"$VENV_DIR/bin/activate\""
echo "  python scripts/verify_install.py"
