#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper.
# Preferred entrypoint is now Python project:
#   python3 scripts/generate_dataset.py [target_size] [output_dir] [--signal-ratio ...]

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "${ROOT_DIR}/scripts/generate_dataset.py" "$@"
