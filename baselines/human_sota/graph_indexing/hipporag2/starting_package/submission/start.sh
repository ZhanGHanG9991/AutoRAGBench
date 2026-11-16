#!/bin/bash
set -euo pipefail
W=$(cd "$(dirname "$0")/.." && pwd)
source "$W/runtime_env.sh"
: "${PORT:?}" "${CORPUS_DIR:?}" "${SCRATCH_DIR:?}" "${LLM_BASE_URL:?}"
mkdir -p "$SCRATCH_DIR"
exec "$PYTHON" "$W/adapter/server.py"
