#!/bin/bash
set -euo pipefail
W=$(cd "$(dirname "$0")/.." && pwd)
source "$W/runtime_env.sh"
: "${PORT:?}" "${CORPUS_DIR:?}" "${SCRATCH_DIR:?}" "${LLM_BASE_URL:?}"
mkdir -p "$SCRATCH_DIR/tmp"
export TMPDIR="$SCRATCH_DIR/tmp" HF_HOME="$SCRATCH_DIR/hf" TORCH_HOME="$SCRATCH_DIR/torch"
cd "$W"
exec "$PYTHON" "$W/submission/server.py"
