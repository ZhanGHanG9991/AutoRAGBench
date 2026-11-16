#!/bin/bash
set -euo pipefail
W=$(cd "$(dirname "$0")/.." && pwd)
source "$W/runtime_env.sh"
: "${PORT:?}" "${CORPUS_DIR:?}" "${SCRATCH_DIR:?}" "${LLM_BASE_URL:?}"
mkdir -p "$SCRATCH_DIR/tmp" "$SCRATCH_DIR/hf"
export TMPDIR="$SCRATCH_DIR/tmp" HF_HOME="$SCRATCH_DIR/hf"
export SGRAG_TASK=webqsp SGRAG_ARTIFACTS="$W/artifacts" SGRAG_ENCODER="$W/models/encoder"
export SGRAG_K=800 SGRAG_EXACT_K=1 SGRAG_SCHEME=formalv1-train1800 SGRAG_N_TRAIN=1800
export SGRAG_PYTHON="$PYTHON" RAGENG_ROOT="$W"
cd "$SCRATCH_DIR"
exec "$PYTHON" "$W/submission/tokenizer_launcher.py" "$W/code/server.py"
