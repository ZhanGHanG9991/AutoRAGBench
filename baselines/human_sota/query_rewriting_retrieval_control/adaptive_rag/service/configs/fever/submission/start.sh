#!/bin/bash
set -euo pipefail
WS=$(cd "$(dirname "$0")/.." && pwd)
: "${LLM_BASE_URL:?}" "${SCRATCH_DIR:?}" "${CORPUS_DIR:?}" "${PORT:?}"
mkdir -p "$SCRATCH_DIR/tmp" "$SCRATCH_DIR/hf"
export TMPDIR="$SCRATCH_DIR/tmp" HF_HOME="$SCRATCH_DIR/hf"
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHON_DOTENV_DISABLED=1
export ARAG_ROOT="$WS"
export RAGENG_ROOT="$WS"
export ARAG_TASK=fever
export ARAG_INDEX="$WS/index"
export ARAG_CLASSIFIER="$WS/models/classifier"
export PYTHONPATH="$WS/code"
export ARAG_TRACE="$SCRATCH_DIR/arag-trace.jsonl"
cd "$SCRATCH_DIR"
exec "$WS/.venv/bin/python" "$WS/code/server.py"
