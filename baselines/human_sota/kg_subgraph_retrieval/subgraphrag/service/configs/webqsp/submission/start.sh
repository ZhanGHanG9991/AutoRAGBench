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
export SGRAG_TASK=webqsp
export SGRAG_ARTIFACTS="$WS/artifacts"
export SGRAG_ENCODER="$WS/models/encoder"
export SGRAG_K=800
export SGRAG_EXACT_K=1
export SGRAG_SCHEME=formalv1-train1800
export SGRAG_N_TRAIN=1800
export SGRAG_PYTHON="$WS/.venv/bin/python"
export RAGENG_ROOT="$WS"
export PYTHONPATH="$WS/code"
cd "$SCRATCH_DIR"
exec "$WS/.venv/bin/python" "$WS/submission/tokenizer_launcher.py" "$WS/code/server.py"
