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
export PYTHONPATH="$WS/code:$WS/upstream"
BGE_SOCK="$SCRATCH_DIR/bge-${PORT}.sock"
cd "$SCRATCH_DIR"
"$WS/bge-venv/bin/python" "$WS/code/bge_server.py" --model "$WS/models/bge-m3" --sock "$BGE_SOCK" --device cuda > "$SCRATCH_DIR/bge.log" 2>&1 &
bge_pid=$!
ready=0
for ((i=0;i<180;i++)); do
  if grep -q "^BGE_READY" "$SCRATCH_DIR/bge.log"; then ready=1; break; fi
  test -d "/proc/$bge_pid" || exit 3
  sleep 5
done
test "$ready" = 1
exec "$WS/.venv/bin/python" "$WS/submission/tokenizer_launcher.py" "$WS/code/query_server.py" --storage "$WS/index/storage" --bge-sock "$BGE_SOCK" --port "$PORT" --llm "$LLM_BASE_URL" --task-info "$WS/tasks/multihop_rag/info.json"
