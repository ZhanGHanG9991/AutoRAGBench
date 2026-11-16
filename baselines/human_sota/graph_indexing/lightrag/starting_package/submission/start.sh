#!/bin/bash
set -euo pipefail
W=$(cd "$(dirname "$0")/.." && pwd)
source "$W/runtime_env.sh"
: "${PORT:?}" "${CORPUS_DIR:?}" "${SCRATCH_DIR:?}" "${LLM_BASE_URL:?}"
mkdir -p "$SCRATCH_DIR/tmp" "$SCRATCH_DIR/hf"
export TMPDIR="$SCRATCH_DIR/tmp" HF_HOME="$SCRATCH_DIR/hf"
export BGE_SOCK="/tmp/arb-bge-${PORT}.sock"
PYTHONHOME="$W/bge-python" PYTHONPATH="$W/code:$W/bge-venv/lib/python3.12/site-packages" "$W/bge-python/bin/python3.12" "$W/code/bge_server.py" --model "$W/models/bge-m3" --sock "$BGE_SOCK" --device cuda > "$SCRATCH_DIR/bge.log" 2>&1 &
bge_pid=$!
cleanup(){ kill "$bge_pid" 2>/dev/null || true; }
trap cleanup EXIT
ready=0
for ((i=0;i<180;i++)); do
  if grep -q '^BGE_READY' "$SCRATCH_DIR/bge.log"; then ready=1; break; fi
  kill -0 "$bge_pid" 2>/dev/null || exit 3
  sleep 5
done
test "$ready" = 1
cd "$SCRATCH_DIR"
"$PYTHON" "$W/submission/tokenizer_launcher.py" "$W/code/query_server.py" --storage "$W/index/storage" --bge-sock "$BGE_SOCK" --port "$PORT" --llm "$LLM_BASE_URL" --task-info "$W/tasks/multihop_rag/info.json"
