#!/bin/bash
set -euo pipefail
W=$(cd "$(dirname "$0")" && pwd)
source "$W/runtime_env.sh"
: "${SLURM_JOB_ID:?Run builds only inside a Slurm allocation}" "${CORPUS_DIR:?}"
test "${SLURM_JOB_PARTITION:-${BUILD_PARTITION:?}}" = "${BUILD_PARTITION:?}"
OUT="${1:-$W/rebuilt-index-$(date +%Y%m%d-%H%M%S)}"
test ! -e "$OUT" || { echo 'Refusing to overwrite an existing index'; exit 2; }

: "${SCRATCH_DIR:?}" "${LLM_BASE_URL:?}"
mkdir -p "$SCRATCH_DIR"
export BGE_SOCK="/tmp/arb-bge-build-$$.sock"
PYTHONHOME="$W/bge-python" PYTHONPATH="$W/code:$W/bge-venv/lib/python3.12/site-packages" "$W/bge-python/bin/python3.12" "$W/code/bge_server.py" --model "$W/models/bge-m3" --sock "$BGE_SOCK" --device cuda > "$SCRATCH_DIR/bge-build.log" 2>&1 &
bge_pid=$!
trap 'kill "$bge_pid" 2>/dev/null || true' EXIT
ready=0
for ((i=0;i<180;i++)); do
  if grep -q '^BGE_READY' "$SCRATCH_DIR/bge-build.log"; then ready=1; break; fi
  kill -0 "$bge_pid" 2>/dev/null || exit 3
  sleep 5
done
test "$ready" = 1
"$PYTHON" "$W/code/rebuild_index.py" "$OUT"
