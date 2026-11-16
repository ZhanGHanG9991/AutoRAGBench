#!/bin/bash
set -euo pipefail
W=$(cd "$(dirname "$0")" && pwd)
source "$W/runtime_env.sh"
: "${SLURM_JOB_ID:?Run builds only inside a Slurm allocation}" "${CORPUS_DIR:?}"
test "${SLURM_JOB_PARTITION:-${BUILD_PARTITION:?}}" = "${BUILD_PARTITION:?}"
OUT="${1:-$W/rebuilt-index-$(date +%Y%m%d-%H%M%S)}"
test ! -e "$OUT" || { echo 'Refusing to overwrite an existing index'; exit 2; }
exec "$PYTHON" "$W/code/rebuild_index.py" "$OUT"
