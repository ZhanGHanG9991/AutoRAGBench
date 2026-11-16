#!/bin/bash
set -euo pipefail
W=$(cd "${1:?workspace}" && pwd)
TASK=${2:?task}
OUT=${3:?new output index directory}
: "${CORPUS_DIR:?}"
export PYTHONPATH="$W/adapter"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
test ! -e "$OUT"
exec "$W/.venv/bin/python" "$W/build_sources/build_formal_v2.py" \
  --task "$TASK" --corpus "$CORPUS_DIR/corpus.jsonl" --out "$OUT" --models "$W/models"
