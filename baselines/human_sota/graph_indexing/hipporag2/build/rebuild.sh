#!/bin/bash
set -euo pipefail
W=$(cd "${1:?workspace}" && pwd)
TASK=${2:?task}
OUT=${3:?new output index directory}
: "${CORPUS_DIR:?}" "${LLM_BASE_URL:?}"
export PYTHONPATH="$W/adapter"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
test ! -e "$OUT"
BUILDER="$W/build_sources/legacy/build.py"
if [ "$TASK" = cwq ]; then BUILDER="$W/build_sources/build.py"; fi
exec "$W/venv/bin/python" "$BUILDER" --task "$TASK" --corpus "$CORPUS_DIR/corpus.jsonl" \
  --out "$OUT" --models "$W/models/bge-base-en-v1.5" --llm "$LLM_BASE_URL" --data-version formal-v2
