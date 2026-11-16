#!/bin/bash
set -euo pipefail
W=$(cd "${1:?workspace}" && pwd)
TASK=${2:?task}
OUT=${3:?new output index directory}
: "${CORPUS_DIR:?}" "${LLM_BASE_URL:?}"
export PYTHONPATH="$W/upstream:$W/adapter"
export TIKTOKEN_CACHE_DIR="$W/tiktoken_cache"
export NUMBA_CACHE_DIR="${SCRATCH_DIR:?}/numba"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMBA_NUM_THREADS=4
test ! -e "$OUT"
if [ "$TASK" = qasper ]; then
  exec "$W/venv/bin/python" "$W/build_sources/build.py" --task "$TASK" \
    --corpus "$CORPUS_DIR/corpus.jsonl" --out "$OUT" --models "$W/models" --config "$W/config.json"
fi
"$W/venv/bin/python" "$W/build_sources/build_collection_sharded.py" --phase trees \
  --task "$TASK" --shard-id 0 --shard-count 1 --corpus "$CORPUS_DIR/corpus.jsonl" \
  --out "$OUT" --models "$W/models" --config "$W/config.json"
exec "$W/venv/bin/python" "$W/build_sources/build_collection_sharded.py" --phase assemble \
  --task "$TASK" --shard-count 1 --corpus "$CORPUS_DIR/corpus.jsonl" \
  --out "$OUT" --models "$W/models" --config "$W/config.json"
