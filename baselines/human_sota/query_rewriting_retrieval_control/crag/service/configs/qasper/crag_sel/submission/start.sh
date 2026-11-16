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
export CRAG_ROOT="$WS"
export CRAG_SCRIPTS="$WS/upstream/scripts"
export PYTHONPATH="$WS/upstream/scripts"
export CRAG_CONFIG="$WS/config.json"
export CRAG_CONTRIEVER_INDEX="$WS/index"
export CRAG_EVALUATOR="$WS/models/evaluator"
export CRAG_CONTRIEVER="$WS/models/contriever-msmarco"
export CRAG_FIRST_RETRIEVER=contriever
export CRAG_EVALUATOR_KIND=t5
export CRAG_REQUIRE_THRESHOLD=1
export CRAG_TOP_N=3
export CRAG_ANSWER_FORMAT='a concise answer: an extracted span or short phrase; '"'"'Yes'"'"' / '"'"'No'"'"' for yes-no questions; '"'"'Unanswerable'"'"' if the paper does not answer it'
export OMP_NUM_THREADS=8
unset CRAG_CALIBRATE CRAG_UPPER CRAG_LOWER
export CRAG_TRACE="$SCRATCH_DIR/crag-trace.jsonl"
cd "$SCRATCH_DIR"
exec "$WS/.venv/bin/python" "$WS/code/server.py"
