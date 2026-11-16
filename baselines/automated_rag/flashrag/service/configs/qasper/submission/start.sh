#!/bin/bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUN=$(dirname -- "$HERE")
export EXP3_PREP="$HERE"
export EXP3_TASK="qasper"
export EXP3_CORPUS="${CORPUS_DIR:-$RUN/../task/corpus}/corpus.jsonl"
export EXP3_DEVICE=cuda HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 EXP3_DISABLE_THINKING=1 LITELLM_LOCAL_MODEL_COST_MAP=True
export EXP3_MODELS_DIR="$RUN/runtime/models"
export PYTHONPATH="$HERE/lib:$HERE/runtime:$RUN/runtime/vendor"
export EXP3_FLASHRAG_CANDIDATE="std_bge"
export EXP3_BM25_DIR="$RUN/indexes/bm25"
export EXP3_DENSE_DIR="$RUN/indexes/bge-large-en-v1.5"
exec "$RUN/runtime/env/bin/python" "$HERE/runtime/exp3_flashrag_server.py"
