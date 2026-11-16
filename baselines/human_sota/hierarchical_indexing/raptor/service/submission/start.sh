#!/bin/bash
set -euo pipefail
W=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$W/upstream:$W/adapter"
export TIKTOKEN_CACHE_DIR="$W/tiktoken_cache"
export NUMBA_CACHE_DIR="$SCRATCH_DIR/numba"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
exec "$W/venv/bin/python" "$W/adapter/server.py"
