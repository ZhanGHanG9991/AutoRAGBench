#!/bin/bash
set -euo pipefail
W=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONHOME="$W/python"
export PYTHONPATH="$W/code:$W/upstream:$W/.venv/lib/python3.11/site-packages"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHON_DOTENV_DISABLED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TIKTOKEN_CACHE_DIR="$W/tiktoken_cache"
export NUMBA_CACHE_DIR="${SCRATCH_DIR:-$W/scratch}/numba"
PYTHON="$W/python/bin/python3.11"
