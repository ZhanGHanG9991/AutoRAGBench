#!/bin/bash
set -euo pipefail
W=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# HyDE runs on the platform's standard read-only Python environment, exactly as the frozen reproduction did.
export PYTHONPATH="$W/submission"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
PYTHON="$W/.venv/bin/python"
