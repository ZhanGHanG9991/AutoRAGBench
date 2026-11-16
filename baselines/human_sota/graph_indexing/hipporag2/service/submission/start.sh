#!/bin/bash
set -euo pipefail
W=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$W/adapter"
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
exec "$W/venv/bin/python" "$W/adapter/server.py"
