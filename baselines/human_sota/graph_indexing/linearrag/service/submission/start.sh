#!/bin/bash
set -euo pipefail
W=$(cd "$(dirname "$0")/.." && pwd)
P=$W/.venv/lib/python3.12/site-packages
export PYTHONPATH="$W/adapter"
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export LD_LIBRARY_PATH="$P/nvidia/cuda_nvrtc/lib:$P/nvidia/cublas/lib:$P/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
exec "$W/.venv/bin/python" "$W/adapter/server.py"
