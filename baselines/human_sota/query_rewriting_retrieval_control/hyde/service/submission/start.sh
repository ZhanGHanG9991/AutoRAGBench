#!/bin/bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$SCRATCH_DIR/hf" TORCH_HOME="$SCRATCH_DIR/torch"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
W=$(cd "$(dirname "$0")/.." && pwd)
exec "$W/.venv/bin/python" "$W/submission/server.py"
