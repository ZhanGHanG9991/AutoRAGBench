#!/bin/bash
set -euo pipefail
W=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export TIKTOKEN_CACHE_DIR="$W/tiktoken_cache" CARROT_UPSTREAM="$W/upstream" CARROT_MODEL_ROOT="$W/models" HF_HOME="$W/hf-home" CARROT_GTE_OFFLINE_DIR="$W/gte-offline"
export CARROT_RAGENG_ROOT="$W"
exec "$W/.venv/bin/python" "$W/submission/carrot_server.py"
