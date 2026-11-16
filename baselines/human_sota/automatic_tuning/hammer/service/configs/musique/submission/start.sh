#!/bin/bash
set -euo pipefail
W=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false HF_HOME="$W/hf-home"
export LLAMA_INDEX_CACHE_DIR="$W/hf-home/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HAMMER_LOCAL_BGE_M3="$W/models/bge-m3"
cd "$SCRATCH_DIR"
exec "$W/.venv/bin/python" "$W/submission/hammer_server.py"
