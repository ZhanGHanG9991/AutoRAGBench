#!/bin/bash
set -euo pipefail
W=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false HF_HOME="$W/hf-home" TIKTOKEN_CACHE_DIR="$W/tiktoken_cache"
export LLAMA_INDEX_CACHE_DIR="$W/hf-home/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$SCRATCH_DIR"
exec "$W/.venv/bin/python" "$W/submission/hammer_server.py"
