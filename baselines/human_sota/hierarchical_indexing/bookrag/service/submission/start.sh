#!/bin/bash
set -euo pipefail
W=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$W/adapter/vendor:$W/adapter" TIKTOKEN_CACHE_DIR="$W/adapter/tiktoken_cache"
export PYTHONHASHSEED=20261022 ANONYMIZED_TELEMETRY=False TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
exec "$W/venv/bin/python" "$W/adapter/bookrag_server.py"
