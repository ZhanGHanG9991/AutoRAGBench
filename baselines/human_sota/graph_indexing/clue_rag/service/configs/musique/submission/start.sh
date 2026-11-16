#!/bin/bash
set -euo pipefail
W=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$W/pylib" NLTK_DATA="$W/pylib/nltk_data" TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=20261022
exec "$W/.venv/bin/python" "$W/submission/clue_server.py"
