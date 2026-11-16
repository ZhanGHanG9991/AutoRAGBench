#!/bin/bash
set -euo pipefail
W=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$W/pylib" PYTHONDONTWRITEBYTECODE=1
exec "$W/.venv/bin/python" "$W/submission/lego_server.py"
