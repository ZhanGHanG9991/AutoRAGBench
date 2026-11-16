#!/bin/bash
set -euo pipefail
R=${AUTORAGBENCH_ROOT:?set AUTORAGBENCH_ROOT}
B=$R/humansota/BookRAG
export UV_PYTHON_INSTALL_DIR=$R/envs/python UV_CACHE_DIR=/tmp/uv-cache-bookrag-$USER UV_LINK_MODE=copy
export UV_INDEX_URL=${UV_INDEX_URL:-https://pypi.org/simple} UV_HTTP_TIMEOUT=300
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
cd "$B"
export PYTHONPATH=$B/rageng/vendor
python3 rageng/stage_source.py
[ -d .venv ] || ~/.local/bin/uv venv --python 3.12 .venv
~/.local/bin/uv pip install --python "$B/.venv/bin/python" -r rageng/requirements.txt
mkdir -p models rageng/tiktoken_cache
for m in bge-base-en-v1.5 bge-reranker-base; do
  [ -d "models/$m" ] || cp -r "$R/envs/models/$m" models/
done
TIKTOKEN_CACHE_DIR=$B/rageng/tiktoken_cache .venv/bin/python -c 'import tiktoken; tiktoken.get_encoding("cl100k_base")'
~/.local/bin/uv pip freeze --python "$B/.venv/bin/python" > rageng/environment.lock.txt
.venv/bin/python -c 'import sys; sys.path.insert(0,"rageng"); import runtime; print("BOOKRAG_IMPORT_OK")'
