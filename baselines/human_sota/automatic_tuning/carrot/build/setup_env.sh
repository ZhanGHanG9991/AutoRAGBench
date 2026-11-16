#!/bin/bash
set -euo pipefail
R=${AUTORAGBENCH_ROOT:?set AUTORAGBENCH_ROOT}
C=$R/humansota/CARROT
export UV_CACHE_DIR=/tmp/uv-cache-carrot-$USER UV_LINK_MODE=copy UV_HTTP_TIMEOUT=300
export UV_PYTHON_INSTALL_DIR=$R/envs/python UV_INDEX_URL=${UV_INDEX_URL:-https://pypi.org/simple}
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
cd "$C"
[ -d .venv ] || ~/.local/bin/uv venv --python 3.12 .venv
~/.local/bin/uv pip install --python "$C/.venv/bin/python" -r requirements.txt 'numpy<2' 'peft==0.13.2' 'datasets<3' 'pyarrow<20' fastapi uvicorn httpx bm25s
~/.local/bin/uv pip freeze --python "$C/.venv/bin/python" > rageng/environment.lock.txt
"$C/.venv/bin/python" -c 'import torch, FlagEmbedding, core.mcts; print("CARROT_IMPORT_OK", torch.__version__)'
