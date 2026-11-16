#!/bin/bash
# Environment setup. Creates the uv environment for LEGO-GraphRAG (upstream requirements + what the
# AutoRAGBench adapter needs) and downloads the EEMs the paper uses (all-MiniLM-L6-v2, bge-reranker-v2-m3).
#   bash humansota/LEGO-GraphRAG/rageng/setup_env.sh
set -euo pipefail
source ${AUTORAGBENCH_ROOT:?set AUTORAGBENCH_ROOT}/configs/pilot.env
L=$RAGENG_ROOT/humansota/LEGO-GraphRAG
export UV_PYTHON_INSTALL_DIR=$RAGENG_ROOT/envs/python UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/uv-cache-autoragbench}  # shared Lustre cache fails with ENODATA on copy
export UV_INDEX_URL=$PIP_MIRROR UV_HTTP_TIMEOUT=300 UV_LINK_MODE=copy
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
UV=~/.local/bin/uv
cd $L
[ -d .venv ] || $UV venv --python 3.12 --seed .venv
# torch pinned to the version of envs/rag-base (used for the evaluated environment)
VIRTUAL_ENV=$L/.venv $UV pip install "torch==2.14.1" -r requirements.txt python-igraph httpx bm25s fastapi "uvicorn[standard]" huggingface_hub
mkdir -p $L/models
export HF_ENDPOINT=$HF_MIRROR HF_HUB_DISABLE_XET=1
$L/.venv/bin/python - <<PY
from huggingface_hub import snapshot_download
for m in ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-reranker-v2-m3"]:
    snapshot_download(m, local_dir="$L/models/" + m.split("/")[1],
                      ignore_patterns=["*.onnx", "onnx/*", "openvino/*", "*.h5", "*.msgpack", "*.ot", "tf_model*", "*.bin"])
PY
du -sh $L/models/*
$L/.venv/bin/python -c "import torch, igraph, sentence_transformers, langchain_community; print('env ok', torch.__version__, igraph.__version__, sentence_transformers.__version__)"
