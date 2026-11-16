#!/bin/bash
# Environment setup. uv environment for HAMMER (upstream requirements.txt unchanged, plus the service
# dependencies of the AutoRAGBench adapter) and the retrieval-side models the official code can load.
#   bash humansota/HAMMER/rageng/setup_env.sh
set -euo pipefail
source ${AUTORAGBENCH_ROOT:?set AUTORAGBENCH_ROOT}/configs/pilot.env
H=$RAGENG_ROOT/humansota/HAMMER
export UV_PYTHON_INSTALL_DIR=$RAGENG_ROOT/envs/python UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/uv-cache-autoragbench}  # Lustre cache copy fails (ENODATA)
export UV_INDEX_URL=$PIP_MIRROR UV_HTTP_TIMEOUT=300 UV_LINK_MODE=copy
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
UV=~/.local/bin/uv
cd $H
[ -d .venv ] || $UV venv --python 3.12 --seed .venv
VIRTUAL_ENV=$H/.venv $UV pip install -r requirements.txt fastapi "uvicorn[standard]" httpx sentence-transformers PyStemmer
# requirements.txt asks for llama-index[...] extras that the 0.10.61 meta-package does not define. The
# integrations the code imports unconditionally are installed explicitly, constrained to the installed core
# (the others are imported inside try/except upstream and are not used by the MCTS path).
CORE=$($H/.venv/bin/python -c "import llama_index.core as c; print(c.__version__)")
VIRTUAL_ENV=$H/.venv $UV pip install "llama-index-core==$CORE" "transformers==4.48.3" "torch==2.7.0" \
  llama-index-llms-openai-like llama-index-embeddings-huggingface llama-index-embeddings-huggingface-optimum \
  llama-index-retrievers-bm25 llama-index-vector-stores-faiss networkx "pyarrow<20"  # newer pyarrow needs numpy>=2; llama-index 0.10 pins numpy<2
# HAMMER passes use_auth_token to HuggingFaceEmbedding: that needs the 0.2.x integration (SentenceTransformer
# backend, **model_kwargs) and sentence-transformers < 3.1. The optimum integration (imported, only used for ONNX
# embeddings, which the MCTS path never builds) pins < 0.2, hence --no-deps.
VIRTUAL_ENV=$H/.venv $UV pip install --no-deps "llama-index-embeddings-huggingface==0.2.3"
VIRTUAL_ENV=$H/.venv $UV pip install "sentence-transformers==3.0.1" "llama-index-core==$CORE" "transformers==4.48.3" "torch==2.7.0"
mkdir -p $H/results $H/studies $H/hammer/data/sota   # created by hammer.configuration at import; read-only later
$H/.venv/bin/python -c "import torch, transformers, llama_index.core, optuna; print('env ok', torch.__version__, transformers.__version__, llama_index.core.__version__, optuna.__version__)"
