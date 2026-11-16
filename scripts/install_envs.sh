#!/bin/bash
# Build the two shared Python environments on the login node (CPU only).
#   bash scripts/install_envs.sh vllm       # evaluator side: vLLM + gateway + evaluator
#   bash scripts/install_envs.sh rag-base   # read-only starting env for every agent and baseline
# Python 3.12 is uv-managed under envs/python; packages come from $PIP_MIRROR (configs/pilot.env).
# Exact package versions of the paper's environments: requirements/vllm.txt, requirements/rag-base.txt.
set -euo pipefail
source "$(dirname "$0")/../configs/pilot.env"
R=$RAGENG_ROOT
export UV_PYTHON_INSTALL_DIR=$R/envs/python UV_CACHE_DIR=$R/envs/.uv-cache
[ -n "${UV_PYTHON_INSTALL_MIRROR:-}" ] && export UV_PYTHON_INSTALL_MIRROR
export UV_INDEX_URL=$PIP_MIRROR UV_HTTP_TIMEOUT=300
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
UV=${UV:-~/.local/bin/uv}
$UV python install 3.12
case ${1:?usage: install_envs.sh vllm|rag-base} in
  vllm)
    $UV venv --python 3.12 $R/envs/vllm
    VIRTUAL_ENV=$R/envs/vllm $UV pip install "vllm==0.31.0" pyyaml ;;
  rag-base)
    $UV venv --python 3.12 --seed $R/envs/rag-base
    VIRTUAL_ENV=$R/envs/rag-base $UV pip install torch sentence-transformers faiss-cpu bm25s rank_bm25 \
      fastapi "uvicorn[standard]" httpx openai pymupdf numpy pandas pyarrow scikit-learn tqdm pyyaml requests
    chmod -R a-w $R/envs/rag-base ;;   # agents build their own .venv on top of it
esac
echo "DONE $1 $(date -Is)"
