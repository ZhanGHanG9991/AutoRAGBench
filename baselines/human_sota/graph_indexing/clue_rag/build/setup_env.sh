#!/bin/bash
# Environment setup. Creates the uv environment for Clue-RAG (the packages of upstream requirements.txt that
# the code imports; torch / transformers pinned to the versions of envs/rag-base so that index-time and
# query-time embeddings come from the same stack) and fetches the models the paper uses:
#   BGE-M3 (embeddings), BGE-Reranker-v2-M3 (re-ranker), spaCy en_core_web_trf 3.8.0 (KU entity NER),
#   NLTK punkt_tab (sentence split of the non-LLM chunks), tiktoken cl100k_base (upstream chunker).
# Place the public spaCy wheel and NLTK punkt_tab archive in vendor/.
#   bash humansota/ClueRAG/rageng/setup_env.sh
set -euo pipefail
source ${AUTORAGBENCH_ROOT:?set AUTORAGBENCH_ROOT}/configs/pilot.env
C=$RAGENG_ROOT/humansota/ClueRAG
export UV_PYTHON_INSTALL_DIR=$RAGENG_ROOT/envs/python UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/uv-cache-autoragbench}  # shared Lustre cache fails with ENODATA on copy
export UV_INDEX_URL=$PIP_MIRROR UV_HTTP_TIMEOUT=300 UV_LINK_MODE=copy
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
UV=~/.local/bin/uv
cd $C
echo "sha256 check:"; echo "272a31e9d8530d1e075351d30a462d7e80e31da23574f1b274e200f3fff35bf5  vendor/en_core_web_trf-3.8.0-py3-none-any.whl" | sha256sum -c
[ -d .venv ] || $UV venv --python 3.12 --seed .venv
VIRTUAL_ENV=$C/.venv $UV pip install "torch==2.14.1" "transformers==5.19.0" \
  "spacy==3.8.7" "spacy-curated-transformers==0.3.1" "curated-transformers==0.1.1" "curated-tokenizers==0.0.9" \
  "cupy-cuda12x" vendor/en_core_web_trf-3.8.0-py3-none-any.whl \
  "nltk==3.9.2" "sacrebleu==2.5.1" "tiktoken==0.12.0" "openai" "ollama==0.6.0" "PyMySQL==1.1.2" \
  numpy tqdm httpx fastapi "uvicorn[standard]" huggingface_hub sentencepiece
# NLTK data + tiktoken BPE inside the project (no writes to $HOME)
mkdir -p $C/nltk_data/tokenizers $C/tiktoken_cache
[ -d $C/nltk_data/tokenizers/punkt_tab ] || unzip -q vendor/punkt_tab.zip -d $C/nltk_data/tokenizers/
TIKTOKEN_CACHE_DIR=$C/tiktoken_cache $C/.venv/bin/python -c "import tiktoken; print('cl100k', tiktoken.get_encoding('cl100k_base').n_vocab)"
mkdir -p $C/models
[ -d $C/models/bge-reranker-v2-m3 ] || cp -al $RAGENG_ROOT/humansota/LEGO-GraphRAG/models/bge-reranker-v2-m3 $C/models/  # same snapshot, hard links
export HF_ENDPOINT=$HF_MIRROR HF_HUB_DISABLE_XET=1
$C/.venv/bin/python - <<PY
from huggingface_hub import snapshot_download
snapshot_download("BAAI/bge-m3", local_dir="$C/models/bge-m3",
                  ignore_patterns=["onnx/*", "*.onnx", "imgs/*", "colbert_linear.pt", "sparse_linear.pt", "long.jpg"])
PY
du -sh $C/models/* $C/.venv
NLTK_DATA=$C/nltk_data $C/.venv/bin/python -c "
import torch, transformers, spacy, nltk, sacrebleu, tiktoken, pymysql, ollama, openai
from nltk.tokenize import sent_tokenize
print('env ok', torch.__version__, transformers.__version__, spacy.__version__, sent_tokenize('A b. C d.'))
spacy.load('en_core_web_trf'); print('en_core_web_trf ok')"
