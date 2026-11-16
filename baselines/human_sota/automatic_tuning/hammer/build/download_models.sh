#!/bin/bash
# Model setup. Retrieval-side models into the HF cache rageng/.hf-home (hf-mirror), so the official
# loading code (HuggingFaceEmbedding / CrossEncoder / AutoTokenizer by repo id) runs offline later.
#   - the 9 embedding models of the paper's technical report (Table 9), all <= 1B parameters
#   - cross-encoder/ms-marco-MiniLM-L-6-v2: the model the official BatchReranker always loads
#   - Qwen/Qwen2.5-7B tokenizer only: used by the official sentence/token splitters for chunk sizes
set -euo pipefail
source ${AUTORAGBENCH_ROOT:?set AUTORAGBENCH_ROOT}/configs/pilot.env
H=$RAGENG_ROOT/humansota/HAMMER
export HF_HOME=$H/rageng/.hf-home HF_ENDPOINT=$HF_MIRROR HF_HUB_DISABLE_XET=1
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
mkdir -p $HF_HOME
$H/.venv/bin/python - <<'PY'
import json, os
from huggingface_hub import snapshot_download
skip = ["*.onnx", "onnx/*", "openvino/*", "*.h5", "*.msgpack", "*.ot", "tf_model*", "rust_model*", "*.gguf", "imgs/*", "colbert*", "sparse*"]
models = json.load(open(os.path.join(os.environ["AUTORAGBENCH_ROOT"], "humansota/HAMMER/rageng/search_space.json")))["embedding_models"]
for m in models + ["cross-encoder/ms-marco-MiniLM-L-6-v2"]:
    p = snapshot_download(m, ignore_patterns=skip + (["*.bin"] if m != "BAAI/bge-m3" else ["*.safetensors"]))
    print("OK", m, p, flush=True)
p = snapshot_download("Qwen/Qwen2.5-7B", allow_patterns=["tokenizer*", "vocab.json", "merges.txt", "config.json", "generation_config.json"])
print("OK tokenizer", p)
PY
du -sh $HF_HOME/hub/* | sort -h
