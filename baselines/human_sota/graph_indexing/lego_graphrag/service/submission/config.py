# Replaces Instance/config.py for the upstream LEGO-GraphRAG modules (they `from config import ...`).
# Model paths are resolved relative to this file: <dir>/../models/{all-MiniLM-L6-v2, bge-reranker-v2-m3}.
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODELS = os.environ.get("LEGO_MODELS", os.path.join(os.path.dirname(_HERE), "models"))  # LEGO-GraphRAG/models or workspace/models

llm_url = (os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")) + "/chat/completions"
reasoning_model = "generator"
temperature = 0
max_tokens = 512

emb_model_dir = os.path.join(_MODELS, "all-MiniLM-L6-v2")      # the paper's EEM (Instance/config.py)
rerank_model_dir = os.path.join(_MODELS, "bge-reranker-v2-m3")

hr_top_k = 10
