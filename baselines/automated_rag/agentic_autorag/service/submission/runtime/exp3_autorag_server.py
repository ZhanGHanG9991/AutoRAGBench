"""Serve AutoRAG retrieval. The index must already exist on disk.

This process does not build an index. The job writes EXP3_LANCE_DIR and
EXP3_TRIAL_JSON, then starts this file.

"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from exp3lib.budget import CallBudget
from exp3lib.serve import load_corpus, serve


def _device() -> str:
    return "cuda" if os.environ.get("EXP3_DEVICE") == "cuda" else "cpu"


def _load_embedder(model_name: str):
    from exp3lib.medcpt import maybe_medcpt_embedder

    custom = maybe_medcpt_embedder(model_name, device=_device())
    if custom is None:
        from sentence_transformers import SentenceTransformer

        custom = SentenceTransformer(model_name, local_files_only=True, device=_device())
    custom.exp3_model_name = model_name
    return custom


def _load_cross_encoder(model_name: str):
    from exp3lib.medcpt import maybe_medcpt_cross_encoder

    custom = maybe_medcpt_cross_encoder(model_name, device=_device())
    if custom is not None:
        return custom
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, local_files_only=True, device=_device())


def build_retrieve():
    lance_dir = os.environ.get("EXP3_LANCE_DIR")
    trial_path = os.environ.get("EXP3_TRIAL_JSON")
    corpus_path = os.environ.get("EXP3_CORPUS")
    if not lance_dir or not trial_path or not corpus_path:
        raise SystemExit("EXP3_LANCE_DIR, EXP3_TRIAL_JSON and EXP3_CORPUS are required")
    corpus = load_corpus(corpus_path)
    trial = json.loads(Path(trial_path).read_text(encoding="utf-8"))
    from agentic_autorag.config.models import IndexType, RuntimeConfig
    from agentic_autorag.engine.pipeline import RAGPipeline
    from agentic_autorag.engine.vector_store import LanceDBStore

    from agentic_autorag.litellm_runtime import configure_litellm_runtime
    configure_litellm_runtime({"generator":{"model":"openai/generator","api_base":os.environ["LLM_BASE_URL"],"api_key":os.environ.get("LLM_API_KEY", "EMPTY"),"extra_body":{"chat_template_kwargs":{"enable_thinking":False}}}})
    store = LanceDBStore(lance_dir)
    store.open_table(os.environ.get("EXP3_LANCE_TABLE", "documents"))
    models = Path(os.environ["EXP3_MODELS_DIR"])
    trial["embedding_model"] = str(models / Path(trial["embedding_model"]).name)
    if trial["runtime"].get("reranker") not in (None, "none"):
        trial["runtime"]["reranker"] = str(models / Path(trial["runtime"]["reranker"]).name)
    runtime = RuntimeConfig.model_validate(trial["runtime"])
    embedder = _load_embedder(trial["embedding_model"])
    cross = None if runtime.reranker == "none" else _load_cross_encoder(runtime.reranker)
    pipeline = RAGPipeline(
        store,
        None,
        runtime,
        embedder,
        IndexType(trial["index_type"]),
        cross,
    )
    budget = CallBudget()

    def retrieve(question, meta):
        budget.reset()
        pipeline.call_meter = budget
        result = asyncio.run(pipeline.retrieve(question, meta=meta or None))
        hits = []
        for doc in result.documents:
            if doc.char_range is None:
                continue
            start, end = doc.char_range
            hits.append(
                {
                    "doc_id": str(doc.metadata.get("doc_id", "")),
                    "text": doc.text,
                    "start": start,
                    "end": end,
                }
            )
        return hits, corpus

    return retrieve


def main() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    serve(build_retrieve())


if __name__ == "__main__":
    main()
