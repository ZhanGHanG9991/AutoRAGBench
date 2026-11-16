"""Serve one frozen FlashRAG candidate. Indexes must already exist.

The candidate id is EXP3_FLASHRAG_CANDIDATE. It must be one of the ids in
flashrag/candidates.json. This process does not build an index, does not
change the candidate list, and does not load MedCPT.

"""

from __future__ import annotations

import json
import os
from pathlib import Path

from exp3lib.budget import CallBudget
from exp3lib.flashrag_flow import run_candidate
from exp3lib.platform_chat import platform_complete
from exp3lib.prefixes import apply_prefix
from exp3lib.serve import load_corpus, serve


def _load_candidate(prep: Path, candidate_id: str) -> dict:
    payload = json.loads((prep / "flashrag" / "candidates.json").read_text(encoding="utf-8"))
    for candidate in payload["candidates"]:
        if candidate["id"] == candidate_id:
            return candidate
    raise SystemExit("EXP3_FLASHRAG_CANDIDATE is not in the frozen list")


def _bm25_search(index_dir: str, corpus: dict):
    import bm25s

    retriever = bm25s.BM25.load(index_dir, load_corpus=True)

    def search(query: str, k: int, doc_id):
        import numpy as np
        mask = None
        if doc_id is not None:
            mask = np.asarray([1.0 if str(x.get("doc_id") or x.get("id")) == doc_id else 0.0 for x in retriever.corpus], dtype=np.float32)
        results, _scores = retriever.retrieve(bm25s.tokenize([query], show_progress=False), k=min(k, len(retriever.corpus)), weight_mask=mask, show_progress=False)
        hits = []
        for item in results[0]:
            if isinstance(item, dict):
                found = str(item.get("doc_id") or item.get("id"))
                text = str(item.get("text", ""))
            else:
                found = str(item)
                text = corpus.get(found, {}).get("text", "")
            if found not in corpus:
                continue
            if doc_id is not None and found != doc_id:
                continue
            source = corpus[found]["text"]
            start = source.find(text) if text else -1
            if start < 0:
                continue
            hits.append(
                {
                    "doc_id": found,
                    "text": source[start : start + len(text)],
                    "start": start,
                    "end": start + len(text),
                }
            )
        return hits

    return search


def _dense_search(index_dir: str, model_name: str, corpus: dict):
    import faiss
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, local_files_only=True, device="cpu")
    if os.environ.get("EXP3_DEVICE") == "cuda":
        model = model.to("cuda")
    index = faiss.read_index(str(Path(index_dir) / "index.faiss"))
    ids = (Path(index_dir) / "doc_ids.txt").read_text(encoding="utf-8").splitlines()
    spans = json.loads((Path(index_dir) / "spans.json").read_text(encoding="utf-8"))

    def search(query: str, k: int, doc_id):
        text = apply_prefix(query, model_name, "query", bool(getattr(model, "prompts", None)))
        vector = model.encode([text], normalize_embeddings=True)
        if doc_id is not None:
            import numpy as np
            allowed = np.asarray([i for i,d in enumerate(ids) if d == doc_id], dtype=np.int64)
            if not len(allowed): return []
            matrix = np.vstack([index.reconstruct(int(i)) for i in allowed])
            ranked = np.argsort(-(matrix @ vector[0]))[:k]
            rows = [allowed[ranked]]
        else:
            _scores, rows = index.search(vector, k)
        hits = []
        for row in rows[0]:
            if row < 0 or row >= len(ids):
                continue
            found = ids[row]
            if doc_id is not None and found != doc_id:
                continue
            start, end = spans[row]
            source = corpus[found]["text"]
            hits.append({"doc_id": found, "text": source[start:end], "start": start, "end": end})
        return hits

    return search


def _rerank(model_name: str):
    from sentence_transformers import CrossEncoder

    encoder = CrossEncoder(model_name, local_files_only=True, device="cpu")
    if os.environ.get("EXP3_DEVICE") == "cuda":
        encoder.model.to("cuda")

    def rerank(question, hits):
        if not hits:
            return []
        pairs = [(question, hit["text"]) for hit in hits]
        scores = encoder.predict(pairs)
        order = sorted(range(len(hits)), key=lambda i: float(scores[i]), reverse=True)
        return [hits[i] for i in order]

    return rerank


def build_retrieve():
    prep = Path(os.environ.get("EXP3_PREP", str(Path(__file__).resolve().parents[1])))
    candidate_id = os.environ.get("EXP3_FLASHRAG_CANDIDATE")
    corpus_path = os.environ.get("EXP3_CORPUS")
    if not candidate_id or not corpus_path:
        raise SystemExit("EXP3_FLASHRAG_CANDIDATE and EXP3_CORPUS are required")
    candidate = _load_candidate(prep, candidate_id)
    corpus = load_corpus(corpus_path)
    models = Path(os.environ.get("EXP3_MODELS_DIR", str(prep / "models")))
    if candidate["retriever"] == "bm25":
        search = _bm25_search(os.environ["EXP3_BM25_DIR"], corpus)
    elif candidate["retriever"] == "e5":
        search = _dense_search(os.environ["EXP3_DENSE_DIR"], str(models / "e5-large-v2"), corpus)
    elif candidate["retriever"] == "bge":
        search = _dense_search(os.environ["EXP3_DENSE_DIR"], str(models / "bge-large-en-v1.5"), corpus)
    else:
        raise SystemExit("candidate retriever is not in the frozen list")
    rerank = None
    if candidate.get("reranker"):
        rerank = _rerank(str(models / "bge-reranker-v2-m3"))
    budget = CallBudget()

    def complete(prompt: str) -> str:
        return platform_complete(prompt, budget)

    def retrieve(question, meta):
        budget.reset()
        hits = run_candidate(candidate, question, search, complete, rerank=rerank, meta=meta)
        return hits, corpus

    return retrieve


def main() -> None:
    serve(build_retrieve())


if __name__ == "__main__":
    main()
