"""BM25 over a AutoRAGBench corpus.jsonl. No learned parameters."""
import json
import os

import bm25s


def load_corpus(corpus_dir):
    path = os.path.join(corpus_dir, "corpus.jsonl")
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            title = (d.get("title") or "").strip()
            body = (d.get("text") or "").strip()
            passage = f"{title}\n{body}".strip() if title else body
            docs.append({"doc_id": d["doc_id"], "title": title, "passage": passage})
    return docs


def build_or_load(corpus_dir, index_dir):
    os.makedirs(index_dir, exist_ok=True)
    meta = os.path.join(index_dir, "docs.jsonl")
    store = os.path.join(index_dir, "bm25")
    if os.path.isfile(meta) and os.path.isdir(store):
        docs = [json.loads(l) for l in open(meta, encoding="utf-8")]
        retriever = bm25s.BM25.load(store, load_corpus=False)
        return docs, retriever
    docs = load_corpus(corpus_dir)
    tokens = bm25s.tokenize([d["passage"] for d in docs], stopwords="english", show_progress=True)
    retriever = bm25s.BM25()
    retriever.index(tokens, show_progress=True)
    retriever.save(store)
    with open(meta, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return docs, retriever


def search(docs, retriever, query, k, only_doc_id=None):
    """only_doc_id limits the candidate set to that document, the QASPER scope.

    MuSiQue leaves it empty, so the retrieve call is the same as before.
    """
    import numpy as np

    q = bm25s.tokenize([query], stopwords="english", show_progress=False)
    mask = None
    if only_doc_id:
        mask = np.array([1.0 if d["doc_id"] == only_doc_id else 0.0 for d in docs], dtype=np.float32)
        if float(mask.sum()) == 0:
            return []
    idxs, scores = retriever.retrieve(
        q, k=min(k, len(docs)), show_progress=False, weight_mask=mask
    )
    out = []
    for i, s in zip(idxs[0].tolist(), scores[0].tolist()):
        d = docs[int(i)]
        if only_doc_id and d["doc_id"] != only_doc_id:
            continue
        out.append({"doc_id": d["doc_id"], "title": d["title"], "passage": d["passage"], "bm25": float(s)})
    return out
