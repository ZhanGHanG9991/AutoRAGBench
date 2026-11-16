"""Contriever-MSMARCO index. Mean pooling and dot product, no L2 normalisation.

This is the retriever Facebook ships with facebook/contriever-msmarco.
It replaces only the first-round retrieval when CRAG_FIRST_RETRIEVER=contriever.
The T5 action rule and the rewrite retrieval stay where the server puts them.
"""
import json
import os

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def mean_pool(hidden, mask):
    weights = mask.unsqueeze(-1).type_as(hidden)
    return (hidden * weights).sum(1) / weights.sum(1).clamp(min=1e-9)


class ContrieverIndex:
    def __init__(self, model_dir, docs, embeddings, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_dir, local_files_only=True)
        self.model.to(self.device).eval()
        self.docs = docs
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.row_of = {}
        for i, doc in enumerate(docs):
            self.row_of.setdefault(doc["doc_id"], []).append(i)

    def encode(self, texts, batch_size=32):
        rows = []
        for start in range(0, len(texts), batch_size):
            if start == 0 or start % (batch_size * 20) == 0:
                print(f"ENCODE {start}/{len(texts)}", flush=True)
            batch = texts[start : start + batch_size]
            enc = self.tokenizer(
                batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                hidden = self.model(**enc).last_hidden_state
                pooled = mean_pool(hidden, enc["attention_mask"])
            rows.append(pooled.float().cpu().numpy())
        return np.concatenate(rows, axis=0)

    def search(self, query, k, only_doc_id=None):
        allowed = None if not only_doc_id else self.row_of.get(only_doc_id, [])
        if allowed is not None and not allowed:
            return []
        query_vec = self.encode([query])[0]
        if allowed is None:
            scores = self.embeddings @ query_vec
            order = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
            order = order[np.argsort(-scores[order])]
        else:
            block = self.embeddings[allowed]
            scores_block = block @ query_vec
            take = min(k, len(scores_block))
            local = np.argpartition(-scores_block, take - 1)[:take]
            local = local[np.argsort(-scores_block[local])]
            order = [allowed[i] for i in local]
            scores = None
        out = []
        for i in order:
            i = int(i)
            doc = self.docs[i]
            if only_doc_id and doc["doc_id"] != only_doc_id:
                continue
            score = float(self.embeddings[i] @ query_vec) if scores is None else float(scores[i])
            out.append({
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "passage": doc["passage"],
                "dot": score,
                "row": i,
            })
        return out


def _paths(index_dir):
    return (
        os.path.join(index_dir, "docs.jsonl"),
        os.path.join(index_dir, "embeddings.npy"),
        os.path.join(index_dir, "contriever.json"),
    )


def load_built(index_dir, model_dir):
    """Load a finished index. Does not build and does not look at chunk_words."""
    meta, emb_path, stamp = _paths(index_dir)
    for path in (meta, emb_path, stamp):
        if not os.path.isfile(path):
            raise SystemExit(f"missing contriever index file {path}")
    docs = [json.loads(line) for line in open(meta, encoding="utf-8")]
    embeddings = np.load(emb_path)
    return ContrieverIndex(model_dir, docs, embeddings)


def load_chunked_corpus(corpus_dir, words):
    """Same windows as baselines/ragcore.py iter_passages.

    Each window keeps the parent doc_id. The stored passage is title plus a
    newline plus the window, which is the string T5 and Contriever score.
    """
    if not isinstance(words, int) or words < 1:
        raise SystemExit(f"chunk words must be a positive int, got {words}")
    path = os.path.join(corpus_dir, "corpus.jsonl")
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            title = d["title"].replace("\t", " ").replace("\n", " ")
            toks = d["text"].split()
            for start in range(0, max(len(toks), 1), words):
                window = " ".join(toks[start : start + words])
                passage = f"{title}\n{window}".strip() if title else window
                docs.append({"doc_id": d["doc_id"], "title": title, "passage": passage})
    return docs


def build_or_load(corpus_dir, index_dir, model_dir, batch_size=32, chunk_words=None):
    from bm25_index import load_corpus

    os.makedirs(index_dir, exist_ok=True)
    meta, emb_path, stamp = _paths(index_dir)
    if os.path.isfile(meta) and os.path.isfile(emb_path) and os.path.isfile(stamp):
        saved = json.load(open(stamp, encoding="utf-8")).get("chunk_words")
        if saved != chunk_words:
            raise SystemExit(
                f"index {index_dir} chunk_words={saved!r} does not match requested {chunk_words!r}"
            )
        return load_built(index_dir, model_dir)
    if chunk_words is None:
        docs = load_corpus(corpus_dir)
    else:
        docs = load_chunked_corpus(corpus_dir, chunk_words)
    if not docs:
        raise SystemExit(f"empty corpus {corpus_dir}")
    index = ContrieverIndex(model_dir, docs, np.zeros((len(docs), 1), dtype=np.float32))
    embeddings = index.encode([d["passage"] for d in docs], batch_size=batch_size)
    np.save(emb_path, embeddings.astype(np.float32))
    with open(meta, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    json.dump({
        "model": "facebook/contriever-msmarco",
        "pooling": "mean",
        "similarity": "dot",
        "normalized": False,
        "max_length": 512,
        "n_docs": len(docs),
        "dim": int(embeddings.shape[1]),
        "chunk_words": chunk_words,
    }, open(stamp, "w"), indent=1)
    index.embeddings = embeddings.astype(np.float32)
    return index
