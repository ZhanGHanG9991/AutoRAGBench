"""Shared code for the evaluator's baselines (copied into each baseline workspace by new_run.sh).

Corpus documents are split into ~200-word passages. Retrieval: BM25 (Lucene/bm25s scoring, k1=1.5, b=0.75), optional
dense (any sentence-transformers model) fused with reciprocal rank fusion, optional cross-encoder reranking.
Questions with meta.doc_id (QASPER) search only the passages of that document.

The index is built once before the workspace is frozen (`python ragcore.py build ...`, run by slurm/run.sbatch for
baselines) into <workspace>/index/, and memory-mapped at service start, so that 21M-passage corpora fit the
20-minute startup limit:
  passages.bin + offsets.npy     "title\\ttext\\n" per passage, byte offsets (N + 1)
  doc_of.npy, doc_ids.json       passage -> document; doc_first.npy / doc_count.npy (passages of a document are contiguous)
  bm25_{data,indices,indptr}.npy term-major (CSC) BM25 weight matrix; vocab.json
  emb.npy                        float16 normalised passage embeddings (dense baselines only)
"""
import array
import collections
import json
import math
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX = os.path.join(os.path.dirname(HERE), "index")
K1, B = 1.5, 0.75


def tokenize(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def iter_passages(corpus_dir, words=200):
    for line in open(os.path.join(corpus_dir, "corpus.jsonl")):
        d = json.loads(line)
        toks = d["text"].split()
        for s in range(0, max(len(toks), 1), words):
            yield d["doc_id"], d["title"].replace("\t", " ").replace("\n", " "), " ".join(toks[s:s + words])


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- build (before freeze)

def build(corpus_dir, out, dense_model=None, device="cuda", batch=512):
    os.makedirs(out, exist_ok=True)
    vocab, doc_ids, doc_index = {}, [], {}
    offsets, doc_of, lengths = array.array("q", [0]), array.array("i"), array.array("i")
    rows, cols, tfs = array.array("i"), array.array("i"), array.array("H")
    pos = n = 0
    with open(os.path.join(out, "passages.bin"), "wb") as f:
        for doc, title, text in iter_passages(corpus_dir):
            b = f"{title}\t{text}\n".encode()
            f.write(b)
            pos += len(b)
            offsets.append(pos)
            if doc not in doc_index:
                doc_index[doc] = len(doc_ids)
                doc_ids.append(doc)
            doc_of.append(doc_index[doc])
            toks = tokenize(f"{title} {text}")
            lengths.append(len(toks))
            for t, c in collections.Counter(toks).items():
                tid = vocab.setdefault(t, len(vocab))
                rows.append(n)
                cols.append(tid)
                tfs.append(min(c, 65535))
            n += 1
            if n % 1_000_000 == 0:
                log(f"passages {n:,}  postings {len(rows):,}  vocab {len(vocab):,}")
    log(f"passages {n:,} done; BM25 weights")
    rows, cols = np.frombuffer(rows, dtype=np.int32), np.frombuffer(cols, dtype=np.int32)
    tf = np.frombuffer(tfs, dtype=np.uint16).astype(np.float32)
    lens = np.frombuffer(lengths, dtype=np.int32).astype(np.float32)
    df = np.bincount(cols, minlength=len(vocab)).astype(np.float32)
    idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
    avg = float(lens.mean()) if n else 1.0
    w = idf[cols] * tf * (K1 + 1) / (tf + K1 * (1 - B + B * lens[rows] / avg))
    del tf
    order = np.argsort(cols, kind="stable")  # term-major
    np.save(os.path.join(out, "bm25_data.npy"), w[order].astype(np.float32))
    np.save(os.path.join(out, "bm25_indices.npy"), rows[order])
    del w, order
    indptr = np.zeros(len(vocab) + 1, dtype=np.int64)
    np.cumsum(np.bincount(cols, minlength=len(vocab)), out=indptr[1:])
    np.save(os.path.join(out, "bm25_indptr.npy"), indptr)
    json.dump(vocab, open(os.path.join(out, "vocab.json"), "w"))
    doc_of = np.frombuffer(doc_of, dtype=np.int32)
    np.save(os.path.join(out, "offsets.npy"), np.frombuffer(offsets, dtype=np.int64))
    np.save(os.path.join(out, "doc_of.npy"), doc_of)
    # documents are numbered in order of appearance and their passages are contiguous
    np.save(os.path.join(out, "doc_first.npy"), np.unique(doc_of, return_index=True)[1].astype(np.int64))
    np.save(os.path.join(out, "doc_count.npy"), np.bincount(doc_of, minlength=len(doc_ids)))
    json.dump(doc_ids, open(os.path.join(out, "doc_ids.json"), "w"))
    if dense_model:
        from numpy.lib.format import open_memmap
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(dense_model, device=device)
        model.half()
        store = Passages(out)
        emb = open_memmap(os.path.join(out, "emb.part.npy"), mode="w+", dtype=np.float16,
                          shape=(n, model.get_sentence_embedding_dimension()))
        step = batch * 64
        for s in range(0, n, step):
            texts = [store.text(i) for i in range(s, min(n, s + step))]
            emb[s:s + len(texts)] = model.encode(texts, batch_size=batch, normalize_embeddings=True,
                                                 convert_to_numpy=True, show_progress_bar=False).astype(np.float16)
            if (s // step) % 20 == 0:
                log(f"embedded {s + len(texts):,} / {n:,}")
        emb.flush()
        del emb
        os.replace(os.path.join(out, "emb.part.npy"), os.path.join(out, "emb.npy"))
    json.dump({"n_passages": n, "vocab": len(vocab), "dense": bool(dense_model)}, open(os.path.join(out, "built.json"), "w"))
    log("index built")


# ---------------------------------------------------------------- serving

class Passages:
    def __init__(self, d):
        self.off = np.load(os.path.join(d, "offsets.npy"), mmap_mode="r")
        self.f = open(os.path.join(d, "passages.bin"), "rb")

    def get(self, i):
        self.f.seek(int(self.off[i]))
        title, text = self.f.read(int(self.off[i + 1] - self.off[i])).decode().rstrip("\n").split("\t", 1)
        return title, text

    def text(self, i):
        t, x = self.get(i)
        return f"{t} {x}"


class Retriever:
    def __init__(self, corpus_dir=None, dense_model=None, reranker=None, device="cuda", index=INDEX):
        if not os.path.exists(os.path.join(index, "built.json")):
            # no prebuilt index (pilot runs, small corpora): build it now in the writable scratch directory
            index = os.path.join(os.environ.get("SCRATCH_DIR") or "/tmp", "baseline_index")
            if not os.path.exists(os.path.join(index, "built.json")):
                build(corpus_dir or os.environ["CORPUS_DIR"], index, dense_model, device)
        ld = lambda name: np.load(os.path.join(index, name), mmap_mode="r")
        self.store = Passages(index)
        self.n = len(self.store.off) - 1
        self.doc_of, self.doc_list = ld("doc_of.npy"), json.load(open(os.path.join(index, "doc_ids.json")))
        # Always index documents. search() only restricts when the caller passes doc_id
        # (QASPER). There is no task.json beside this copy, so TASK.uses_meta_doc_id is empty.
        self.doc_index = {d: i for i, d in enumerate(self.doc_list)}
        self.first, self.count = ld("doc_first.npy"), ld("doc_count.npy")
        self.vocab = json.load(open(os.path.join(index, "vocab.json")))
        self.data, self.indices, self.indptr = ld("bm25_data.npy"), ld("bm25_indices.npy"), ld("bm25_indptr.npy")
        self.dense = self.emb = self.rerank = None
        if dense_model:
            import torch
            from sentence_transformers import SentenceTransformer
            self.dense = SentenceTransformer(dense_model, device=device)
            emb = ld("emb.npy")
            free = torch.cuda.mem_get_info()[0] if device == "cuda" and torch.cuda.is_available() else 0
            # on the GPU if it fits with 12 GB to spare (the generator holds its own share), else float32 in RAM
            self.emb = torch.from_numpy(np.ascontiguousarray(emb)).to(device) if emb.nbytes + (12 << 30) < free \
                else np.asarray(emb, dtype=np.float32)
        if reranker:
            from sentence_transformers import CrossEncoder
            self.rerank = CrossEncoder(reranker, device=device)

    def _allowed(self, doc_id):
        d = self.doc_index.get(doc_id)
        return None if d is None else np.arange(self.first[d], self.first[d] + self.count[d])

    @staticmethod
    def _top(scores, k):
        k = min(k, len(scores))
        part = np.argpartition(-scores, k - 1)[:k]
        return part[np.argsort(-scores[part])]

    def _bm25(self, q, k, allowed=None):
        scores = np.zeros(self.n, dtype=np.float32)
        for t in set(tokenize(q)):
            tid = self.vocab.get(t)
            if tid is not None:
                s, e = self.indptr[tid], self.indptr[tid + 1]
                scores[self.indices[s:e]] += self.data[s:e]
        if allowed is not None:
            return [int(allowed[i]) for i in self._top(scores[allowed], k)]
        return [int(i) for i in self._top(scores, k)]

    def _dense(self, q, k, allowed=None):
        qv = self.dense.encode([q], normalize_embeddings=True, convert_to_numpy=True)[0]
        if isinstance(self.emb, np.ndarray):
            sims = (self.emb[allowed] if allowed is not None else self.emb) @ qv
        else:
            import torch
            m = self.emb[torch.as_tensor(allowed, device=self.emb.device)] if allowed is not None else self.emb
            sims = (m @ torch.as_tensor(qv, dtype=m.dtype, device=m.device)).float().cpu().numpy()
        top = self._top(sims, k)
        return [int(allowed[i]) for i in top] if allowed is not None else [int(i) for i in top]

    def search(self, q, k=5, doc_id=None, pool=50, rerank_top=30):
        allowed = self._allowed(doc_id) if doc_id else None
        ranked = self._bm25(q, pool, allowed)
        if self.dense is not None:
            fused = {}
            for lst in (ranked, self._dense(q, pool, allowed)):
                for r, i in enumerate(lst):
                    fused[i] = fused.get(i, 0) + 1 / (60 + r)
            ranked = sorted(fused, key=lambda i: -fused[i])
        if self.rerank is not None:
            cand = ranked[:rerank_top]
            scores = self.rerank.predict([(q, self.store.text(i)) for i in cand], show_progress_bar=False)
            ranked = [cand[j] for j in np.argsort(-np.asarray(scores))]
        return ranked[:k]

    def fmt(self, idx):
        return "\n\n".join(f"[{n + 1}] {t}\n{x}".strip() for n, (t, x) in enumerate(self.store.get(i) for i in idx))

    def doc_ids(self, idx):
        return list(dict.fromkeys(self.doc_list[self.doc_of[i]] for i in idx))


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "build":
    # python ragcore.py build <corpus_dir> [dense_model_dir]   (no LLM needed)
    build(sys.argv[2], INDEX, dense_model=sys.argv[3] if len(sys.argv) > 3 else None)
    raise SystemExit(0)

_task_file = os.path.join(HERE, "task.json")
TASK = json.load(open(_task_file)) if os.path.exists(_task_file) else {}
LLM = os.environ.get("LLM_BASE_URL", "")


def generate(prompt, max_tokens=64):
    import httpx
    r = httpx.post(f"{LLM}/chat/completions", timeout=55, json={
        "model": "generator", "max_tokens": max_tokens, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": prompt}],
    })
    r.raise_for_status()
    return (r.json()["choices"][0]["message"]["content"] or "").strip()


def clean_answer(text):
    text = re.sub(r"^(final answer|answer)\s*:\s*", "", text.strip(), flags=re.I)
    return text.split("\n")[0].strip().strip('"').strip()


def answer_prompt(question, context=None):
    head = f"Answer format: {TASK['answer_format']}.\n"
    if context:
        return (f"Use the passages to answer the question.\n{head}Reply with the answer only.\n\n"
                f"Passages:\n{context}\n\nQuestion: {question}\nAnswer:")
    return f"{head}Reply with the answer only.\n\nQuestion: {question}\nAnswer:"
