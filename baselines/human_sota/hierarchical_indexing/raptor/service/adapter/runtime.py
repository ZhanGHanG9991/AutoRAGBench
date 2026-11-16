"""RAPTOR formal-v2 adapter; supports audited QASPER document-id remapping."""
import json
import os
import pickle
import sqlite3
import time
from types import SimpleNamespace
from pathlib import Path
from urllib.parse import urlparse

import httpx
import numpy as np
import tiktoken
from sentence_transformers import SentenceTransformer
from raptor.EmbeddingModels import BaseEmbeddingModel
from raptor.tree_retriever import TreeRetriever, TreeRetrieverConfig

TOKENIZER = tiktoken.get_encoding("cl100k_base")


def generate(messages, max_tokens):
    base = os.environ["LLM_BASE_URL"].rstrip("/")
    parsed = urlparse(base)
    if parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError("RAPTOR requires the local project gateway")
    if not base.endswith("/v1"):
        base += "/v1"
    with httpx.Client(trust_env=False, timeout=55) as client:
        response = client.post(base + "/chat/completions", json={
            "model": "generator", "messages": messages, "temperature": 0,
            "max_tokens": max_tokens, "seed": 224,
            "chat_template_kwargs": {"enable_thinking": False}})
        response.raise_for_status()
        value = response.json()
    answer = value["choices"][0]["message"]["content"]
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Empty generation; cannot substitute a synthetic summary/answer")
    return answer.strip(), value.get("usage", {})


class SBERT(BaseEmbeddingModel):
    def __init__(self, path, device=None):
        self.model = SentenceTransformer(str(path), device=device, local_files_only=True)
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        assert self.parameter_count <= 1_000_000_000

    def encode(self, texts):
        return self.model.encode(texts, batch_size=128, normalize_embeddings=True,
                                 show_progress_bar=False, convert_to_numpy=True)

    def create_embedding(self, text):
        return self.encode([text])[0]


class Engine:
    def __init__(self, root):
        self.root = Path(root)
        self.config = json.loads((self.root / "config.json").read_text())
        self.catalog = json.loads((self.root / "index/catalog.json").read_text())
        mapping = self.root / "index/doc_id_map_v2_to_index.json"
        self.doc_id_map = json.loads(mapping.read_text()) if mapping.exists() else {}
        self.embedding = SBERT(self.root / "models/multi-qa-mpnet-base-cos-v1")
        self.cache = {}
        self.disk = None
        if self.catalog.get("storage") == "sqlite_faiss":
            import faiss
            faiss.omp_set_num_threads(4)
            self.disk = sqlite3.connect(
                f'file:{self.root / "index/nodes.sqlite"}?mode=ro',
                uri=True, check_same_thread=False)
            self.search = faiss.read_index(str(self.root / "index/search.faiss"))
            if hasattr(self.search, "nprobe"):
                self.search.nprobe = 64
            self.vectors = np.memmap(
                self.root / "index/vectors.f32", dtype="float32", mode="r"
            ).reshape(-1, 768)
            self.roots = np.load(self.root / "index/roots.npy")
            assert self.search.ntotal == self.catalog["nodes"] == len(self.vectors)

    def disk_retrieve(self, query, collapse_tree):
        vector = self.embedding.create_embedding(query)
        if collapse_tree:
            k = min(128, self.search.ntotal)
            while True:
                _, ids = self.search.search(vector.reshape(1, -1), k)
                order = [int(i) for i in ids[0] if i >= 0]
                candidate_tokens = sum(len(TOKENIZER.encode(self.disk.execute(
                    "SELECT text FROM nodes WHERE id=?", (idx,)).fetchone()[0])) for idx in order)
                if candidate_tokens > self.config["context_tokens"] or k == self.search.ntotal or len(order) < k:
                    break
                k = min(k * 2, self.search.ntotal)
        else:
            current = self.roots
            order = []
            while len(current):
                scores = np.asarray(self.vectors[current]) @ vector
                best = current[np.argsort(-scores)[:self.config["traversal_top_k"]]]
                order.extend(map(int, best))
                children = []
                for idx in best:
                    children.extend(json.loads(self.disk.execute(
                        "SELECT children FROM nodes WHERE id=?", (int(idx),)).fetchone()[0]))
                current = np.asarray(sorted(set(children)), dtype="int64")
        selected, texts, nodes, tokens = [], [], {}, 0
        for idx in order:
            text, layer, children, docs = self.disk.execute(
                "SELECT text,layer,children,doc_ids FROM nodes WHERE id=?", (idx,)
            ).fetchone()
            cost = len(TOKENIZER.encode(text))
            if tokens + cost > self.config["context_tokens"]:
                break
            tokens += cost
            texts.append(text)
            selected.append({"node_index": idx, "layer_number": layer})
            nodes[idx] = SimpleNamespace(doc_ids=json.loads(docs))
        return "\n\n".join(texts), selected, SimpleNamespace(all_nodes=nodes)

    def tree(self, key):
        if key not in self.cache:
            if len(self.cache) >= 4:
                self.cache.pop(next(iter(self.cache)))
            with open(self.root / "index" / self.catalog["trees"][key], "rb") as f:
                self.cache[key] = pickle.load(f)
        return self.cache[key]

    def retrieve(self, question, meta, collapse_tree=True):
        query = question
        if self.catalog["task"] in ("webqsp", "cwq"):
            query += "\nTopic entities: " + json.dumps(
                meta.get("topic_entities", []), ensure_ascii=False)
        if self.disk is not None:
            return self.disk_retrieve(query, collapse_tree)
        if self.catalog["task"] == "qasper":
            external_key = str(meta.get("doc_id", ""))
            key = self.doc_id_map.get(external_key, external_key)
            if key not in self.catalog["trees"]:
                raise ValueError("QASPER meta.doc_id absent from audited indexed corpus")
        else:
            key = "corpus"
        tree = self.tree(key)
        retriever = TreeRetriever(TreeRetrieverConfig(
            tokenizer=TOKENIZER, embedding_model=self.embedding,
            context_embedding_model="SBERT", top_k=self.config["traversal_top_k"]), tree)
        context, selected = retriever.retrieve(
            query, collapse_tree=collapse_tree, top_k=len(tree.all_nodes),
            max_tokens=self.config["context_tokens"], return_layer_information=True)
        if not collapse_tree:
            context = TOKENIZER.decode(
                TOKENIZER.encode(context)[:self.config["context_tokens"]])
        return context, selected, tree

    def answer(self, question, meta):
        context, selected, tree = self.retrieve(question, meta)
        task = json.loads((self.root / "submission/task.json").read_text())
        answer, usage = generate([
            {"role": "system", "content": "Answer the question using the supplied context. "
             "Return only the answer. " + task["answer_format"]},
            {"role": "user", "content": "Context:\n" + context + "\n\nQuestion: " + question}], 512)
        record = {"time": time.time(), "selected": selected, "usage": usage,
                  "summary_nodes": sum(x["layer_number"] > 0 for x in selected)}
        scratch = Path(os.environ["SCRATCH_DIR"])
        scratch.mkdir(parents=True, exist_ok=True)
        with (scratch / "raptor_retrieval.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        print("RAPTOR_RETRIEVAL " + json.dumps(record), flush=True)
        sources = sorted({d for x in selected for d in getattr(
            tree.all_nodes[x["node_index"]], "doc_ids", [])})
        return {"answer": answer, "citations": sources}
