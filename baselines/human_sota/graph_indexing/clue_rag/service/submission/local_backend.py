"""Local stand-ins for the three external services the upstream Clue-RAG code talks to.

Upstream (Feesuu/ClueRAG @cb44eaa) expects
  - an OpenAI-compatible embedding server for BGE-M3            (index/embedding.py: EmbeddingModel)
  - a /v1/rerank server for BGE-Reranker-v2-M3                 (index/rerank.py:    RerankModel)
  - an OceanBase (MySQL protocol) server with VECTOR columns    (index/ob_connection.py: OBConnection)
None of them can run inside the AutoRAGBench evaluation sandbox (no network, no daemons, read-only workspace), so
`install()` swaps them for in-process classes with the same interface. The algorithm code (HybridExtraction,
MultiLayerGraph, IterativeRetrieval, Generation) is imported from upstream unchanged.

  LocalEmbedding  BGE-M3 dense vectors (CLS pooling, L2-normalised, fp16), the model the paper uses.
  LocalReranker   BGE-Reranker-v2-M3, sigmoid of the logit (= `normalize=True`).
  LocalVectorDB   the three tables (knowledge units, chunks, entities) as arrays; the only SQL the upstream code
                  issues is CREATE/DROP/count(*)/INSERT and the cosine top-k query
                    SELECT .. ORDER BY cosine_distance(embedding, q) LIMIT k [WHERE JSON_CONTAINS(entity_list_json, e)]
                  which `search()` answers exactly (brute force; OceanBase used an approximate HNSW_SQ index).
                  JSON_CONTAINS on the KU table is an inverted index entity -> KUs (exact string match).
Scope: `SCOPE` (a ContextVar) restricts every search to the rows of one document (QASPER's meta.doc_id).
"""
import contextvars
import json
import os
import pickle
import re
import threading

import numpy as np
import torch

SCOPE = contextvars.ContextVar("cluerag_scope", default=None)  # doc_id or None
_DEV = "cuda" if torch.cuda.is_available() else "cpu"
_MODELS = {}
_LOCK = threading.Lock()   # upstream runs searches / re-ranking in threads; HF fast tokenizers are not thread-safe


def _load(kind, path):
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
    if (kind, path) not in _MODELS:
        tok = AutoTokenizer.from_pretrained(path)
        cls = AutoModel if kind == "emb" else AutoModelForSequenceClassification
        model = cls.from_pretrained(path, torch_dtype=torch.float16 if _DEV == "cuda" else torch.float32)
        _MODELS[(kind, path)] = (tok, model.to(_DEV).eval())
    return _MODELS[(kind, path)]


class LocalEmbedding:
    """index/embedding.py:EmbeddingModel interface: get_embedding(texts) -> float32 [n, 1024], normalised."""
    path = None          # set by install()
    max_length = 2048    # upstream chunks are <= 1200 cl100k tokens
    batch_tokens = 64000

    def __init__(self, global_config=None):
        self.tok, self.model = _load("emb", self.path)

    @torch.inference_mode()
    def get_embedding(self, texts, batch_size=None):
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.zeros((0, self.model.config.hidden_size), dtype=np.float32)
        with _LOCK:
            return self._embed(texts)

    def _embed(self, texts):
        lens = [len(t) for t in texts]
        order = np.argsort(lens)[::-1]
        out = np.zeros((len(texts), self.model.config.hidden_size), dtype=np.float32)
        i = 0
        while i < len(order):  # length-sorted batches of ~batch_tokens tokens
            est = max(8, min(self.max_length, lens[order[i]] // 3 + 8))
            n = max(1, min(512, self.batch_tokens // est))
            idx = order[i:i + n]
            enc = self.tok([texts[j] for j in idx], padding=True, truncation=True, max_length=self.max_length,
                           return_tensors="pt").to(_DEV)
            cls = self.model(**enc).last_hidden_state[:, 0].float()
            out[idx] = torch.nn.functional.normalize(cls, dim=-1).cpu().numpy()
            i += n
        return out


class LocalReranker:
    """index/rerank.py:RerankModel interface: compute_score(pairs, normalize=True) -> list[float]."""
    path = None
    max_length = 2048
    batch_tokens = 48000

    def __init__(self, global_config=None):
        self.tok, self.model = _load("rerank", self.path)

    @torch.inference_mode()
    def compute_score(self, pairs, normalize=True):
        if not pairs:
            return []
        with _LOCK:
            return self._score(pairs, normalize)

    def _score(self, pairs, normalize):
        lens = [len(q) + len(d) for q, d in pairs]
        order = np.argsort(lens)[::-1]
        scores = np.zeros(len(pairs), dtype=np.float32)
        i = 0
        while i < len(order):
            est = max(16, min(self.max_length, lens[order[i]] // 3 + 16))
            n = max(1, min(256, self.batch_tokens // est))
            idx = order[i:i + n]
            enc = self.tok([pairs[j][0] for j in idx], [pairs[j][1] for j in idx], padding=True,
                           truncation="only_second", max_length=self.max_length, return_tensors="pt").to(_DEV)
            logit = self.model(**enc).logits.view(-1).float()
            scores[idx] = (torch.sigmoid(logit) if normalize else logit).cpu().numpy()
            i += n
        return scores.tolist()


class _Table:
    def __init__(self, rows=None, emb=None):
        self.rows = rows or []          # dicts without the embedding column
        self.ids = {r["id"] for r in self.rows}
        self.emb = emb                  # np.float16 [n, d]
        self.pending = []
        self._gpu = None
        self._inv = None                # entity -> row indices (KU table)
        self._scope_cache = {}

    def flush(self):
        if self.pending:
            new = np.stack(self.pending).astype(np.float16)
            self.emb = new if self.emb is None else np.concatenate([self.emb, new])
            self.pending = []
            self._gpu = None

    def gpu(self):
        if self._gpu is None:
            self.flush()
            self._gpu = torch.from_numpy(self.emb).to(_DEV)
            if _DEV == "cpu":
                self._gpu = self._gpu.float()
        return self._gpu

    def inverted(self):
        if self._inv is None:
            inv = {}
            for i, r in enumerate(self.rows):
                for e in json.loads(r["entity_list_json"]):
                    inv.setdefault(e, []).append(i)
            self._inv = {e: np.array(v) for e, v in inv.items()}
        return self._inv


class LocalVectorDB:
    """index/ob_connection.py:OBConnection interface, backed by in-memory tables persisted under STORE_DIR."""
    STORE_DIR = None
    TABLES = {}
    SCOPES = None      # ScopeIndex, set by the query-time pipeline

    def __init__(self, config=None):
        self.current_db = None

    # --- connection protocol used by upstream ---
    def connect(self, database=None):
        self.current_db = database

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def execute(self, sql, params=None, is_many=False):
        s = " ".join(sql.split())
        u = s.upper()
        if u.startswith("SELECT COUNT(*)"):
            t = re.search(r"FROM (\w+)", s).group(1)
            return [{"c": len(self.TABLES[t].rows) if t in self.TABLES else 0}]
        if u.startswith("DROP TABLE"):
            self.TABLES.pop(s.split()[-1], None)
            return 0
        if u.startswith("CREATE TABLE"):
            self.TABLES.setdefault(re.search(r"EXISTS (\w+)", s).group(1), _Table())
            return 0
        if u.startswith("CREATE DATABASE") or "VECTOR INDEX" in u:
            return 0   # exact search needs no ANN index
        raise NotImplementedError(f"LocalVectorDB: unsupported SQL {s[:120]}")

    def batch_insert_dicts(self, table, data_list, chunk_size=1000):  # INSERT IGNORE semantics
        t = self.TABLES.setdefault(table, _Table())
        n = 0
        for d in data_list:
            if d["id"] in t.ids:
                continue
            t.ids.add(d["id"])
            emb = d["embedding"]
            t.pending.append(np.asarray(json.loads(emb) if isinstance(emb, str) else emb, dtype=np.float16))
            t.rows.append({k: v for k, v in d.items() if k != "embedding"})
            n += 1
        t.flush()
        return n

    # --- persistence ---
    @classmethod
    def save_all(cls, store_dir):
        os.makedirs(store_dir, exist_ok=True)
        for name, t in cls.TABLES.items():
            t.flush()
            np.save(os.path.join(store_dir, f"{name}.npy"), t.emb)
            with open(os.path.join(store_dir, f"{name}.rows.pkl"), "wb") as f:
                pickle.dump(t.rows, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_all(cls, store_dir):
        for fn in os.listdir(store_dir):
            if fn.endswith(".rows.pkl"):
                name = fn[:-len(".rows.pkl")]
                rows = pickle.load(open(os.path.join(store_dir, fn), "rb"))
                cls.TABLES[name] = _Table(rows, np.load(os.path.join(store_dir, f"{name}.npy")))

    # --- the cosine top-k query of IterativeRetrieval._execute_vector_search_async ---
    @classmethod
    def search(cls, table, vector, top_k, output_fields, filter_condition=None):
        t = cls.TABLES[table]
        cand = None
        if filter_condition:
            m = re.fullmatch(r"JSON_CONTAINS\(entity_list_json, '(.*)'\)", filter_condition, flags=re.S)
            if not m:
                raise NotImplementedError(filter_condition)
            ent = json.loads(m.group(1).replace("''", "'").replace("\\\\", "\\"))
            cand = t.inverted().get(ent)
            if cand is None:
                return []
        scope = SCOPE.get()
        if scope is not None and cls.SCOPES is not None:
            allowed = cls.SCOPES.rows(table, scope)
            cand = allowed if cand is None else np.intersect1d(cand, allowed)
            if len(cand) == 0:
                return []
        E = t.gpu()
        q = torch.as_tensor(np.asarray(vector, dtype=np.float32).ravel(), device=E.device).to(E.dtype)
        if cand is None:
            sims = E @ q
            k = min(top_k, sims.shape[0])
            val, idx = torch.topk(sims.float(), k)
            idx, val = idx.cpu().numpy(), val.cpu().numpy()
        else:
            ci = torch.as_tensor(cand, device=E.device)
            sims = E.index_select(0, ci) @ q
            k = min(top_k, sims.shape[0])
            val, pos = torch.topk(sims.float(), k)
            idx, val = cand[pos.cpu().numpy()], val.cpu().numpy()
        out = []
        for i, v in zip(idx.tolist(), val.tolist()):
            r = t.rows[i]
            d = {}
            for f in output_fields:
                if f == "embedding":
                    d[f] = t.emb[i].astype(np.float32)
                elif f == "id" and "chunk_id_raw" in r:
                    # Upstream direct chunk retrieval requests `id`, while
                    # corpora_data is keyed by the original `chunk_id_raw`.
                    # Return the original ID so direct and graph candidates
                    # share the same namespace and can be reranked together.
                    d[f] = r["chunk_id_raw"]
                else:
                    d[f] = r[f]
            d["dist"] = 1.0 - v
            out.append(d)
        return out


class ScopeIndex:
    """Rows of each table that belong to one document (QASPER meta.doc_id)."""

    def __init__(self, tables, chunk2doc, ku2chunkids):
        self.tables, self.chunk2doc, self.ku2chunkids = tables, chunk2doc, ku2chunkids
        self.cache = {}

    def rows(self, table, doc):
        key = (table, doc)
        if key not in self.cache:
            t = LocalVectorDB.TABLES[table]
            if table == self.tables["chunk"]:
                idx = [i for i, r in enumerate(t.rows) if self.chunk2doc.get(r["chunk_id_raw"]) == doc]
            elif table == self.tables["ku"]:
                idx = [i for i, r in enumerate(t.rows)
                       if any(self.chunk2doc.get(c) == doc for c in self.ku2chunkids.get(r["text"], ()))]
            else:
                ku = LocalVectorDB.TABLES[self.tables["ku"]]
                ents = set()
                for i in self.rows(self.tables["ku"], doc):
                    ents.update(json.loads(ku.rows[i]["entity_list_json"]))
                idx = [i for i, r in enumerate(t.rows) if r["entity_name"] in ents]
            self.cache[key] = np.array(idx, dtype=np.int64)
        return self.cache[key]


def install(models_dir):
    """Point the upstream modules at the local classes (call after `sys.path` holds the upstream root)."""
    import index.construction
    import index.embedding
    import index.hybrid_extraction
    import index.ob_connection
    import index.rerank
    import retrieval.retrieval
    LocalEmbedding.path = os.path.join(models_dir, "bge-m3")
    LocalReranker.path = os.path.join(models_dir, "bge-reranker-v2-m3")
    for mod in (index.embedding, index.hybrid_extraction, index.construction, retrieval.retrieval):
        mod.EmbeddingModel = LocalEmbedding
    for mod in (index.rerank, retrieval.retrieval):
        mod.RerankModel = LocalReranker
    for mod in (index.ob_connection, index.construction, retrieval.retrieval):
        mod.OBConnection = LocalVectorDB
