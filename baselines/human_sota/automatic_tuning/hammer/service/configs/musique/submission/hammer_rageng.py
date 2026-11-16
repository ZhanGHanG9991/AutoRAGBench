"""AutoRAGBench adapter for HAMMER (Cyan9061/HAMMER @679514d), shared by the search and the frozen service.

The upstream package is imported unchanged. This module only
  * routes the pipeline LLM (query decomposition + answer) to the fixed generator behind the AutoRAGBench
    gateway (thinking off, upstream payload otherwise) and the memory LLM to DeepSeek-flash,
  * presents AutoRAGBench data in the upstream SimpleDataset format and adds the task's answer-format line,
  * restricts QASPER retrieval to meta.doc_id,
  * scores every rollout with the task's native metric instead of HAMMER's own F1,
  * caches chunked/embedded indexes on disk and answers dense queries from one numpy matrix
    (same cosine ranking as llama-index's SimpleVectorStore, which rebuilds the matrix per query).
Call setup() before importing anything from `hammer`.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(os.environ.get("AUTORAGBENCH_ROOT", Path(__file__).resolve().parents[1]))
UPSTREAM_COMMIT = "679514dc40779ecb4c56d175bef660bd616c62e6"
GENERATOR_WORKERS = 16  # concurrent generator requests in the search (upstream MODEL_MAXWORKERS = 6)

_tls = threading.local()   # QASPER: paper scope of the sub-query retrieved in this thread
_state = {"metas": None, "task": None, "hint": "", "cache_dir": None, "llm_base": None}
_lock = threading.Lock()


def setup(upstream: Path, llm_base_url: str, cache_dir: Path, task: str):
    """Environment and patches that must precede `import hammer`."""
    parsed = urlsplit(llm_base_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError(f"HAMMER requires a loopback-only LLM gateway, got: {llm_base_url!r}")
    import dotenv
    dotenv.load_dotenv = lambda *a, **k: False   # the repository .env only holds placeholder keys
    os.environ["REPO_ROOT"] = str(upstream)
    # utils_getAPI sizes its thread pool by the number of keys; the keys themselves are never sent anywhere
    os.environ["SILICONFLOW_API_KEYS"] = ",".join(["generator-gateway"] * GENERATOR_WORKERS)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.path.insert(0, str(upstream))
    _state.update(llm_base=llm_base_url.rstrip("/"), cache_dir=Path(cache_dir), task=task)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    info = json.loads((ROOT / "tasks" / task / "info.json").read_text()) if (ROOT / "tasks" / task).exists() \
        else json.loads((upstream.parent / "submission" / "task.json").read_text())
    _state["info"] = info
    _state["hint"] = "Answer format: " + info["answer_format"] + "."
    # hammer.configuration chmods its (pre-created) directories at import; the code tree is read-only here.
    import errno
    import pathlib
    chmod = pathlib.Path.chmod

    def tolerant_chmod(self, mode, *, follow_symlinks=True):
        try:
            return chmod(self, mode, follow_symlinks=follow_symlinks)
        except OSError as e:
            if e.errno not in (errno.EROFS, errno.EPERM):
                raise

    pathlib.Path.chmod = tolerant_chmod
    import hammer.configuration  # noqa: F401
    pathlib.Path.chmod = chmod
    _patch_compat()
    _patch_generator()
    _patch_indexes()
    _patch_prompt_builder()


# ---------------------------------------------------------------- library compatibility

def _patch_compat():
    """llama-index-core 0.10.61 (pinned upstream) uses the pydantic v1 bridge, whose metaclass only collects
    PrivateAttr declared on model classes; EmbeddingTimeoutMixin declares them on a plain mixin. Same class,
    with the mixin's private attributes declared on the model itself."""
    import typing as T
    from hammer import huggingface_helper as hh
    from hammer.embeddings import timeouts
    from llama_index.core.bridge.pydantic import PrivateAttr
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    # initial values as assigned by the mixin's __init__ / reset_timeouts (timeouts are inactive upstream)
    defaults = {"_chunks_processed": 0, "_chunks_remaining": None, "_done_building_index": False,
                "_seconds_per_chunk": None, "_time_remaining": None, "_time_start": None,
                "_timeout_config": timeouts.TimeoutConfig(), "_total_chunks": 0, "_total_time": 0.0,
                "_warning_logged": False}
    body = {n: PrivateAttr(default=v) for n, v in defaults.items()}
    body["__annotations__"] = {n: T.Any for n in defaults}
    body["__module__"] = hh.__name__
    hh.HuggingFaceEmbeddingWithTimeout = type("HuggingFaceEmbeddingWithTimeout",
                                              (timeouts.EmbeddingTimeoutMixin, HuggingFaceEmbedding), body)

    # llama-index-core 0.10.61 KVDocumentStore._merge_ref_doc_kv_pairs repeats its whole loop once per ref doc
    # (O(R^2): a 30k-document corpus never finishes in the BM25 path). The outer loop rebuilds the same list each
    # time, so one pass returns exactly the same pairs (as fixed in later llama-index releases).
    from llama_index.core.storage.docstore.keyval_docstore import KVDocumentStore

    def merge_ref_doc_kv_pairs(self, ref_doc_kv_pairs):
        merged = []
        for key, kv_pairs in ref_doc_kv_pairs.items():
            node_ids, seen, metadata = [], set(), {}
            for kv_pair in kv_pairs:
                for node in kv_pair[1].get("node_ids", []):
                    if node not in seen:
                        seen.add(node)
                        node_ids.append(node)
                metadata.update(kv_pair[1].get("metadata", {}))
            merged.append((key, {"node_ids": node_ids, "metadata": metadata}))
        return merged

    KVDocumentStore._merge_ref_doc_kv_pairs = merge_ref_doc_kv_pairs

    # Resolve the Hugging Face model ID to the prepared offline model.
    # Formal evaluation starts the frozen workspace inside a restricted
    # sandbox. Prefer the model copied into that workspace instead of
    # reaching into a different HumanSOTA checkout outside the sandbox.
    local_bge_m3 = Path(
        os.environ.get(
            "HAMMER_LOCAL_BGE_M3",
            str(ROOT / "humansota/ClueRAG/models/bge-m3"),
        )
    )
    if not (local_bge_m3 / "config.json").is_file():
        raise FileNotFoundError(f"local bge-m3 is incomplete: {local_bge_m3}")
    author_paths = {
        "BAAI/bge-m3": str(local_bge_m3),
    }
    get_hf = hh.get_hf_embedding_model
    hh.get_hf_embedding_model = lambda name, *a, **k: get_hf(author_paths.get(name, name), *a, **k)


# ---------------------------------------------------------------- generator (pipeline LLM)

def _patch_generator():
    import requests
    from hammer.utils import utils_getAPI

    # Fail closed: HAMMER does not need any HTTP endpoint other than the local
    # AutoRAGBench gateway.  This also turns accidental Hugging Face/API lookups
    # into immediate, auditable failures instead of silently using the network.
    original_request = requests.sessions.Session.request
    if not getattr(original_request, "_rageng_loopback_only", False):
        def loopback_only_request(session, method, url, *args, **kwargs):
            host = urlsplit(str(url)).hostname
            if host not in {"127.0.0.1", "localhost", "::1"}:
                raise RuntimeError(f"blocked non-loopback HTTP request from HAMMER: {url}")
            return original_request(session, method, url, *args, **kwargs)

        loopback_only_request._rageng_loopback_only = True
        requests.sessions.Session.request = loopback_only_request

    def predict_single(input_text, api_key, model_name, max_retries=3):
        """utils_getAPI.predict_single with the endpoint replaced by the AutoRAGBench gateway.
        Same payload as upstream (max_tokens 1024, temperature 0, top_p 0.9); Qwen3.5 thinking off."""
        payload = {"model": "generator", "messages": [{"role": "user", "content": input_text}],
                   "stream": False, "max_tokens": 1024, "temperature": 0, "top_p": 0.9,
                   "chat_template_kwargs": {"enable_thinking": False}}
        start = time.time()
        for attempt in range(max_retries):
            try:
                r = requests.post(_state["llm_base"] + "/chat/completions", json=payload, timeout=60)
                if r.status_code == 200:
                    body = r.json()
                    content = (body["choices"][0]["message"].get("content") or "").strip()
                    return (input_text, content, round((time.time() - start) * 1000),
                            body.get("usage", {}).get("total_tokens", 0))
                if r.status_code in (429, 403):   # per-question budget or closed window: never retry
                    return (input_text, None, None, None)
                if r.status_code < 500:
                    return (input_text, None, None, None)
            except requests.exceptions.RequestException:
                pass
            time.sleep(1.5 ** attempt)
        return (input_text, None, None, None)

    def batch_predict(input_texts, model_name, max_workers=None):
        """Local-only replacement for upstream's SiliconFlow-labelled helper.

        The upstream symbolic model name remains part of HAMMER's search
        configuration, but every generation is served by the fixed local
        Qwen3.5-4B gateway.
        """
        workers = min(max_workers or GENERATOR_WORKERS, GENERATOR_WORKERS)
        print(f"AutoRAGBench local batch generator | requested={model_name} | "
              f"served=Qwen3.5-4B | workers={workers}", flush=True)
        results = [None] * len(input_texts)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(predict_single, text, "generator-gateway", "generator"): i
                for i, text in enumerate(input_texts)
            }
            for future in as_completed(futures):
                i = futures[future]
                try:
                    results[i] = future.result()
                except Exception:
                    results[i] = (input_texts[i], None, None, None)
        return results

    utils_getAPI.predict_single = predict_single
    utils_getAPI.batch_predict = batch_predict
    utils_getAPI.get_provider_for_model = lambda model_name: "rageng-local"
    utils_getAPI.get_api_keys_for_model = lambda model_name: ["generator-gateway"] * GENERATOR_WORKERS

    # batch_api_evaluator imports these functions directly, so replace its
    # module-level references too; patching utils_getAPI alone is not enough
    # if it was imported earlier by hammer.configuration.
    from hammer.utils import batch_api_evaluator
    batch_api_evaluator.batch_predict = batch_predict
    batch_api_evaluator.get_api_keys_for_model = utils_getAPI.get_api_keys_for_model


def call_generator(prompt):
    from hammer.utils import utils_getAPI
    return utils_getAPI.predict_single(prompt, "generator-gateway", "generator")


# ---------------------------------------------------------------- data

def write_upstream_dataset(task, rows, corpus_path, out_dir):
    """AutoRAGBench public files -> the list/JSONL layout SimpleDataset reads (id, title, text / question, answer)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus = out_dir / "corpus.json"
    if not corpus.exists():
        docs = [json.loads(line) for line in open(corpus_path)]
        # Preserve the formal-v2 title verbatim.  PubMedQA titles are empty;
        # replacing them with doc_id makes otherwise verbatim evidence fail
        # the fixed-evidence corpus contract.
        corpus.write_text(json.dumps([{"id": d["doc_id"], "title": d.get("title") or "", "text": d["text"]}
                                      for d in docs], ensure_ascii=False))
    qa = out_dir / "qa.jsonl"
    with qa.open("w") as f:
        for r in rows:
            f.write(json.dumps({"id": r["qid"], "question": r["question"], "answer": r["answers"][0],
                                "text_ground_truth": r.get("supporting_doc_ids", [])}, ensure_ascii=False) + "\n")
    return corpus, qa


def dataset_description(task):
    info = _state["info"]
    return (f"AutoRAGBench {info['display_name']}: {info['question_type']}. Answer format: {info['answer_format']}. "
            f"The reported score is the task's native metric ({info['primary_metric']}: {info['metric_description']}), "
            f"recorded in the f1_score fields. The answer and query-decomposition model is fixed (Qwen3.5-4B) and "
            f"may answer at most 4 generator calls per question."
            + (" Retrieval is restricted to the paper named by meta.doc_id." if task == "qasper" else ""))


def set_question_metas(metas):
    _state["metas"] = metas


def native_scores(task, answers, golds):
    """Per-question native metric; a missing response scores 0 as in tasks.score."""
    from tasks import registry
    mod, primary = registry.load(task), registry.info(task)["primary_metric"]
    return [mod.score_one(a, g)[primary] if a is not None else 0.0 for a, g in zip(answers, golds)]


def final_answer(response):
    from hammer.multihop_evaluation import extract_answer_from_prediction
    return extract_answer_from_prediction(response or "").strip()


# ---------------------------------------------------------------- indexes

_splitter_params = {}   # id(splitter) -> the parameters build_splitter was called with


def _splitter_key(transforms):
    parts = []
    for t in transforms:
        inner = getattr(t, "_lc_splitter", None)
        if inner is not None:   # LangchainNodeParser(RecursiveCharacterTextSplitter), tokenizer = embedding model's
            parts.append(("recursive", inner._chunk_size, inner._chunk_overlap))
        else:
            parts.append((type(t).__name__, getattr(t, "chunk_size", None), getattr(t, "chunk_overlap", None)))
        parts.append(_splitter_params.get(id(t)))
    return parts


def _key(*parts):
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:20]


def _chunks(documents, transforms, extra_key):
    """Upstream IngestionPipeline output, cached on disk per (splitter, chunk size, overlap)."""
    from llama_index.core.ingestion import IngestionPipeline
    from llama_index.core.schema import TextNode
    key = _key(_splitter_key(transforms), extra_key)
    path = _state["cache_dir"] / f"chunks-{key}.jsonl"
    _state.setdefault("used", set()).add(path.name)
    if not path.exists() and _state.get("require_cache"):
        raise RuntimeError("index cache miss in the frozen service: " + path.name)
    if path.exists():
        return [TextNode.from_dict(json.loads(line)) for line in open(path)], key
    nodes = IngestionPipeline(transformations=transforms).run(documents=documents, show_progress=False)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        for n in nodes:
            f.write(json.dumps(n.to_dict(), ensure_ascii=False) + "\n")
    tmp.replace(path)
    (_state["cache_dir"] / f"chunks-{key}.json").write_text(json.dumps({"splitter": _splitter_key(transforms)}, default=str))
    return nodes, key


_matrices = {}   # id(SimpleVectorStore) -> _MatrixStore (pydantic-v1 stores take no extra attributes)
_bm25_nodes = {}  # id(BM25Retriever) -> its nodes (for QASPER per-paper BM25)


def _remember(registry, key, value, keep=2):
    registry[key] = value
    while len(registry) > keep:   # one flow is alive at a time; older indexes are released
        registry.pop(next(iter(registry)))


class _MatrixStore:
    """Normalised embedding matrix kept next to llama-index's SimpleVectorStore (same cosine scores)."""

    def __init__(self, ids, emb):
        import numpy as np
        self.ids = ids
        self.pos = {i: k for k, i in enumerate(ids)}
        e = np.asarray(emb, dtype=np.float32)
        self.mat = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)


def _patch_indexes():
    import numpy as np
    from hammer.retrievers import build
    from llama_index.core import VectorStoreIndex
    from llama_index.core.storage.storage_context import StorageContext
    from llama_index.core.vector_stores import SimpleVectorStore
    from llama_index.core.vector_stores.types import VectorStoreQueryResult

    def build_dense(documents, transforms, embedding_model, max_chunks=0):
        nodes, ckey = _chunks(documents, transforms, None)
        name = getattr(embedding_model, "model_name", str(embedding_model))
        path = _state["cache_dir"] / f"emb-{_key(ckey, name)}.npy"
        _state.setdefault("used", set()).add(path.name)
        if not path.exists() and _state.get("require_cache"):
            raise RuntimeError("embedding cache miss in the frozen service: " + path.name)
        if path.exists():
            emb = np.load(path)
        else:
            from llama_index.core.schema import MetadataMode
            texts = [n.get_content(metadata_mode=MetadataMode.EMBED) for n in nodes]
            emb = np.asarray(embedding_model.get_text_embedding_batch(texts, show_progress=False), dtype=np.float32)
            np.save(path.with_suffix(".tmp.npy"), emb)
            os.replace(path.with_suffix(".tmp.npy"), path)
        for n, e in zip(nodes, emb):
            n.embedding = e.tolist()
        index = VectorStoreIndex(nodes=nodes, embed_model=embedding_model, insert_batch_size=2048, show_progress=False)
        _remember(_matrices, id(index.vector_store), _MatrixStore([n.node_id for n in nodes], emb))
        for n in nodes:
            n.embedding = None
        return index, index.docstore

    def build_sparse(documents, transforms, top_k):
        nodes, _ = _chunks(documents, transforms, None)
        docstore = StorageContext.from_defaults().docstore
        docstore.add_documents(nodes)
        retriever = build._create_bm25_retriever({"nodes": list(docstore.docs.values()), "similarity_top_k": top_k})
        _remember(_bm25_nodes, id(retriever), list(docstore.docs.values()))
        return retriever, docstore

    original_query = SimpleVectorStore.query

    def query(self, query, **kwargs):
        m = _matrices.get(id(self))
        if m is None or query.query_embedding is None or (query.mode and str(query.mode) not in ("default", "VectorStoreQueryMode.DEFAULT")):
            return original_query(self, query, **kwargs)
        q = np.asarray(query.query_embedding, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-12)
        rows = None
        if query.filters is not None:   # only our QASPER scope filter: metadata "id" == doc_id
            allowed = set(query.node_ids or []) or None
            rows = np.asarray([m.pos[i] for i in (allowed or [])], dtype=np.int64)
        sims = m.mat @ q if rows is None else m.mat[rows] @ q
        k = min(query.similarity_top_k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k] if k else []
        top = sorted(top, key=lambda j: -sims[j])
        ids = [m.ids[j] if rows is None else m.ids[rows[j]] for j in top]
        return VectorStoreQueryResult(similarities=[float(sims[j]) for j in top], ids=ids)

    original_splitter = build.build_splitter

    def build_splitter(study_config, params):
        splitter = original_splitter(study_config, params)
        _splitter_params[id(splitter)] = [params["splitter_method"], int(params["splitter_chunk_exp"]),
                                          float(params["splitter_chunk_overlap_frac"]),
                                          params.get("rag_embedding_model") if params["splitter_method"] == "recursive" else None]
        return splitter

    build.build_splitter = build_splitter
    build._build_dense_index = build_dense
    build._build_sparse_index = build_sparse
    SimpleVectorStore.query = query


# ---------------------------------------------------------------- prompt builder (task hint, QASPER scope)

def _patch_prompt_builder():
    from hammer.utils import optimized_rag_prompt_builder as orpb

    class RagEngPromptBuilder(orpb.OptimizedRAGPromptBuilder):
        def __init__(self, flow, max_workers=None):
            super().__init__(flow, max_workers)
            if _state["task"] == "qasper":
                self.retriever._retrievers = [_ScopedRetriever(r) for r in self.retriever._retrievers]

        def batch_build_prompts(self, queries, is_coreset=False, coreset_size=100, train_data_size=0):
            metas = _state["metas"] or [{} for _ in queries]
            assert len(metas) == len(queries), "question metas out of sync with the evaluated questions"
            res = super().batch_build_prompts(queries, is_coreset, coreset_size, train_data_size)
            res.final_prompts = self._batch_build_final_prompts([q + "\n" + _state["hint"] for q in queries],
                                                                res.context_strs)
            return res

        def _two_stage_decomposition_retrieval(self, queries):
            """Upstream _two_stage_decomposition_retrieval (batch decomposition, flattened parallel sub-query
            retrieval, regroup + fuse). QASPER only: every sub-query is retrieved within its question's paper."""
            if _state["task"] != "qasper":
                return super()._two_stage_decomposition_retrieval(queries)
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from llama_index.core.schema import QueryBundle
            from hammer.utils.batch_api_evaluator import get_batch_llm_caller
            params = self.flow.params or {}
            model = params.get("rag_query_decomposition_llm_name", params.get("query_decomposition_llm", "Qwen2-7b"))
            caller = get_batch_llm_caller(model_name=model, max_workers=self.max_workers)
            decomposed = caller.batch_query_decomposition(queries, num_queries=self._get_num_queries())
            flat, mapping = [], []
            for orig, subs in enumerate(decomposed):
                for sub_idx, sub in enumerate(subs if subs else [queries[orig]]):
                    flat.append(sub)
                    mapping.append((orig, sub_idx))
            metas = _state["metas"]

            def retrieve(i):
                _tls.scope = metas[mapping[i][0]]["doc_id"]
                try:
                    return self._direct_retrieve_without_decomposition(QueryBundle(flat[i]))
                finally:
                    _tls.scope = None

            results = [[] for _ in flat]
            with ThreadPoolExecutor(max_workers=32) as pool:
                futures = {pool.submit(retrieve, i): i for i in range(len(flat))}
                for f in as_completed(futures):
                    try:
                        results[futures[f]] = f.result()
                    except Exception:
                        results[futures[f]] = []
            return self._regroup_and_fuse_results(results, mapping, len(queries))

    orpb.create_optimized_rag_prompt_builder = lambda flow, max_workers=None: RagEngPromptBuilder(flow, max_workers)
    orpb.RagEngPromptBuilder = RagEngPromptBuilder


def _ScopedRetriever(inner):
    """Wrap a sub-retriever so that it only returns chunks of the paper in _tls.scope (QASPER)."""
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.vector_stores.types import ExactMatchFilter, MetadataFilters

    class Scoped(BaseRetriever):
        def __init__(self):
            super().__init__()
            self.inner = inner
            self._similarity_top_k = getattr(inner, "_similarity_top_k", 10)
            self._bm25 = {}
            self._ids = None
            self._lock = threading.Lock()

        def _retrieve(self, query_bundle):
            scope = getattr(_tls, "scope", None)
            if scope is None:   # fail closed: QASPER retrieval is never global
                raise RuntimeError("QASPER retrieval without a paper scope")
            if id(self.inner) in _bm25_nodes:   # BM25 over the paper's chunks
                with self._lock:   # sub-queries arrive from parallel threads
                    if scope not in self._bm25:
                        from hammer.retrievers.build import _create_bm25_retriever
                        nodes = [n for n in _bm25_nodes[id(self.inner)] if n.metadata.get("id") == scope]
                        self._bm25[scope] = _create_bm25_retriever(
                            {"nodes": nodes, "similarity_top_k": min(self._similarity_top_k, len(nodes))}) if nodes else None
                    r = self._bm25[scope]
                return r.retrieve(query_bundle) if r is not None else []
            with self._lock:
                if self._ids is None:   # VectorIndexRetriever over our cached dense index
                    ids_by_doc = {}
                    for n in self.inner._index.docstore.docs.values():
                        ids_by_doc.setdefault(n.metadata.get("id"), []).append(n.node_id)
                    self._ids = ids_by_doc
            ids = self._ids.get(scope, [])
            if not ids:
                return []
            # a fresh retriever per call: sub-queries run in parallel threads, so the shared one is never mutated
            from llama_index.core.retrievers import VectorIndexRetriever
            scoped = VectorIndexRetriever(index=self.inner._index, similarity_top_k=self.inner._similarity_top_k,
                                          embed_model=self.inner._embed_model, node_ids=ids,
                                          filters=MetadataFilters(filters=[ExactMatchFilter(key="id", value=scope)]))
            return scoped.retrieve(query_bundle)

    return Scoped()


# ---------------------------------------------------------------- memory LLM (local Qwen3.5-4B)

class LocalQwenClient:
    """OpenAI-compatible local Qwen client for HAMMER memory and simulation.

    This is an explicit campaign deviation from both the existing DeepSeek
    adaptation and the paper's Qwen2.5-72B memory model.
    """

    MODEL = "generator"

    def __init__(self, log_path):
        import openai
        self._client = openai.OpenAI(api_key="EMPTY", base_url=os.environ["LLM_BASE_URL"],
                                     timeout=120, max_retries=3)
        self._log = Path(log_path)
        self.chat = self
        self.completions = self
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "seconds": 0.0}

    def create(self, model=None, messages=None, **kwargs):
        start = time.time()
        kwargs.pop("timeout", None)
        response = self._client.chat.completions.create(
            model=self.MODEL, messages=messages,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}, **kwargs)
        u = response.usage
        cached = getattr(u, "prompt_cache_hit_tokens", None) or 0
        with _lock:
            self.usage["calls"] += 1
            self.usage["prompt_tokens"] += u.prompt_tokens
            self.usage["completion_tokens"] += u.completion_tokens
            self.usage["cached_tokens"] += cached
            self.usage["seconds"] += time.time() - start
            with self._log.open("a") as f:
                f.write(json.dumps({"requested_model": model, "kwargs": kwargs, "messages": messages,
                                    "content": response.choices[0].message.content,
                                    "usage": u.model_dump(), "seconds": time.time() - start}, ensure_ascii=False) + "\n")
        return response


# Kept only so old audit scripts fail visibly less often if imported.  Formal
# campaign code uses LocalQwenClient by name and records the deviation.
DeepSeekClient = LocalQwenClient
