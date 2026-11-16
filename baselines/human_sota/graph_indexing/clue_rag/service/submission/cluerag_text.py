"""Clue-RAG query time (Q-Iter + generation) for one question, with the upstream code (Feesuu/ClueRAG @cb44eaa).

Per question, exactly what upstream IterativeRetrieval.query_async + Generation do for a batch:
  1. query NER: one generator call (NER_PROMPT + NER_SCHEMA, upstream `_ner_by_llm_async`)
  2. upstream `_process_single_query` unchanged:
       knowledge anchoring  top 10*M KUs by cosine, re-ranked with the question, entities of the top M
       entity linking       3 nearest entity nodes per query entity
       iterative retrieval  D rounds: per frontier entity the top-K KUs mentioning it (cosine to the *updated*
                            query vector q - ku), paths re-ranked with the question, beam of M entities
       final re-ranking     chunks of all KUs on the paths, re-ranked, top N
     (its direct chunk-vector branch returns hashed ids that are not chunk ids, so upstream drops those chunks;
      kept as is -- the paper's Q-Iter reaches chunks only through KUs)
  3. upstream Generation (GENERATION_PROMPT, one-shot "Thought: ... Answer: ...") on the top N chunks: one call.
Hyper-parameters are the paper's / upstream defaults: K=3, M=5, D=3, N=5 (utils/config.py).

AutoRAGBench additions (same as for every system in the benchmark):
  - the task's answer-format line is appended to GENERATION_PROMPT; the answer is the text after "Answer:".
  - QASPER (meta.doc_id): every search is restricted to the KUs / entities / chunks of that paper.
  - max_tokens 1024 per call (upstream 4096) to stay inside the 60 s per-question budget.
2 generator calls per question (budget: 4).
"""
import asyncio
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, "cluerag_upstream"), os.path.dirname(HERE)):  # packaged workspace / repo checkout
    if os.path.isdir(os.path.join(_p, "retrieval")):
        UPSTREAM = _p
        break
sys.path.insert(0, UPSTREAM)
sys.path.insert(0, HERE)

import local_backend  # noqa: E402
from local_backend import SCOPE, LocalVectorDB, ScopeIndex  # noqa: E402


class _Graph:  # the attributes IterativeRetrieval reads from MultiLayerGraph
    def __init__(self, name, ku2chunkids):
        self.db_name = "clueragdb"
        self.table_ku = f"{name}_knowledge_units"
        self.table_chunk = f"{name}_chunks"
        self.table_entity = f"{name}_entities"
        self.ku2chunkids = ku2chunkids


class _ScopedChunks(dict):
    """corpora_data as upstream uses it (`cid in corpora_data`, `corpora_data[cid]`), restricted to SCOPE."""

    def __contains__(self, cid):
        if not dict.__contains__(self, cid):
            return False
        scope = SCOPE.get()
        return scope is None or dict.__getitem__(self, cid)["doc_id"] == scope


class _Dataset:
    def __init__(self, chunks):
        self.corpora_data = chunks
        self.questions_data = {}


def parse_answer(text):
    m = list(re.finditer(r"answer\s*:", text, flags=re.I))
    ans = text[m[-1].end():] if m else text
    lines = [l.strip() for l in ans.strip().split("\n") if l.strip()]
    if not lines:
        return text.strip().split("\n")[-1].strip() if text.strip() else ""
    return lines[0].strip().strip('"').strip()


class ClueRAG:
    def __init__(self, index_dir, models_dir, llm_base_url, task_info, work_dir, max_tokens=1024, api_key="EMPTY"):
        self.trace = []
        os.makedirs(work_dir, exist_ok=True)
        os.chdir(work_dir)  # upstream logger / caches write relative to the cwd; the workspace is read-only
        local_backend.install(models_dir)
        import generation.generation as gen_mod
        from generation.generation import Generation
        from retrieval.retrieval import IterativeRetrieval
        from utils.config import BaseConfig

        stats = json.load(open(os.path.join(index_dir, "index_stats.json")))
        name = f"rageng_{stats['task']}"
        self.cfg = BaseConfig(dataset_name=name, save_dir=os.path.join(work_dir, "cluerag_out"), llm_name="generator",
                              llm_base_url=llm_base_url, api_key=api_key, seed=20261022, llm_max_tokens=max_tokens,
                              llm_num_processes=4, alpha=stats["alpha"], select_metric="COSINE")
        LocalVectorDB.load_all(os.path.join(index_dir, "store"))
        # Materialize fixed vector tables before health readiness.
        for table in LocalVectorDB.TABLES.values():
            table.gpu()
        gc = os.path.join(index_dir, "graph_cache.json")  # packaged workspace; index dir: upstream's COSINE_<alpha>/
        if not os.path.exists(gc):
            gc = os.path.join(index_dir, f"COSINE_{stats['alpha']:.2f}", "graph_cache.json")
        graph_cache = json.load(open(gc))
        chunks = _ScopedChunks(json.load(open(os.path.join(index_dir, "chunks.json"))))
        self.chunks = chunks
        g = _Graph(name, graph_cache["ku2chunkids"])
        LocalVectorDB.SCOPES = ScopeIndex({"ku": g.table_ku, "chunk": g.table_chunk, "entity": g.table_entity},
                                          {cid: c["doc_id"] for cid, c in chunks.items()}, g.ku2chunkids)
        fmt = task_info["answer_format"].replace("{", "{{").replace("}", "}}")
        gen_mod.GENERATION_PROMPT = (gen_mod.GENERATION_PROMPT.rstrip("\n")
                                     + f"\nAnswer format (the text after \"Answer: \"): {fmt}.\n")

        class Retrieval(IterativeRetrieval):
            async def _execute_vector_search_async(self, table, vector, top_k, output_fields, filter_condition=None):
                return await asyncio.to_thread(LocalVectorDB.search, table, vector, top_k, output_fields,
                                               filter_condition)

            async def _expand_layer_async(self, queue, origin_question):  # upstream, plus a size trace
                nq, paths = await super()._expand_layer_async(queue, origin_question)
                tr.append({"frontier": len(queue), "paths": len(paths), "next": len(nq)})
                return nq, paths

        class Gen(Generation):
            def save_generation_cache(self, results):  # no per-question cache file
                pass

        tr = self.trace
        self.ds = _Dataset(chunks)
        self.retriever = Retrieval(global_config=self.cfg, graph=g, dataset=self.ds)
        self.generator = Gen(global_config=self.cfg, dataset=self.ds)

    async def _answer(self, question, doc_id):
        token = SCOPE.set(doc_id)
        try:
            self.trace.clear()
            t0 = time.time()
            qid = "q"
            ner, _ = await self.retriever._ner_by_llm_async(texts=[(qid, question)])
            emb = await asyncio.to_thread(self.retriever.embedder.get_embedding, [question])
            res = await self.retriever._process_single_query(
                qid, {"question": question, "answer": "", "ground_truth_chunks": []}, {qid: emb[0]}, ner)
            t1 = time.time()
            out = await self.generator.augmentated_generation(retrieval_results=[res])
            raw = out[0].get("generation", "") if out else ""
            info = {"ner": len(ner.get(qid, [])), "layers": list(self.trace),
                    "chunks": sum(1 for c in res["chunks"] if c in self.chunks),
                    "t_retrieval": round(t1 - t0, 2), "t_total": round(time.time() - t0, 2)}
            return parse_answer(raw or ""), res["chunks"], raw, info
        finally:
            SCOPE.reset(token)

    def answer(self, question, doc_id=None):
        return asyncio.run(self._answer(question, doc_id))

    def doc_ids(self, chunk_ids):
        # upstream may return hashed ids of its direct-chunk branch when no graph path exists (no context); skip
        return list(dict.fromkeys(dict.__getitem__(self.chunks, c)["doc_id"] for c in chunk_ids
                                  if dict.__contains__(self.chunks, c)))
