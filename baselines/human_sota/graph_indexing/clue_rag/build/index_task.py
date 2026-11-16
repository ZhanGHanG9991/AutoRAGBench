"""Clue-Index for one AutoRAGBench task, with the upstream code (Feesuu/ClueRAG @cb44eaa) unchanged:

  Dataset           upstream chunker (tiktoken cl100k, 1200 tokens, 100 overlap) over "Title: ..\nContent: .." docs
  HybridExtraction  knapsack chunk selection under alpha (alpha = 1.0, the paper's default: every chunk goes to the
                    LLM; chunks whose LLM output does not parse fall back to NLTK sentences), LLM knowledge-unit
                    extraction (EXTRACTION_PROMPT + KU_SCHEMA), spaCy en_core_web_trf NER on every KU
  MultiLayerGraph   normalisation, KU / entity / chunk embeddings (BGE-M3) and the three tables

The LLM is the fixed AutoRAGBench generator (Qwen3.5-4B on a local vLLM, thinking off, temperature 0) -- the same
model and rule as for the agents (index-time LLM use only through the fixed generator). The OceanBase tables and the
embedding service are replaced by local_backend (see there). Output in <out>:
  data/<name>_corpus.json, data/<name>.json   upstream input files (corpus; no questions)
  COSINE_1.00/{selection_cache.pkl, extraction_cache.pkl, graph_cache.json}   upstream caches
  store/                                      the three tables (LocalVectorDB)
  chunks.json                                 upstream chunk dict + doc_id of every chunk
  index_stats.json                            sizes, token usage, timings

  python index_task.py --task musique --corpus <corpus_dir> --out <dir> --llm http://127.0.0.1:PORT/v1 --key KEY
"""
import argparse
import asyncio
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
UPSTREAM = os.path.dirname(HERE)
EXTRACT_COUNTS = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--llm", required=True)
    ap.add_argument("--key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--concurrency", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="only the first N corpus documents (smoke test)")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    name = f"rageng_{a.task}"   # not 'hotpotqa' / '2wikimultihopqa': those load the corpus from a question file
    os.makedirs(os.path.join(out, "data"), exist_ok=True)
    docs = []
    for line in open(os.path.join(a.corpus, "corpus.jsonl")):
        d = json.loads(line)
        docs.append({"doc_id": d["doc_id"], "title": d.get("title") or "", "text": d["text"]})
        if a.limit and len(docs) >= a.limit:
            break
    json.dump(docs, open(os.path.join(out, "data", f"{name}_corpus.json"), "w"), ensure_ascii=False)
    json.dump([], open(os.path.join(out, "data", f"{name}.json"), "w"))

    os.chdir(out)  # upstream reads data/<name>*.json and writes its logger under ./outputs relative to the cwd
    sys.path.insert(0, UPSTREAM)
    sys.path.insert(0, HERE)
    import local_backend
    from local_backend import LocalVectorDB
    local_backend.install(os.path.join(UPSTREAM, "models"))
    from dataset.dataclass import Dataset
    from index.construction import MultiLayerGraph
    from index.hybrid_extraction import HybridExtraction
    from utils.config import BaseConfig

    cfg = BaseConfig(dataset_name=name, save_dir=out, llm_name="generator", llm_base_url=a.llm, api_key=a.key,
                     llm_num_processes=a.concurrency, num_processes=16, alpha=a.alpha, select_metric="COSINE")
    LocalVectorDB.STORE_DIR = os.path.join(out, "store")
    if os.path.isdir(LocalVectorDB.STORE_DIR):
        LocalVectorDB.load_all(LocalVectorDB.STORE_DIR)
    t0 = time.time()
    ds = Dataset(global_config=cfg)
    sid2doc = {item["source_id"]: item["doc_id"] for item in ds._data}
    chunks = {cid: {**c, "doc_id": sid2doc[c["source_id"]]} for cid, c in ds.corpora_data.items()}
    json.dump(chunks, open(os.path.join(out, "chunks.json"), "w"), ensure_ascii=False)

    orig = HybridExtraction._extract_batch

    async def counted(self, inputs, use_llm):  # bookkeeping only; calls the upstream method unchanged
        res, md = await orig(self, inputs, use_llm)
        EXTRACT_COUNTS["chunks_llm_parsed" if use_llm else "chunks_nltk"] = len(res)
        if use_llm:
            EXTRACT_COUNTS["chunks_llm_empty"] = sum(1 for r in res if not r["knowledge_units"])
        return res, md
    HybridExtraction._extract_batch = counted
    prev = os.path.join(out, "index_stats.json")
    if os.path.exists(prev):   # extraction cache reused: keep the counts of the run that made it
        EXTRACT_COUNTS.update({k: v for k, v in json.load(open(prev)).items() if k.startswith("chunks_llm") or k == "chunks_nltk"})

    t1 = time.time()
    hybrid = HybridExtraction(global_config=cfg, dataset=ds)
    asyncio.run(hybrid.process())
    t2 = time.time()
    graph = MultiLayerGraph(cfg, hybrid, ds)
    LocalVectorDB.save_all(LocalVectorDB.STORE_DIR)
    t3 = time.time()

    meta = hybrid.res_metadata or {}
    ent_per_ku = [len(v) for v in graph.ku2entities.values()]
    ku_rows = LocalVectorDB.TABLES[graph.table_ku].rows
    stats = {
        "task": a.task, "alpha": a.alpha, "documents": len(ds._data), "chunks": len(ds.corpora_data),
        "chunks_selected_for_llm": len(hybrid.selection_results[2]),
        **EXTRACT_COUNTS,   # chunks whose LLM output parsed / chunks split by NLTK (unselected or failed)
        "knowledge_units_raw": sum(len(k["knowledge_units"]) for k in hybrid.knowledge_units),
        "chunks_without_entities": len(ds.corpora_data) - len(hybrid.chunkid2ku2entity),
        "tables": {k: len(t.rows) for k, t in LocalVectorDB.TABLES.items()},
        "chunk_ku_edges": sum(len(v) for v in graph.ku2chunkids.values()),
        "ku_entity_edges": sum(ent_per_ku),
        "kus_without_entities": sum(1 for r in ku_rows if r["entity_list_json"] == "[]"),
        "entities_per_ku_mean": round(sum(ent_per_ku) / max(1, len(ent_per_ku)), 3),
        "llm_requests": meta.get("num_requests"), "llm_prompt_tokens": meta.get("prompt_tokens"),
        "llm_completion_tokens": meta.get("completion_tokens"),
        "seconds": {"chunking": round(t1 - t0), "hybrid_extraction": round(t2 - t1),
                    "graph_construction": round(t3 - t2)},
    }
    json.dump(stats, open(os.path.join(out, "index_stats.json"), "w"), indent=1)
    print("INDEX_STATS", json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
