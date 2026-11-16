"""Build a LinearRAG index only from a frozen formal-v2 public corpus.

No QA split is read while indexing. The relation-free Tri-Graph algorithm and
model bundle come from the pinned adapter in rageng/runtime.py.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from runtime import model_bundle, make_rag, passages, PARAMS, COMMIT


p = argparse.ArgumentParser()
p.add_argument("--task", required=True)
p.add_argument("--corpus", required=True)
p.add_argument("--out", required=True)
p.add_argument("--models", required=True)
p.add_argument("--limit", type=int, default=0)
a = p.parse_args()

src = Path(a.corpus).resolve()
expected_fragment = f"/data_formal-v2/{a.task}/public/corpus/corpus.jsonl"
assert str(src).endswith(expected_fragment), (src, expected_fragment)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
assert not (out / "COMPLETE").exists()

started = time.monotonic()
rows = []
with src.open(encoding="utf-8") as source:
    for line in source:
        rows.append(json.loads(line))
        if a.limit and len(rows) >= a.limit:
            break

embed, ner, nparams = model_bundle(a.models)
# QASPER questions are scoped by meta.doc_id. Independent per-paper graphs
# preserve that restriction without changing LinearRAG's retrieval equations.
groups = [(r["doc_id"], [r]) for r in rows] if a.task == "qasper" else [("global", rows)]
mapping, stats = {}, []
for key, group in groups:
    sub = hashlib.sha256(str(key).encode()).hexdigest()[:24]
    idx = out / "graphs" / sub
    texts, metadata = passages(group)
    rag = make_rag(idx, embed, ner)
    rag.index(texts)
    (idx / "mention_adjacency.json").write_text(
        json.dumps(
            {
                "e2s": rag.entity_hash_id_to_sentence_hash_ids,
                "s2e": rag.sentence_hash_id_to_entity_hash_ids,
            }
        )
    )
    meta = {
        rag.passage_embedding_store.text_to_hash_id[t]: m
        for t, m in zip(texts, metadata)
    }
    (idx / "passage_metadata.json").write_text(json.dumps(meta))
    stat = dict(
        key=key,
        entities=len(rag.entity_embedding_store.hash_ids),
        sentences=len(rag.sentence_embedding_store.hash_ids),
        passages=len(rag.passage_embedding_store.hash_ids),
        mention_edges=sum(map(len, rag.sentence_hash_id_to_entity_hash_ids.values())),
        ppr_edges=rag.graph.ecount(),
    )
    stats.append(stat)
    mapping[str(key)] = sub
    (out / "mapping.json").write_text(json.dumps(mapping))
    (out / "graph_stats.json").write_text(json.dumps(stats))

summary = dict(
    task=a.task,
    data_version="formal-v2",
    corpus_sha256=hashlib.file_digest(src.open("rb"), "sha256").hexdigest(),
    corpus=str(src),
    source_documents=len(rows),
    limit=a.limit,
    seconds=time.monotonic() - started,
    embedding_parameters=nparams,
    generation_calls=0,
    upstream_commit=COMMIT,
    parameters=PARAMS,
    job_id=os.environ.get("SLURM_JOB_ID"),
    **{
        key: sum(stat[key] for stat in stats)
        for key in ("entities", "sentences", "passages", "mention_edges", "ppr_edges")
    },
)
assert all(
    summary[key] > 0
    for key in ("entities", "sentences", "passages", "mention_edges", "ppr_edges")
)
summary["index_bytes"] = sum(
    file.stat().st_size for file in out.rglob("*") if file.is_file()
)
(out / "index_stats.json").write_text(json.dumps(summary, indent=2))
(out / "COMPLETE").write_text("complete\n")
print("INDEX_COMPLETE", json.dumps(summary), flush=True)
