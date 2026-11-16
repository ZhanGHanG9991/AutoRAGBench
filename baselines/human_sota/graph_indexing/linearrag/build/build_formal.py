"""Build a LinearRAG index only from the frozen formal-v1 public corpus.

This is intentionally separate from the pilot builder.  It reuses the pinned
upstream implementation through rageng/runtime.py and refuses any non-formal
input path.  No QA split is read while indexing.
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
p.add_argument("--smoke-dev")
p.add_argument("--smoke-queries", type=int, default=20)
a = p.parse_args()

src = Path(a.corpus).resolve()
expected_fragment = f"/data_formal-v1/{a.task}/public/corpus/corpus.jsonl"
assert str(src).endswith(expected_fragment), (src, expected_fragment)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
assert not (out / "COMPLETE").exists()

started = time.monotonic()
rows = []
smoke_support_ids = set()
if a.smoke_dev:
    assert a.limit > 0
    with Path(a.smoke_dev).open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= a.smoke_queries:
                break
            smoke_support_ids.update(str(x) for x in json.loads(line).get("supporting_doc_ids", []))
    assert smoke_support_ids, "Smoke dev rows do not expose supporting_doc_ids"
    filler_budget = max(0, a.limit - len(smoke_support_ids))
    seen_support, fillers = set(), 0
    with src.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            doc_id = str(row["doc_id"])
            if doc_id in smoke_support_ids:
                rows.append(row)
                seen_support.add(doc_id)
            elif fillers < filler_budget:
                rows.append(row)
                fillers += 1
            if len(rows) >= a.limit and seen_support == smoke_support_ids:
                break
    assert seen_support == smoke_support_ids, sorted(smoke_support_ids - seen_support)[:10]
else:
    with src.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if a.limit and len(rows) >= a.limit:
                break

embed, ner, nparams = model_bundle(a.models)
# QASPER questions are scoped by meta.doc_id.  Independent per-paper graphs
# preserve that restriction without changing LinearRAG's retrieval equations.
groups = [(r["doc_id"], [r]) for r in rows] if a.task == "qasper" else [("global", rows)]
mapping, stats = {}, []
for key, group in groups:
    sub = hashlib.sha256(str(key).encode()).hexdigest()[:24]
    idx = out / "graphs" / sub
    texts, metadata = passages(group)
    rag = make_rag(idx, embed, ner)
    rag.index(texts)
    (idx / "mention_adjacency.json").write_text(json.dumps({
        "e2s": rag.entity_hash_id_to_sentence_hash_ids,
        "s2e": rag.sentence_hash_id_to_entity_hash_ids,
    }))
    meta = {rag.passage_embedding_store.text_to_hash_id[t]: m for t, m in zip(texts, metadata)}
    (idx / "passage_metadata.json").write_text(json.dumps(meta))
    s = dict(
        key=key,
        entities=len(rag.entity_embedding_store.hash_ids),
        sentences=len(rag.sentence_embedding_store.hash_ids),
        passages=len(rag.passage_embedding_store.hash_ids),
        mention_edges=sum(map(len, rag.sentence_hash_id_to_entity_hash_ids.values())),
        ppr_edges=rag.graph.ecount(),
    )
    stats.append(s)
    mapping[str(key)] = sub
    (out / "mapping.json").write_text(json.dumps(mapping))
    (out / "graph_stats.json").write_text(json.dumps(stats))

summary = dict(
    task=a.task,
    data_version="formal-v1",
    corpus_sha256=hashlib.file_digest(src.open("rb"), "sha256").hexdigest(),
    corpus=str(src),
    source_documents=len(rows),
    limit=a.limit,
    smoke_support_documents=len(smoke_support_ids),
    seconds=time.monotonic() - started,
    embedding_parameters=nparams,
    generation_calls=0,
    upstream_commit=COMMIT,
    parameters=PARAMS,
    job_id=os.environ.get("SLURM_JOB_ID"),
    **{k: sum(s[k] for s in stats) for k in ("entities", "sentences", "passages", "mention_edges", "ppr_edges")},
)
assert all(summary[k] > 0 for k in ("entities", "sentences", "passages", "mention_edges", "ppr_edges"))
summary["index_bytes"] = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
(out / "index_stats.json").write_text(json.dumps(summary, indent=2))
(out / "COMPLETE").write_text("complete\n")
print("INDEX_COMPLETE", json.dumps(summary), flush=True)
