"""Shared preparation for knowledge-graph QA tasks (WebQSP, CWQ) built from the RoG releases
(rmanluo/RoG-webqsp, rmanluo/RoG-cwq): each question comes with its Freebase subgraph (all triples within the
dataset's maximum reasoning hops of the topic entities, entity ids mapped to names; unnamed CVT nodes keep
their m./g. ids), topic entities (q_entity) and answers.

The corpus is one knowledge graph: the union of the subgraphs of every answerable question of the release (all
splits, independent of which questions are drawn). It is written twice into public/corpus/:
  corpus.jsonl  one document per entity that has outgoing edges: {doc_id, title: entity, text: one
                "head | relation | tail" line per outgoing triple}  (what baselines and the contract use)
  triples.tsv   the same triples, "head\\trelation\\ttail", sorted
Questions are kept only if a topic entity and an answer both occur in their own subgraph. test is drawn from the
official test split and train from the official train (+ validation) split, as released: questions of the two
splits share topic entities, as in the original benchmark (reported in the manifest, not filtered).
meta = {"topic_entities": [...]}, as in the standard KGQA setting.
"""
import glob
import os

from tasks.common import Ids, Pool, drop_duplicates_of, new_rng, pick, sha256_file, write_split

COLS = ["id", "question", "answer", "q_entity", "a_entity"]


def _rows(files, columns):
    import pyarrow.parquet as pq  # preparation only; scoring needs no pyarrow
    for f in files:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=64, columns=columns):
            yield from batch.to_pylist()


def prepare_kgqa(task, root, out, n_train, n_test, seed, raw_glob, test_splits=("test",),
                 train_splits=("train", "validation"), file_splits=None):
    rng = new_rng(seed)
    ids = Ids(rng)
    split_of_file = lambda f: os.path.basename(f).split("-")[0]
    files = sorted(f for f in glob.glob(f"{root}/{raw_glob}") if file_splits is None or split_of_file(f) in file_splits)
    if not files:
        raise FileNotFoundError(f"{root}/{raw_glob}")

    # questions answerable inside their own subgraph, and the union of all their subgraphs (the corpus)
    rows, triples = [], set()
    for f in files:
        for r in _rows([f], COLS + ["graph"]):
            g = r["graph"] or []
            nodes = {x for t in g for x in (t[0], t[2])}
            topic = [e for e in (r["q_entity"] or []) if e in nodes]
            answers = [a for a in (r["answer"] or []) if a]
            if not topic or not answers or not any(a in nodes for a in (r["a_entity"] or [])):
                continue
            triples.update(map(tuple, g))
            rows.append({k: r[k] for k in COLS} | {"topic": topic, "answers": answers, "split": split_of_file(f)})
    test_c = [r for r in rows if r["split"] in test_splits]
    test = pick(test_c, n_test, rng)
    train_c, n_dup = drop_duplicates_of([r for r in rows if r["split"] in train_splits], test, lambda r: r["question"])
    train = pick(train_c, n_train, rng)

    out_edges = {}
    for h, rel, t in sorted(triples):
        out_edges.setdefault(h, []).append(f"{h} | {rel} | {t}")
    pool = Pool(ids)
    for h in sorted(out_edges):
        pool.add(h, h, "\n".join(out_edges[h]))
    records = {}
    for name, drawn in (("train", train), ("test", test)):
        records[name] = [{"qid": ids.new(), "question": r["question"], "meta": {"topic_entities": r["topic"]},
                          "answers": r["answers"], "orig_id": r["id"]} for r in drawn]
    train_topics = {t for r in train for t in r["topic"]}
    m = write_split(out, pool, records, rng, {
        "task": task, "source": raw_glob, "source_sha256": {os.path.basename(f): sha256_file(f) for f in files},
        "seed": seed, "test_from": list(test_splits), "train_from": list(train_splits),
        "n_answerable": {s: sum(r["split"] == s for r in rows) for s in sorted({r["split"] for r in rows})},
        "train_dropped_duplicate_questions": n_dup,
        "test_q_with_topic_in_train": sum(any(t in train_topics for t in r["topic"]) for r in test),
        "n_triples": len(triples), "n_entities": len({x for t in triples for x in (t[0], t[2])}),
        "n_relations": len({t[1] for t in triples})}, gold_extra=("orig_id",))
    with open(os.path.join(out, "public", "corpus", "triples.tsv"), "w") as f:
        for h, rel, t in sorted(triples):
            f.write(f"{h}\t{rel}\t{t}\n")
    return m
