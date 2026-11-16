"""NQ-open: single-hop open-domain questions from Google search queries, over the full DPR Wikipedia
(psgs_w100, 21M 100-word passages, the corpus of the original open-domain setting).

test  = official NQ-open test (HF nq_open "validation", 3,610 questions), drawn uniformly. No retrievability filter:
        Tevatron's dev keeps only questions whose gold passage DPR found (6,489 / 8,757), which biases towards easy.
train = Tevatron nq-train (DPR train questions with a positive passage; the filter is fine for supervision);
        supporting_doc_ids = its positive passages.
corpus = all of psgs_w100."""
import gzip
import json

from tasks.common import (Ids, StreamCorpus, drop_duplicates_of, exact_match, new_rng, pick, sha256_file, token_f1,
                          write_split)

TEST = "benchmarks/NQ-open/raw/google-research-datasets__nq_open/nq_open/validation-00000-of-00001.parquet"
TRAIN = "benchmarks/NQ-open/raw/Tevatron__wikipedia-nq/nq-train.jsonl.gz"
PSGS = "benchmarks/ALCE-ASQA/corpus/psgs_w100.tsv.gz"


def prepare(root, out, n_train, n_test, seed):
    import pandas as pd
    rng = new_rng(seed)
    ids = Ids(rng)
    tq = pd.read_parquet(f"{root}/{TEST}").to_dict("records")
    test_c = [{"question": r["question"], "answers": list(r["answer"]), "pos": [], "orig_id": i} for i, r in enumerate(tq)]
    train_c = []
    with gzip.open(f"{root}/{TRAIN}", "rt") as f:
        for line in f:
            r = json.loads(line)
            if r["positive_passages"] and r["answers"]:
                train_c.append({"question": r["query"], "answers": list(r["answers"]),
                                "pos": [p["docid"] for p in r["positive_passages"]], "orig_id": r["query_id"]})
    test = pick(test_c, n_test, rng)
    train_c, n_dup = drop_duplicates_of(train_c, test, lambda r: r["question"])
    train = pick(train_c, n_train, rng)

    want = {d for r in train for d in r["pos"]}
    with StreamCorpus(f"{out}/public/corpus/corpus.jsonl", ids, want) as corpus:
        with gzip.open(f"{root}/{PSGS}", "rt") as f:
            next(f)  # header: id, text, title
            for line in f:
                pid, text, title = line.rstrip("\n").split("\t")
                corpus.add(pid, title, text.strip('"').replace('""', '"'))
    missing = want - set(corpus.found)
    records = {}
    for name, drawn in (("train", train), ("test", test)):
        records[name] = [{
            "qid": ids.new(), "question": r["question"], "meta": {}, "answers": r["answers"],
            "supporting_doc_ids": [corpus.doc_id(d) for d in r["pos"] if d in corpus.found], "orig_id": r["orig_id"],
        } for r in drawn]
    return write_split(out, corpus, records, rng, {
        "task": "nq_open", "seed": seed, "test_from": "official NQ-open test (nq_open validation)",
        "train_from": "Tevatron nq-train (DPR train with positives)", "corpus": PSGS,
        "source_sha256": {"test": sha256_file(f"{root}/{TEST}"), "train": sha256_file(f"{root}/{TRAIN}")},
        "n_test_candidates": len(test_c), "train_dropped_duplicate_questions": n_dup,
        "n_positive_passages_missing_from_corpus": len(missing)},
        public_extra=("supporting_doc_ids",), gold_extra=("orig_id",))


def score_one(pred, gold):
    return {"em": exact_match(pred, gold["answers"]), "f1": token_f1(pred, gold["answers"])}
