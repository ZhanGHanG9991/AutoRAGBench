"""HotpotQA, fullwiki setting: 2-hop bridge / comparison questions over the official HotpotQA Wikipedia corpus
(the first paragraph of every article of enwiki-20171001, 5.2M documents).

test  = official dev (all level=hard), stratified by type (bridge / comparison).
train = official train, level=hard only (the dev has no easy / medium questions), drawn with the test type mix.
corpus = the whole fullwiki corpus; a question's supporting documents are its supporting titles.
Train and dev share supporting articles in the original release (Wikipedia is one graph); that overlap is kept and
reported, not filtered."""
import bz2
import glob
import json

from tasks.common import (Ids, StreamCorpus, drop_duplicates_of, exact_match, mix_of, new_rng, pick, sha256_file,
                          token_f1, write_split)

DIST = "benchmarks/HotpotQA/raw/hf/distractor"
WIKI = "benchmarks/HotpotQA/corpus/enwiki-20171001-pages-meta-current-withlinks-abstracts/*/wiki_*.bz2"


def _rows(root, pattern, columns=("id", "question", "answer", "type", "level", "supporting_facts")):
    import pandas as pd
    files = sorted(glob.glob(f"{root}/{DIST}/{pattern}"))
    return pd.concat([pd.read_parquet(f, columns=list(columns)) for f in files]).to_dict("records"), files


TRAIN_MIX = "test"  # "test": train follows the test type mix; "natural": the hard train's own mix (tasks.prepare --rules)


def prepare(root, out, n_train, n_test, seed):
    rng = new_rng(seed)
    ids = Ids(rng)
    dev, dev_files = _rows(root, "validation-*.parquet")
    train_all, train_files = _rows(root, "train-*.parquet")
    typ = lambda r: r["type"]
    sup = lambda r: sorted(set(r["supporting_facts"]["title"]))

    test = pick(dev, n_test, rng, typ)
    train_c, n_dup = drop_duplicates_of([r for r in train_all if r["level"] == "hard"], test, lambda r: r["question"])
    train = pick(train_c, n_train, rng, typ, mix=mix_of(test, typ) if TRAIN_MIX == "test" else None)

    want = {t for r in train + test for t in sup(r)}
    wiki_files = sorted(glob.glob(f"{root}/{WIKI}"))
    n_empty = 0
    with StreamCorpus(f"{out}/public/corpus/corpus.jsonl", ids, want) as corpus:
        for f in wiki_files:
            for line in bz2.open(f, "rt"):
                d = json.loads(line)
                text = "".join(d["text"]).strip()
                if not text:
                    n_empty += 1
                    continue
                corpus.add(d["title"], d["title"], text)
    missing = want - set(corpus.found)
    records = {}
    for name, drawn in (("train", train), ("test", test)):
        records[name] = [{
            "qid": ids.new(), "question": r["question"], "meta": {}, "answers": [r["answer"]],
            "supporting_doc_ids": [corpus.doc_id(t) for t in sup(r) if t in corpus.found],
            "type": r["type"], "level": r["level"], "orig_id": r["id"],
        } for r in drawn]
    train_titles = {t for r in train for t in sup(r)}
    return write_split(out, corpus, records, rng, {
        "task": "hotpotqa", "seed": seed, "test_from": "official distractor dev (hard)",
        "train_from": "official train, level=hard",
        "source_sha256": {**{f.split("/")[-1]: sha256_file(f) for f in dev_files + train_files}},
        "corpus": WIKI, "corpus_files": len(wiki_files), "corpus_empty_skipped": n_empty,
        "type_mix": {n: mix_of(v, typ) for n, v in (("test", test), ("train", train))},
        "train_dropped_duplicate_questions": n_dup,
        "supporting_titles_missing_from_corpus": sorted(missing)[:50], "n_supporting_titles_missing": len(missing),
        "test_q_with_supporting_title_in_train": sum(any(t in train_titles for t in sup(r)) for r in test)},
        public_extra=("supporting_doc_ids",), gold_extra=("type", "level", "orig_id"))


def score_one(pred, gold):
    return {"f1": token_f1(pred, gold["answers"]), "em": exact_match(pred, gold["answers"])}
