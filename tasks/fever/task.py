"""FEVER claim verification (SUPPORTS / REFUTES / NOT ENOUGH INFO) against the official FEVER Wikipedia dump
(wiki-pages: the introduction of 5.4M pages, June 2017), as in the original shared task. Source: fever.ai.

test  = official paper_test (labelled), stratified by label.
train = official train, drawn with the test label mix; claims that also occur in the test are removed.
corpus = every non-empty page of wiki-pages: title = page name, text = its whole introduction (detokenised).
supporting_doc_ids = the pages of all evidence sets (none for NOT ENOUGH INFO)."""
import json
import re
import unicodedata
import zipfile

from tasks.common import (Ids, StreamCorpus, drop_duplicates_of, first_label, mix_of, new_rng, pick, sha256_file,
                          write_split)

RAW = "benchmarks/FEVER/raw/fever.ai"
LABELS = {
    "NOT ENOUGH INFO": ["not enough info", "not enough information", "nei", "insufficient"],
    "SUPPORTS": ["supports", "support", "supported"],
    "REFUTES": ["refutes", "refute", "refuted"],
}
_TOK = {"-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]", "-LCB-": "{", "-RCB-": "}", "-COLON-": ":"}


def clean(s):
    for k, v in _TOK.items():
        s = s.replace(k, v)
    s = re.sub(r"\s+([,.;:)\]!?%])", r"\1", s)
    s = re.sub(r"([(\[])\s+", r"\1", s).replace("``", '"').replace("''", '"')
    return re.sub(r"\s+'s\b", "'s", s).strip()


def page_key(t):
    return unicodedata.normalize("NFD", t)


def _claims(root, name):
    rows = [json.loads(l) for l in open(f"{root}/{RAW}/{name}.jsonl")]
    for r in rows:
        r["pages"] = sorted({page_key(e[2]) for s in r["evidence"] for e in s if e[2]})
    return rows


TRAIN_MIX = "test"  # "test": train follows the test label mix; "natural": the official train's own mix (tasks.prepare --rules)


def prepare(root, out, n_train, n_test, seed):
    rng = new_rng(seed)
    ids = Ids(rng)
    lab = lambda r: r["label"]
    test = pick(_claims(root, "paper_test"), n_test, rng, lab)
    train_c, n_dup = drop_duplicates_of(_claims(root, "train"), test, lambda r: r["claim"])
    train = pick(train_c, n_train, rng, lab, mix=mix_of(test, lab) if TRAIN_MIX == "test" else None)

    want = {p for r in train + test for p in r["pages"]}
    n_empty = 0
    with StreamCorpus(f"{out}/public/corpus/corpus.jsonl", ids, want) as corpus, \
            zipfile.ZipFile(f"{root}/{RAW}/wiki-pages.zip") as z:
        names = sorted(n for n in z.namelist() if n.startswith("wiki-pages/") and n.endswith(".jsonl"))
        for n in names:
            for line in z.open(n):
                d = json.loads(line)
                text = clean(d["text"])
                if not d["id"] or not text:
                    n_empty += 1
                    continue
                corpus.add(page_key(d["id"]), clean(d["id"].replace("_", " ")), text)
    missing = want - set(corpus.found)
    records = {}
    for name, drawn in (("train", train), ("test", test)):
        records[name] = [{
            "qid": ids.new(), "question": r["claim"], "meta": {}, "answers": [r["label"]],
            "supporting_doc_ids": [corpus.doc_id(p) for p in r["pages"] if p in corpus.found], "orig_id": r["id"],
        } for r in drawn]
    return write_split(out, corpus, records, rng, {
        "task": "fever", "seed": seed, "test_from": "official paper_test", "train_from": "official train",
        "source_sha256": {f: sha256_file(f"{root}/{RAW}/{f}") for f in ("paper_test.jsonl", "train.jsonl", "wiki-pages.zip")},
        "corpus_files": len(names), "corpus_empty_skipped": n_empty,
        "label_mix": {n: mix_of(v, lab) for n, v in (("test", test), ("train", train))},
        "train_dropped_duplicate_claims": n_dup, "n_evidence_pages_missing_from_corpus": len(missing),
        "evidence_pages_missing_sample": sorted(missing)[:20]},
        public_extra=("supporting_doc_ids",), gold_extra=("orig_id",))


def score_one(pred, gold):
    return {"acc": float(first_label(pred, LABELS) == gold["answers"][0])}
