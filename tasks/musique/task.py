"""MuSiQue-Ans: 2-4 hop questions over Wikipedia paragraphs.

test  = official dev (answers public; the official test has none), stratified by hop count. The official dev
        shares no supporting paragraph with the official train (MuSiQue's anti-shortcut construction).
train = official train, drawn with the test hop mix; a supporting paragraph backs at most 5 drawn questions (MuSiQue
        composes questions from shared single-hop questions: with 3, too few 4-hop questions remain).
corpus = every paragraph of the Ans train, dev and test files (deduplicated; MuSiQue has no global corpus, this is
         the largest in-distribution one). Answers of the official test file are not used."""
import json

from tasks.common import (Ids, Pool, drop_duplicates_of, exact_match, mix_of, new_rng, pick, sha1, sha256_file,
                          token_f1, write_split)

DATA = "benchmarks/MuSiQue/data"
# sampling rules (tasks.prepare --rules overrides them per data version; these defaults are formal-v1's)
TRAIN_MIX = "test"  # "test": train follows the test hop mix; "natural": the official train's own mix
TRAIN_CAP = 5       # a supporting paragraph backs at most this many drawn train questions (None: no cap)
FILES = {s: f"{DATA}/musique_ans_v1.0_{s}.jsonl" for s in ("train", "dev", "test")}


def prepare(root, out, n_train, n_test, seed):
    rng = new_rng(seed)
    ids = Ids(rng)
    rows = {s: [json.loads(l) for l in open(f"{root}/{f}")] for s, f in FILES.items()}
    key = lambda p: sha1(p["title"] + "\x00" + p["paragraph_text"])
    hops = lambda r: len(r["question_decomposition"])
    sup = lambda r: [key(p) for p in r["paragraphs"] if p["is_supporting"]]

    dev = [r for r in rows["dev"] if r.get("answerable", True)]
    test = pick(dev, n_test, rng, hops)
    train_c, n_dup = drop_duplicates_of([r for r in rows["train"] if r.get("answerable", True)], test, lambda r: r["question"])
    train = pick(train_c, n_train, rng, hops, mix=mix_of(test, hops) if TRAIN_MIX == "test" else None,
                 keys_of=sup if TRAIN_CAP else None, cap=TRAIN_CAP)

    pool = Pool(ids)
    for s in ("train", "dev", "test"):
        for r in rows[s]:
            for p in r["paragraphs"]:
                pool.add(key(p), p["title"], p["paragraph_text"])
    train_sup = {k for r in train for k in sup(r)}
    records = {}
    for name, drawn in (("train", train), ("test", test)):
        records[name] = [{
            "qid": ids.new(), "question": r["question"], "meta": {},
            "answers": [r["answer"]] + list(r.get("answer_aliases", [])),
            "supporting_doc_ids": [pool.doc_id(k) for k in sup(r)], "hops": hops(r), "orig_id": r["id"],
        } for r in drawn]
    return write_split(out, pool, records, rng, {
        "task": "musique", "source": FILES, "source_sha256": {s: sha256_file(f"{root}/{f}") for s, f in FILES.items()},
        "seed": seed, "test_from": "official dev", "train_from": "official train",
        "hop_mix": {n: mix_of(v, hops) for n, v in (("test", test), ("train", train))},
        "train_dropped_duplicate_questions": n_dup, "rules": {"train_mix": TRAIN_MIX, "train_cap": TRAIN_CAP},
        "test_q_with_supporting_doc_in_train": sum(any(k in train_sup for k in sup(r)) for r in test)},
        public_extra=("supporting_doc_ids",), gold_extra=("hops", "orig_id"))


def score_one(pred, gold):
    return {"f1": token_f1(pred, gold["answers"]), "em": exact_match(pred, gold["answers"])}
