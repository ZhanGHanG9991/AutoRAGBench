"""QASPER: questions about NLP papers, answered from the paper's full text. Each question targets one
paper, given in meta.doc_id. The corpus holds every paper of the train / validation / test releases, one
document per paper (title, abstract, sections).

test  = questions on the official test papers; train = questions on the official train + validation papers
(so a paper belongs to one split, as in the release). At most 3 questions per paper; train follows the test
answer-type mix."""

from tasks.common import (Ids, Pool, mix_of, new_rng, pick, sha256_file, token_f1,
                          write_split)

RAW = "benchmarks/QASPER/raw/allenai__qasper/parquet/qasper"
SPLITS = ("train", "validation", "test")


def answer_string(a):
    """Reference answer text, as in the official qasper_evaluator."""
    if a["unanswerable"]:
        return "Unanswerable", "none"
    if a["extractive_spans"] is not None and len(a["extractive_spans"]):
        return ", ".join(a["extractive_spans"]), "extractive"
    if a["free_form_answer"]:
        return a["free_form_answer"], "abstractive"
    if a["yes_no"] is not None:
        return ("Yes" if a["yes_no"] else "No"), "boolean"
    return "", "none"


def paper_text(r):
    parts = [f"Abstract\n{r['abstract']}"]
    ft = r["full_text"]
    for name, paras in zip(ft["section_name"], ft["paragraphs"]):
        body = "\n".join(p for p in paras if p)
        if body:
            parts.append(f"## {name}\n{body}")
    return "\n\n".join(parts)


# sampling rules (tasks.prepare --rules overrides them per data version; these defaults are formal-v1's)
TRAIN_MIX = "test"  # "test": train follows the test answer-type mix; "natural": the train papers' own mix
TRAIN_CAP = 3       # at most this many train questions per paper (None: every question); the test keeps 3


def prepare(root, out, n_train, n_test, seed):
    import pandas as pd
    rng = new_rng(seed)
    ids = Ids(rng)
    pool = Pool(ids)
    items, shas = [], {}
    for s in SPLITS:
        path = f"{root}/{RAW}/{s}/0.parquet"
        shas[s] = sha256_file(path)
        for r in pd.read_parquet(path).to_dict("records"):
            pool.add(r["id"], r["title"], paper_text(r))
            qa = r["qas"]
            for q, qid, ans in zip(qa["question"], qa["question_id"], qa["answers"]):
                refs = [answer_string(a) for a in ans["answer"]]
                refs = [x for x in refs if x[0]]
                if refs:
                    items.append({"paper": r["id"], "question": q, "refs": refs, "orig_id": qid, "split": s})
    typ = lambda it: it["refs"][0][1]
    paper = lambda it: [it["paper"]]
    test = pick([it for it in items if it["split"] == "test"], n_test, rng, typ, keys_of=paper, cap=3)
    # no question-text deduplication: generic questions ("What datasets do they use?") recur across papers and are
    # different questions; train and test papers are disjoint
    train = pick([it for it in items if it["split"] != "test"], n_train, rng, typ,
                 mix=mix_of(test, typ) if TRAIN_MIX == "test" else None, keys_of=paper if TRAIN_CAP else None, cap=TRAIN_CAP)
    records = {}
    for name, drawn in (("train", train), ("test", test)):
        records[name] = []
        for it in drawn:
            doc = pool.doc_id(it["paper"])
            records[name].append({
                "qid": ids.new(), "question": it["question"], "meta": {"doc_id": doc},
                "answers": [a for a, _ in it["refs"]], "answer_type": typ(it),
                "supporting_doc_ids": [doc], "orig_id": it["orig_id"],
            })
    return write_split(out, pool, records, rng, {
        "task": "qasper", "source": RAW, "source_sha256": shas, "seed": seed, "n_papers": len(pool),
        "test_from": "official test papers", "train_from": "official train + validation papers",
        "answer_type_mix": {n: mix_of(v, typ) for n, v in (("test", test), ("train", train))}},
        public_extra=("supporting_doc_ids", "answer_type"), gold_extra=("orig_id",))


def score_one(pred, gold):
    return {"f1": token_f1(pred, gold["answers"])}
