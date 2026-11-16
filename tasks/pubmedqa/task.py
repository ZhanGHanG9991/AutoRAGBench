"""PubMedQA (PQA-L): yes / no / maybe research questions, answered from PubMed abstracts.

test  = the official PQA-L test set (500 pubids, pubmedqa/data/test_ground_truth.json on GitHub).
train = the other 500 expert-labelled PQA-L questions (all there is; PQA-A labels are rule-generated and have no
        "maybe"). n_train is capped at 500.
corpus = every PubMedQA abstract (PQA-L 1k + PQA-U 61k + PQA-A 211k, deduplicated by pubid), all in the same form:
         labelled sections without the title and the conclusion (questions are derived from titles, answers from
         conclusions), as in the reasoning-required setting."""
import json

from tasks.common import Ids, Pool, first_label, mix_of, new_rng, pick, sha256_file, write_split

RAW = "benchmarks/PubMedQA/raw/qiaojin__PubMedQA"
SOURCE = f"{RAW}/pqa_labeled/train-00000-of-00001.parquet"
OTHER = [f"{RAW}/pqa_unlabeled/train-00000-of-00001.parquet", f"{RAW}/pqa_artificial/train-00000-of-00001.parquet"]
TEST_IDS = "benchmarks/PubMedQA/raw/pubmedqa__github/test_ground_truth.json"
LABELS = {"yes": ["yes"], "no": ["no"], "maybe": ["maybe"]}


def _text(ctx):
    labels = list(ctx.get("labels", [])) or [""] * len(ctx["contexts"])
    return "\n".join(f"{l}: {c}" if l else c for l, c in zip(labels, ctx["contexts"]))


def prepare(root, out, n_train, n_test, seed):
    import pandas as pd
    rng = new_rng(seed)
    ids = Ids(rng)
    rows = pd.read_parquet(f"{root}/{SOURCE}").to_dict("records")
    gt = json.load(open(f"{root}/{TEST_IDS}"))
    by_id = {str(r["pubid"]): r for r in rows}
    assert all(by_id[k]["final_decision"] == v for k, v in gt.items()), "test_ground_truth disagrees with pqa_labeled"
    lab = lambda r: r["final_decision"]
    test = pick([by_id[k] for k in gt], min(n_test, len(gt)), rng, lab)
    train = pick([r for k, r in by_id.items() if k not in gt], min(n_train, len(rows) - len(gt)), rng, lab)

    pool = Pool(ids)
    for r in rows:
        pool.add(str(r["pubid"]), "", _text(r["context"]))
    for f in OTHER:
        for pid, ctx in pd.read_parquet(f"{root}/{f}", columns=["pubid", "context"]).itertuples(index=False):
            pool.add(str(pid), "", _text(ctx))
    records = {}
    for name, drawn in (("train", train), ("test", test)):
        records[name] = [{
            "qid": ids.new(), "question": r["question"], "meta": {}, "answers": [r["final_decision"]],
            "supporting_doc_ids": [pool.doc_id(str(r["pubid"]))], "orig_id": int(r["pubid"]),
        } for r in drawn]
    return write_split(out, pool, records, rng, {
        "task": "pubmedqa", "seed": seed, "test_from": "official PQA-L test (500)", "train_from": "rest of PQA-L",
        "source_sha256": {f.split("/")[-2] + "/" + f.split("/")[-1]: sha256_file(f"{root}/{f}") for f in [SOURCE, TEST_IDS, *OTHER]},
        "label_mix": {n: mix_of(v, lab) for n, v in (("test", test), ("train", train))}},
        public_extra=("supporting_doc_ids",), gold_extra=("orig_id",))


def score_one(pred, gold):
    return {"acc": float(first_label(pred, LABELS) == gold["answers"][0])}
