#!/usr/bin/env python3
"""Retriever pairs for the formal-v1 evaluator. Qasper is not included.

Labels follow the pilot final rule. Multihop positives are retrieved gold
windows that pass the window filter. m and tau are fit on the evaluator-train
questions only. Failed gold windows are dropped, not labeled negative.
Negatives are downsampled to the positive count with seed 20261019.

The ablation file keeps the balanced main negatives and the other tasks'
positives, and replaces filtered multihop positives with every retrieved gold
hit. It is not rebalanced and is not a training job.
"""

def _release_path(value):
    """Resolve portable build/workspace paths without changing their layout."""
    import os
    from pathlib import Path
    workspace = os.environ.get("WORKSPACE", str(Path(__file__).resolve().parents[1]))
    root = os.environ.get("AUTORAGBENCH_ROOT", workspace)
    model_dir = os.environ.get("MODEL_DIR", str(Path(workspace) / "models"))
    for key, replacement in (("@WORKSPACE@", workspace), ("@AUTORAGBENCH_ROOT@", root),
                             ("@MODEL_DIR@", model_dir), ("@BUILD_TMP@", os.environ.get("BUILD_TMP", "/tmp"))):
        value = value.replace(key, replacement)
    return value

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from formal_index import DIM, FORMAL, MODEL, TASKS, disk_ok, index_dir, refuse  # noqa: E402

SEED = 20261019
TRAIN_TASKS = ["musique", "hotpotqa", "nq_open", "fever", "pubmedqa", "multihop_rag"]
SPLIT = os.path.join(FORMAL, "split")
SRC = _release_path("@AUTORAGBENCH_ROOT@/humansota/formal-v1-split")
STOPWORDS = frozenset(
    """a an the of to and in on for with as at by from or is are was were be been
    being it this that these those not no nor but if then than so such into over
    after before about between within without during under above up down out off
    again further once here there all any both each few more most other some own
    same than too very can will just don should now do does did doing have has
    had having i you he she they we me him her them my your our their what which
    who whom whose when where why how""".split()
)
CLOSED = frozenset({"yes", "no", "insufficient information"})
TOKEN = re.compile(r"[a-z0-9]+")
SEC_PER_STEP = 5628 / 2532  # sacct Elapsed 01:33:48 of pilot job 694511, 2532 steps


def norm_tokens(text):
    return TOKEN.findall(str(text).lower())


def content_tokens(text):
    return [tok for tok in norm_tokens(text) if tok not in STOPWORDS and len(tok) > 1]


def answer_class(answer):
    folded = " ".join(norm_tokens(answer))
    if not folded:
        raise SystemExit(f"empty normalized answer {answer!r}")
    return "closed" if folded in CLOSED else "entity"


def window_body(passage):
    text = str(passage)
    if "\n" not in text:
        raise SystemExit("multihop passage has no title line")
    return text.split("\n", 1)[1]


def contains_answer(window, answer):
    words = norm_tokens(window)
    answer_words = norm_tokens(answer)
    width = len(answer_words)
    if width == 0:
        return False
    for start in range(len(words) - width + 1):
        if words[start : start + width] == answer_words:
            return True
    return False


def overlap_score(question, window):
    question_words = set(content_tokens(question))
    if not question_words:
        return 0.0
    return len(question_words & set(content_tokens(window))) / len(question_words)


def keep_gold_window(kind, contained, rank, overlap, m_keep, tau):
    if kind == "entity" and contained:
        return True
    return rank < m_keep and overlap >= tau


def percentile_index(count):
    return (count + 3) // 4 - 1


def read_qids(task, kind):
    path = os.path.join(SPLIT, task, f"{kind}_qids.txt")
    refuse(path)
    qids = [line.strip() for line in open(path, encoding="utf-8") if line.strip()]
    if len(qids) != len(set(qids)):
        raise SystemExit(f"duplicate qids in {path}")
    return qids


def load_questions(task, qids):
    path = os.path.join(SRC, task, "train_fit.jsonl")
    refuse(path)
    if not path.endswith(f"/formal-v1-split/{task}/train_fit.jsonl"):
        raise SystemExit(f"REFUSE {path}")
    want = set(qids)
    found = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["qid"] in want:
                found[row["qid"]] = row
    missing = [qid for qid in qids if qid not in found]
    if missing:
        raise SystemExit(f"{task} missing {len(missing)} qids")
    tune = set(read_qids(task, "tune"))
    if set(qids) & tune:
        raise SystemExit(f"{task} evaluator-train overlaps tune")
    return [found[qid] for qid in qids]


def read_doc(docs_path, offsets, row):
    with open(docs_path, "rb") as handle:
        handle.seek(int(offsets[row]))
        return json.loads(handle.readline())


def encode_queries(questions):
    import torch
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        raise SystemExit("REFUSE retrieve without CUDA")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModel.from_pretrained(MODEL, local_files_only=True)
    model.to("cuda").eval()
    rows = []
    texts = [row["question"] for row in questions]
    for start in range(0, len(texts), 32):
        batch = texts[start : start + 32]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        enc = {k: v.to("cuda") for k, v in enc.items()}
        with torch.no_grad():
            hidden = model(**enc).last_hidden_state
            weights = enc["attention_mask"].unsqueeze(-1).type_as(hidden)
            pooled = (hidden * weights).sum(1) / weights.sum(1).clamp(min=1e-9)
        rows.append(pooled.float().cpu().numpy())
    return np.concatenate(rows, axis=0)


def _dot_block(emb, block_q, device):
    """Unnormalised dot of the stored fp16 rows, upcast per block to float32."""
    n = int(emb.shape[0])
    qn = int(block_q.shape[0])
    scores = np.empty((n, qn), dtype=np.float32)
    query = np.ascontiguousarray(block_q, dtype=np.float32)
    if device == "cuda":
        import torch

        query_t = torch.from_numpy(query).to("cuda")
        for row in range(0, n, 500000):
            piece = np.ascontiguousarray(np.asarray(emb[row : row + 500000], dtype=np.float32))
            part = torch.from_numpy(piece).to("cuda") @ query_t.T
            scores[row : row + len(piece)] = part.detach().float().cpu().numpy()
            del part
        del query_t
        return scores
    for row in range(0, n, 500000):
        piece = np.asarray(emb[row : row + 500000], dtype=np.float32)
        scores[row : row + len(piece)] = piece @ query.T
    return scores


def topk_search(task, vectors, k=10, device="cuda", allowed_doc_ids=None):
    directory = index_dir(task)
    done = json.load(open(os.path.join(directory, "done.json"), encoding="utf-8"))
    if done.get("check") != "pass" or done.get("normalized") is not False:
        raise SystemExit(f"REFUSE index not passed or normalized {directory}")
    if done.get("chunk_words") != TASKS[task]:
        raise SystemExit(f"REFUSE chunk {done.get('chunk_words')}")
    emb = np.load(os.path.join(directory, "embeddings.npy"), mmap_mode="r")
    if emb.dtype != np.float16 or emb.shape[1] != DIM or emb.shape[0] != done["n_passages"]:
        raise SystemExit(f"REFUSE embeddings {emb.shape} {emb.dtype}")
    offsets = np.load(os.path.join(directory, "offsets.npy"), mmap_mode="r")
    docs_path = os.path.join(directory, "docs.jsonl")
    n = int(emb.shape[0])
    if k < 1 or n < 1:
        raise SystemExit(f"REFUSE empty search n={n} k={k}")
    doc_ids = None
    if allowed_doc_ids is not None:
        if len(allowed_doc_ids) != len(vectors):
            raise SystemExit("REFUSE allowed_doc_ids length")
        doc_ids = []
        with open(docs_path, encoding="utf-8") as handle:
            for line in handle:
                doc_ids.append(json.loads(line)["doc_id"])
        if len(doc_ids) != n:
            raise SystemExit(f"REFUSE doc id count {len(doc_ids)} != {n}")
        doc_ids = np.asarray(doc_ids)
    hits = []
    batch_i = 0
    t_batch = time.time()
    # One pass over the matrix covers 32 queries. A pass per 4 queries would
    # reread the nq_open matrix about 400 times.
    for start in range(0, len(vectors), 32):
        block_q = np.asarray(vectors[start : start + 32], dtype=np.float32)
        qn = block_q.shape[0]
        scores = _dot_block(emb, block_q, device)
        for col in range(qn):
            column = scores[:, col]
            allow = None if allowed_doc_ids is None else allowed_doc_ids[start + col]
            if allow is not None:
                column = column.copy()
                column[doc_ids != allow] = -np.inf
            if not np.isfinite(column).any():
                hits.append([])
                continue
            take = min(k, int(np.isfinite(column).sum()))
            order = np.argpartition(-column, take - 1)[:take]
            order = order[np.argsort(-column[order])]
            chosen = []
            for row_id in order:
                row_id = int(row_id)
                doc = read_doc(docs_path, offsets, row_id)
                chosen.append({
                    "doc_id": doc["doc_id"],
                    "passage": doc["passage"],
                    "dot": float(column[row_id]),
                    "row": row_id,
                })
            hits.append(chosen)
        spent = time.time() - t_batch
        batch_i += 1
        done_n = start + qn
        eta = spent * (len(vectors) - done_n) / qn
        print(f"SEARCH {task} {done_n}/{len(vectors)} batch_sec {spent:.1f} eta_sec {eta:.0f}", flush=True)
        # The first batch pays for a cold read. Judge the limit by the next full batch.
        if batch_i >= 2 and qn == 32 and eta > 3 * 3600:
            print(f"ESTIMATE_OVER_3H {task} search eta_sec {eta:.0f} done {done_n}", flush=True)
            raise SystemExit(2)
        t_batch = time.time()
    return hits


def retrieve(task):
    disk_ok()
    questions = load_questions(task, read_qids(task, "evaluator_train"))
    out = os.path.join(FORMAL, "pairs", f"{task}_hits.jsonl")
    refuse(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.isfile(out) and sum(1 for _ in open(out, encoding="utf-8")) == len(questions):
        print(f"HITS_DONE {task} already", flush=True)
        return
    vectors = encode_queries(questions)
    found = topk_search(task, vectors, 10)
    if len(found) != len(questions):
        raise SystemExit(f"REFUSE hits {len(found)} != {len(questions)}")
    with open(out, "w", encoding="utf-8") as handle:
        for question, hits in zip(questions, found):
            handle.write(json.dumps({"qid": question["qid"], "hits": hits}, ensure_ascii=False) + "\n")
    print(f"HITS_DONE {task} n {len(questions)}", flush=True)


def load_all_docs(task):
    directory = index_dir(task)
    docs = [json.loads(line) for line in open(os.path.join(directory, "docs.jsonl"), encoding="utf-8")]
    return docs


def calibrate_multihop(questions, docs):
    rows_of = {}
    for row_id, doc in enumerate(docs):
        rows_of.setdefault(doc["doc_id"], []).append(row_id)
    counts = []
    overlaps = []
    for question in questions:
        if len(question.get("answers") or []) != 1:
            raise SystemExit(f"{question.get('qid')} does not have one answer")
        if answer_class(question["answers"][0]) != "entity":
            continue
        answer = question["answers"][0]
        for doc_id in dict.fromkeys(question.get("supporting_doc_ids") or []):
            contained = 0
            for row_id in rows_of.get(doc_id, []):
                body = window_body(docs[row_id]["passage"])
                if contains_answer(body, answer):
                    contained += 1
                    overlaps.append(overlap_score(question["question"], body))
            if contained:
                counts.append(contained)
    if not counts or not overlaps:
        raise SystemExit("multihop calibration found no answer-containing windows")
    counts.sort()
    overlaps.sort()
    detail = {
        "version": "formal_v1_gold_window",
        "m": counts[len(counts) // 2],
        "tau": overlaps[percentile_index(len(overlaps))],
        "n_articles_with_containment": len(counts),
        "n_containment_windows": len(overlaps),
        "m_index": len(counts) // 2,
        "tau_index": percentile_index(len(overlaps)),
        "calibration_split": "formal-v1 evaluator_train",
        "seed": SEED,
    }
    print("MULTIHOP_FILTER", json.dumps(detail), flush=True)
    return detail, rows_of


def positive_rows(question, docs, rows_of, m_keep, tau):
    kind = answer_class(question["answers"][0])
    answer = question["answers"][0]
    kept = set()
    for doc_id in dict.fromkeys(question.get("supporting_doc_ids") or []):
        scored = []
        for row_id in rows_of.get(doc_id, []):
            body = window_body(docs[row_id]["passage"])
            overlap = overlap_score(question["question"], body)
            contained = kind == "entity" and contains_answer(body, answer)
            scored.append((overlap, row_id, contained))
        scored.sort(key=lambda item: (-item[0], item[1]))
        for rank, (overlap, row_id, contained) in enumerate(scored):
            if keep_gold_window(kind, contained, rank, overlap, m_keep, tau):
                kept.add(row_id)
    return kept


def evaluator_text(question, passage):
    doc = str(passage).replace("\n", " ").replace("\t", " ").strip()
    if not doc:
        return None
    return str(question) + " [SEP] " + doc


def label_hits(task, question, hits, positive_rows_for=None, filter_multihop=True):
    gold = set(question.get("supporting_doc_ids") or [])
    rows = []
    dropped = 0
    seen = set()
    for hit in hits:
        text = evaluator_text(question["question"], hit["passage"])
        if text is None:
            continue
        key = (hit["doc_id"], text)
        if key in seen:
            continue
        seen.add(key)
        in_gold = hit["doc_id"] in gold
        if task == "multihop_rag" and in_gold and filter_multihop:
            if positive_rows_for is None or hit["row"] not in positive_rows_for:
                dropped += 1
                continue
        rows.append({
            "task": task,
            "qid": question["qid"],
            "doc_id": hit["doc_id"],
            "label": 1 if in_gold else 0,
            "text": text,
            "row": hit["row"],
        })
    return rows, dropped


def balance(pairs, seed):
    positive = [row for row in pairs if row["label"] == 1]
    negative = [row for row in pairs if row["label"] == 0]
    if not positive or not negative:
        raise SystemExit(f"cannot balance positive={len(positive)} negative={len(negative)}")
    rng = random.Random(seed)
    rng.shuffle(positive)
    rng.shuffle(negative)
    keep = min(len(positive), len(negative))
    kept = positive[:keep] + negative[:keep]
    rng.shuffle(kept)
    return kept, {
        "positive_before": len(positive),
        "negative_before": len(negative),
        "kept_each": keep,
        "n_balanced": len(kept),
    }


def load_hits(task):
    path = os.path.join(FORMAL, "pairs", f"{task}_hits.jsonl")
    refuse(path)
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def pool():
    disk_ok()
    tune = set()
    questions = {}
    for task in TRAIN_TASKS:
        train_q = read_qids(task, "evaluator_train")
        tune_q = read_qids(task, "tune")
        tune.update(tune_q)
        questions[task] = {row["qid"]: row for row in load_questions(task, train_q)}
        if task == "qasper":
            raise SystemExit("REFUSE qasper in training")
    detail = None
    rows_of = None
    docs = None
    mh_q = list(questions["multihop_rag"].values())
    docs = load_all_docs("multihop_rag")
    detail, rows_of = calibrate_multihop(mh_q, docs)
    filtered = []
    unfiltered_mh_pos = []
    per_task = {}
    for task in TRAIN_TASKS:
        hits_rows = load_hits(task)
        qids = read_qids(task, "evaluator_train")
        if [row["qid"] for row in hits_rows] != qids:
            raise SystemExit(f"REFUSE {task} hit order")
        pos = neg = dropped = 0
        task_rows = []
        for item in hits_rows:
            question = questions[task][item["qid"]]
            positive = None
            if task == "multihop_rag":
                positive = positive_rows(question, docs, rows_of, detail["m"], detail["tau"])
            rows, dropped_one = label_hits(task, question, item["hits"], positive, True)
            dropped += dropped_one
            task_rows.extend(rows)
            if task == "multihop_rag":
                raw, _ = label_hits(task, question, item["hits"], None, False)
                unfiltered_mh_pos.extend([row for row in raw if row["label"] == 1])
        pos = sum(row["label"] == 1 for row in task_rows)
        neg = sum(row["label"] == 0 for row in task_rows)
        per_task[task] = {"positive": pos, "negative": neg, "dropped_gold_hits": dropped, "rows": len(task_rows)}
        print(f"PAIR_COUNTS {task} pos {pos} neg {neg} dropped_gold {dropped}", flush=True)
        filtered.extend(task_rows)
    if any(row["qid"] in tune or row["task"] == "qasper" for row in filtered):
        raise SystemExit("REFUSE tune or qasper row in pairs")
    kept, balanced = balance(filtered, SEED)
    pair_dir = os.path.join(FORMAL, "pairs")
    main_path = os.path.join(pair_dir, "evaluator_formal_v1_pairs.jsonl")
    with open(main_path, "w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    negatives = [row for row in kept if row["label"] == 0]
    other_pos = [row for row in kept if row["label"] == 1 and row["task"] != "multihop_rag"]
    ablation = negatives + other_pos + unfiltered_mh_pos
    ab_path = os.path.join(pair_dir, "evaluator_formal_v1_pairs.mh_unfiltered.jsonl")
    with open(ab_path, "w", encoding="utf-8") as handle:
        for row in ablation:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    steps = ((balanced["n_balanced"] + 11) // 12) * 8
    hours = steps * SEC_PER_STEP / 3600
    ab_pos = sum(row["label"] == 1 for row in ablation)
    ab_neg = sum(row["label"] == 0 for row in ablation)
    stats = {
        "seed": SEED,
        "per_task": per_task,
        "balance": balanced,
        "multihop_filter": detail,
        "qasper_in_training": False,
        "main_pairs": main_path,
        "main_sha256": hashlib.sha256(open(main_path, "rb").read()).hexdigest(),
        "ablation_pairs": ab_path,
        "ablation_positive": ab_pos,
        "ablation_negative": ab_neg,
        "ablation_n": len(ablation),
        "ablation_note": "Same balanced negatives and non-multihop positives as the main set. Multihop positives are every retrieved gold hit, not rebalanced. Training steps stay the main step count.",
        "main_steps": steps,
        "sec_per_step_pilot": SEC_PER_STEP,
        "main_gpu_hours_estimate": hours,
        "ablation_gpu_hours_estimate_same_steps": hours,
    }
    json.dump(stats, open(os.path.join(pair_dir, "pair_stats.json"), "w"), indent=1)
    print(
        f"PAIR_COUNTS_TOTAL pos {balanced['kept_each']} neg {balanced['kept_each']} n {balanced['n_balanced']}",
        flush=True,
    )
    print(f"ABLATION_PAIRS n {len(ablation)} pos {ab_pos} neg {ab_neg}", flush=True)
    print(f"GPU_HOURS_ESTIMATE main {hours:.2f} ablation_same_steps {hours:.2f}", flush=True)
    print("PAIRS_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("retrieve")
    p.add_argument("--task", required=True)
    sub.add_parser("pool")
    args = ap.parse_args()
    if args.cmd == "retrieve":
        if args.task not in TRAIN_TASKS:
            raise SystemExit(f"REFUSE task {args.task}")
        retrieve(args.task)
    else:
        pool()


if __name__ == "__main__":
    main()
