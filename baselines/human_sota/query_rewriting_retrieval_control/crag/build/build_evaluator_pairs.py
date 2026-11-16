"""Build balanced Contriever top-10 pairs for the joint T5-large evaluator.

Uses each task's 400 evaluator-train questions from seed 20261018.
Qasper is not included. The 100 tune questions are not included.
Does not read dev or test and does not modify an existing index.

multihop_rag gold windows are filtered. A retrieved window of a gold article
is a positive only when it contains the answer or, for closed answers, only
when its question overlap is in the top m for that article and at least tau.
Failed gold windows are dropped, not labeled negative. m and tau are fixed
from the 400 evaluator-train questions.
"""
import hashlib
import json
import os
import random
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from evaluator_rageng_tasks import (  # noqa: E402
    CHUNK_WORDS,
    CONTRIEVER,
    EXPECTED_M,
    EXPECTED_TAU,
    FILTER_VERSION,
    OLD_POSITIVES,
    PAIR_STATS,
    PAIRS,
    SEED,
    SPLIT,
    TRAIN_TASKS,
    index_dir,
    refuse,
    train_path,
)

STOPWORDS = frozenset(
    """a an the of to and in on for with as at by from or is are was were be been
    being it this that these those not no nor but if then than so such into over
    after before about between within without during under above up down out off
    again further once here there all any both each few more most other some own
    same than too very can will just don should now do does did doing have has
    had having i you he she they we me him her them my your our their what which
    who whom whose when where why how""".split()
)
CLOSED_ANSWERS = frozenset({"yes", "no", "insufficient information"})
TOKEN = re.compile(r"[a-z0-9]+")


def evaluator_text(question, passage):
    """Same string the server scores: raw question, normalized passage."""
    if passage is None:
        return None
    doc = str(passage).replace("\n", " ").replace("\t", " ").strip()
    if not doc:
        return None
    return str(question) + " [SEP] " + doc


def norm_tokens(text):
    return TOKEN.findall(str(text).lower())


def content_tokens(text):
    return [tok for tok in norm_tokens(text) if tok not in STOPWORDS and len(tok) > 1]


def answer_class(answer):
    folded = " ".join(norm_tokens(answer))
    if not folded:
        raise SystemExit(f"empty normalized answer {answer!r}")
    if folded in CLOSED_ANSWERS:
        return "closed"
    return "entity"


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
    """Entity windows that contain the answer stay. Others need top-m overlap."""
    if kind == "entity" and contained:
        return True
    return rank < m_keep and overlap >= tau


def percentile_index(count):
    """Index of the nearest-rank 25th percentile: ceil(count / 4) - 1."""
    if count < 1:
        raise SystemExit("percentile of an empty list")
    return (count + 3) // 4 - 1


def load_docs(task):
    path = os.path.join(index_dir(task), "docs.jsonl")
    refuse(path)
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def calibrate_multihop(questions, docs):
    """Set m and tau from entity questions in the 400. Does not use holdout or dev."""
    rows_of = {}
    for row_id, doc in enumerate(docs):
        rows_of.setdefault(doc["doc_id"], []).append(row_id)
    counts = []
    overlaps = []
    for question in questions:
        if answer_class(question["answers"][0]) != "entity":
            continue
        if len(question.get("answers") or []) != 1:
            raise SystemExit(f"{question.get('qid')} does not have one answer")
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
    m_keep = counts[len(counts) // 2]
    tau = overlaps[percentile_index(len(overlaps))]
    detail = {
        "version": FILTER_VERSION,
        "m": m_keep,
        "tau": tau,
        "n_articles_with_containment": len(counts),
        "n_containment_windows": len(overlaps),
        "m_index": len(counts) // 2,
        "tau_index": percentile_index(len(overlaps)),
        "m_rule": "sorted per-article containment counts, drop c=0, take index n//2",
        "tau_rule": "sorted overlap of answer-containing windows, index ceil(n/4)-1",
        "closed_answers": sorted(CLOSED_ANSWERS),
        "candidates": "contriever_top10",
        "dropped_gold_windows_are_not_negatives": True,
        "calibration_split": "evaluator_train_400",
    }
    if m_keep != EXPECTED_M or tau != EXPECTED_TAU:
        raise SystemExit(f"calibration drifted m={m_keep} tau={tau!r} detail={detail}")
    if detail["n_articles_with_containment"] != 426 or detail["n_containment_windows"] != 1930:
        raise SystemExit(f"calibration counts drifted {detail}")
    return detail


def positive_rows_for_question(question, docs, rows_of, m_keep, tau):
    kind = answer_class(question["answers"][0])
    answer = question["answers"][0]
    kept = set()
    by_containment = 0
    by_overlap = 0
    gold_windows = 0
    for doc_id in dict.fromkeys(question.get("supporting_doc_ids") or []):
        scored = []
        for row_id in rows_of.get(doc_id, []):
            body = window_body(docs[row_id]["passage"])
            overlap = overlap_score(question["question"], body)
            contained = kind == "entity" and contains_answer(body, answer)
            scored.append((overlap, row_id, contained))
        scored.sort(key=lambda item: (-item[0], item[1]))
        gold_windows += len(scored)
        for rank, (overlap, row_id, contained) in enumerate(scored):
            if not keep_gold_window(kind, contained, rank, overlap, m_keep, tau):
                continue
            kept.add(row_id)
            if contained:
                by_containment += 1
            else:
                by_overlap += 1
    return kept, {
        "class": kind,
        "gold_windows": gold_windows,
        "candidate_positives": len(kept),
        "by_containment": by_containment,
        "by_overlap_only": by_overlap,
    }


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
    stats = {
        "positive_before": len(positive),
        "negative_before": len(negative),
        "kept_each": keep,
        "dropped_positive": len(positive) - keep,
        "dropped_negative": len(negative) - keep,
    }
    return kept, stats


def load_questions(task, qids):
    path = train_path(task)
    want = list(qids)
    found = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["qid"] in found or row["qid"] not in want:
                continue
            found[row["qid"]] = row
    missing = [qid for qid in want if qid not in found]
    if missing:
        raise SystemExit(f"{task} missing {len(missing)} evaluator-train qids")
    return [found[qid] for qid in want]


def search_top(index, questions, k=10, only_doc_ids=None):
    """Top-k by dot product. only_doc_ids restricts each question to one paper."""
    import numpy as np

    vectors = index.encode([row["question"] for row in questions], batch_size=32)
    hits_per_question = []
    full = None
    if only_doc_ids is None:
        full = index.embeddings @ vectors.T
    for col, _question in enumerate(questions):
        rows = None
        if only_doc_ids is None:
            column = full[:, col]
        else:
            rows = index.row_of.get(only_doc_ids[col], [])
            if not rows:
                hits_per_question.append([])
                continue
            column = index.embeddings[rows] @ vectors[col]
        take = min(k, column.shape[0])
        if take <= 0:
            hits_per_question.append([])
            continue
        order = np.argpartition(-column, take - 1)[:take]
        order = order[np.argsort(-column[order])]
        hits = []
        for local in order:
            row_id = int(local if rows is None else rows[int(local)])
            doc = index.docs[row_id]
            hits.append({
                "doc_id": doc["doc_id"],
                "passage": doc["passage"],
                "dot": float(column[int(local)]),
                "row": row_id,
            })
        hits_per_question.append(hits)
    return hits_per_question


def label_hits(task, question, hits, positive_rows=None):
    gold = set(question.get("supporting_doc_ids") or [])
    rows = []
    skipped = 0
    dropped_gold = 0
    seen = set()
    for hit in hits:
        text = evaluator_text(question["question"], hit["passage"])
        if text is None:
            skipped += 1
            continue
        key = (hit["doc_id"], text)
        if key in seen:
            continue
        seen.add(key)
        in_gold = hit["doc_id"] in gold
        if task == "multihop_rag" and in_gold:
            if positive_rows is None or hit["row"] not in positive_rows:
                dropped_gold += 1
                continue
        rows.append({
            "task": task,
            "qid": question["qid"],
            "doc_id": hit["doc_id"],
            "label": 1 if in_gold else 0,
            "text": text,
        })
    return rows, skipped, dropped_gold


def pairs_for_task(task, questions, multihop_detail=None):
    import torch
    from contriever_index import load_built

    directory = index_dir(task)
    stamp_path = os.path.join(directory, "contriever.json")
    stamp = json.load(open(stamp_path, encoding="utf-8"))
    expected = CHUNK_WORDS[task]
    if stamp.get("chunk_words") != expected:
        raise SystemExit(f"{task} index chunk_words={stamp.get('chunk_words')!r} expected {expected!r}")
    positive_by_qid = {}
    filter_counts = None
    if task == "multihop_rag":
        docs = load_docs(task)
        rows_of = {}
        for row_id, doc in enumerate(docs):
            rows_of.setdefault(doc["doc_id"], []).append(row_id)
        filter_counts = {
            "retrieved_gold_hits": 0,
            "dropped_gold_hits": 0,
            "eligible_by_containment": 0,
            "eligible_by_overlap_only": 0,
            "closed_questions": 0,
            "entity_questions": 0,
        }
        for question in questions:
            if len(question.get("answers") or []) != 1:
                raise SystemExit(f"{question.get('qid')} does not have one answer")
            kept, info = positive_rows_for_question(
                question, docs, rows_of, multihop_detail["m"], multihop_detail["tau"]
            )
            positive_by_qid[question["qid"]] = kept
            filter_counts["eligible_by_containment"] += info["by_containment"]
            filter_counts["eligible_by_overlap_only"] += info["by_overlap_only"]
            filter_counts[info["class"] + "_questions"] += 1
    index = load_built(directory, CONTRIEVER)
    try:
        if task == "multihop_rag" and [doc["doc_id"] for doc in index.docs] != [
            doc["doc_id"] for doc in docs
        ]:
            raise SystemExit("multihop docs.jsonl order does not match the loaded index")
        retrieved = search_top(index, questions, 10)
    finally:
        del index
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    rows = []
    skipped = 0
    dropped_gold = 0
    with_positive = 0
    for question, hits in zip(questions, retrieved):
        task_rows, skipped_one, dropped_one = label_hits(
            task, question, hits, positive_by_qid.get(question["qid"])
        )
        skipped += skipped_one
        dropped_gold += dropped_one
        if any(row["label"] == 1 for row in task_rows):
            with_positive += 1
        rows.extend(task_rows)
    stats = {
        "n_questions": len(questions),
        "pairs": len(rows),
        "positive": sum(row["label"] == 1 for row in rows),
        "negative": sum(row["label"] == 0 for row in rows),
        "skipped_empty": skipped,
        "dropped_gold_hits": dropped_gold,
        "questions_with_positive": with_positive,
        "index": directory,
        "chunk_words": expected,
        "index_n_docs": stamp.get("n_docs"),
    }
    if filter_counts is not None:
        filter_counts["retrieved_gold_hits"] = stats["positive"] + dropped_gold
        filter_counts["dropped_gold_hits"] = dropped_gold
        stats["multihop_filter_counts"] = filter_counts
    return rows, stats


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def backup_unfiltered():
    """Keep the pre-filter pairs. Do not overwrite an existing backup."""
    if not os.path.exists(PAIRS):
        return
    stats = {}
    if os.path.exists(PAIR_STATS):
        stats = json.load(open(PAIR_STATS, encoding="utf-8"))
    if stats.get("multihop_filter", {}).get("version") == FILTER_VERSION:
        return
    for source, name in (
        (PAIRS, "evaluator_rageng_pairs.unfiltered.jsonl"),
        (PAIR_STATS, "evaluator_rageng_pair_stats.unfiltered.json"),
    ):
        if not os.path.exists(source):
            continue
        target = os.path.join(os.path.dirname(source), name)
        if os.path.exists(target):
            continue
        shutil.copy2(source, target)
        print("BACKED_UP", target, flush=True)


def reusable(stats):
    filt = stats.get("multihop_filter") or {}
    return (
        stats.get("seed") == SEED
        and filt.get("version") == FILTER_VERSION
        and filt.get("m") == EXPECTED_M
        and filt.get("tau") == EXPECTED_TAU
        and stats.get("n_balanced")
    )


def build():
    if not __import__("torch").cuda.is_available():
        raise SystemExit("refusing to retrieve without a GPU")
    for path in (SPLIT, PAIRS, PAIR_STATS, CONTRIEVER):
        refuse(path)
    split = json.load(open(SPLIT, encoding="utf-8"))
    if split.get("seed") != SEED:
        raise SystemExit(f"split seed {split.get('seed')}")
    backup_unfiltered()
    tune = set()
    multihop_questions = load_questions("multihop_rag", split["tasks"]["multihop_rag"]["evaluator_train"])
    if set(split["tasks"]["multihop_rag"]["tune"]) & {row["qid"] for row in multihop_questions}:
        raise SystemExit("multihop calibration saw a tune question")
    detail = calibrate_multihop(multihop_questions, load_docs("multihop_rag"))
    print("MULTIHOP_FILTER", json.dumps(detail, ensure_ascii=False), flush=True)
    pooled = []
    per_task = {}
    for task in TRAIN_TASKS:
        spec = split["tasks"][task]
        if len(spec["evaluator_train"]) != 400 or len(spec["tune"]) != 100:
            raise SystemExit(f"{task} split is not 400/100")
        tune.update(spec["tune"])
        if set(spec["evaluator_train"]) & set(spec["tune"]):
            raise SystemExit(f"{task} train and tune overlap")
        questions = multihop_questions if task == "multihop_rag" else load_questions(
            task, spec["evaluator_train"]
        )
        rows, stats = pairs_for_task(task, questions, detail if task == "multihop_rag" else None)
        old = OLD_POSITIVES[task]
        print(
            "PAIRS", task, "pos", stats["positive"], "neg", stats["negative"],
            "with_positive_questions", stats["questions_with_positive"],
            "dropped_gold", stats["dropped_gold_hits"],
            flush=True,
        )
        print(f"POSITIVE_COMPARE {task} old={old} new={stats['positive']}", flush=True)
        if task != "multihop_rag" and stats["positive"] != old:
            raise SystemExit(f"{task} positive count drifted from {old} to {stats['positive']}")
        pooled.extend(rows)
        per_task[task] = stats
    if any(row["task"] == "qasper" or row["qid"] in tune for row in pooled):
        raise SystemExit("tune or qasper question entered the training pairs")
    kept, balanced = balance(pooled, SEED)
    if any(row["qid"] in tune for row in kept):
        raise SystemExit("tune question survived balancing")
    compare = {
        task: {"old": OLD_POSITIVES[task], "new": per_task[task]["positive"]}
        for task in TRAIN_TASKS
    }
    payload = {
        "seed": SEED,
        "split": SPLIT,
        "tasks": per_task,
        "qasper_in_training_pairs": False,
        "multihop_filter": detail,
        "positive_vs_unfiltered": compare,
        "pooled_before": {
            "positive": balanced["positive_before"],
            "negative": balanced["negative_before"],
        },
        "balanced": balanced,
        "n_balanced": len(kept),
    }
    os.makedirs(os.path.dirname(PAIRS), exist_ok=True)
    tmp_pairs = PAIRS + ".tmp"
    with open(tmp_pairs, "w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp_pairs, PAIRS)
    payload["pairs_sha256"] = file_sha256(PAIRS)
    tmp_stats = PAIR_STATS + ".tmp"
    with open(tmp_stats, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)
        handle.write("\n")
    os.replace(tmp_stats, PAIR_STATS)
    print("BALANCED", balanced, flush=True)
    print("WROTE", PAIRS, flush=True)
    return payload


def self_test():
    pairs = [{"label": 1, "qid": f"p{i}"} for i in range(5)]
    pairs += [{"label": 0, "qid": f"n{i}"} for i in range(2)]
    kept, stats = balance(pairs, SEED)
    assert stats["kept_each"] == 2 and len(kept) == 4
    assert sum(row["label"] for row in kept) == 2
    again, _ = balance(pairs, SEED)
    assert [row["qid"] for row in kept] == [row["qid"] for row in again]
    assert evaluator_text("q", "a\nb") == "q [SEP] a b"
    assert evaluator_text("q", "  ") is None
    assert answer_class("Yes") == "closed"
    assert answer_class("no") == "closed"
    assert answer_class("Insufficient information.") == "closed"
    assert answer_class("Google.") == "entity"
    assert answer_class("Sam Bankman-Fried") == "entity"
    assert contains_answer("Sam Bankman-Fried left", "Sam Bankman-Fried")
    assert not contains_answer("instrument", "True")
    assert contains_answer("Google said so", "Google.")
    assert keep_gold_window("entity", True, 9, 0.0, 2, 0.08)
    assert not keep_gold_window("entity", False, 2, 0.9, 2, 0.08)
    assert keep_gold_window("entity", False, 1, 0.08, 2, 0.08)
    assert not keep_gold_window("closed", False, 0, 0.07, 2, 0.08)
    assert keep_gold_window("closed", False, 0, 0.2, 2, 0.08)
    assert percentile_index(1930) == 482
    assert percentile_index(1) == 0
    question = {
        "qid": "q",
        "question": "where is the answer",
        "answers": ["Yes"],
        "supporting_doc_ids": ["d"],
    }
    hits = [
        {"doc_id": "d", "passage": "yes the whole article", "row": 0},
        {"doc_id": "other", "passage": "unrelated page", "row": 1},
    ]
    rows, skipped, dropped = label_hits("multihop_rag", question, hits, positive_rows=set())
    assert dropped == 1 and skipped == 0
    assert [row["label"] for row in rows] == [0]
    rows, _, dropped = label_hits("multihop_rag", question, hits, positive_rows={0})
    assert dropped == 0 and [row["label"] for row in rows] == [1, 0]
    plain = {
        "qid": "q",
        "question": "q",
        "supporting_doc_ids": ["d"],
    }
    rows, _, dropped = label_hits(
        "musique", plain, [{"doc_id": "d", "passage": "gold", "row": 3}], None
    )
    assert dropped == 0 and rows[0]["label"] == 1
    print("PAIR_SELF_TEST_OK")


def calibrate_only():
    split = json.load(open(SPLIT, encoding="utf-8"))
    questions = load_questions("multihop_rag", split["tasks"]["multihop_rag"]["evaluator_train"])
    detail = calibrate_multihop(questions, load_docs("multihop_rag"))
    print("CALIBRATE_OK", json.dumps(detail, ensure_ascii=False), flush=True)


def main():
    if "--self-test" in sys.argv:
        self_test()
        return
    if "--calibrate-only" in sys.argv:
        calibrate_only()
        return
    if os.path.exists(PAIRS) and os.path.exists(PAIR_STATS):
        stats = json.load(open(PAIR_STATS, encoding="utf-8"))
        if reusable(stats):
            print("PAIRS_REUSED", stats["n_balanced"], stats["balanced"], flush=True)
            for task, item in stats.get("positive_vs_unfiltered", {}).items():
                print(f"POSITIVE_COMPARE {task} old={item['old']} new={item['new']}", flush=True)
            return
    build()


if __name__ == "__main__":
    main()
