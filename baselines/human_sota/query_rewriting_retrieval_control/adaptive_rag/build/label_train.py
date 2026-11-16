"""Run A, B and C on a public train split and write silver / bias labels.

Resumes from the output jsonl. Does not read dev or test.
Retrieval is locked; the generator calls run concurrently.
"""
import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from guard import (
    assert_task,
    bias_label,
    corpus_dir,
    formal_expect,
    formal_index,
    formal_jsonl,
    info,
    public_jsonl,
    refuse_index_build,
)
from llm import chat
from ragcore import Retriever
from scoring import is_correct
from strategies import Budget, run_a, run_b, run_c

HERE = os.path.dirname(os.path.abspath(__file__))
ROUTES = ("A", "B", "C")


class LockedRetriever:
    def __init__(self, inner):
        self.inner = inner
        self.lock = threading.Lock()

    def search(self, *args, **kwargs):
        with self.lock:
            return self.inner.search(*args, **kwargs)

    def fmt(self, ids):
        with self.lock:
            return self.inner.fmt(ids)


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def label_row(row, task, fmt, retriever):
    question = row["question"]
    doc_id = (row.get("meta") or {}).get("doc_id")
    routes = {}

    def finish(name, answer, budget):
        ok, scores = is_correct(task, answer, row)
        snap = budget.snapshot()
        snap.update({
            "answer": answer,
            "ok": bool(ok) and not snap["error"],
            "scores": scores,
        })
        routes[name] = snap

    budget = Budget()
    answer, _raw = run_a(chat, budget, fmt, question)
    finish("A", answer, budget)

    budget = Budget()
    answer, _raw = run_b(chat, retriever, budget, fmt, question, doc_id)
    finish("B", answer, budget)

    budget = Budget()
    answer, _raw = run_c(chat, retriever, budget, fmt, question, doc_id)
    finish("C", answer, budget)

    if any(routes[name].get("error") for name in ROUTES):
        label, source = bias_label(task), "error"
    elif routes["A"]["ok"]:
        label, source = "A", "silver"
    elif routes["B"]["ok"]:
        label, source = "B", "silver"
    elif routes["C"]["ok"]:
        label, source = "C", "silver"
    else:
        label, source = bias_label(task), "bias"
    return {
        "qid": row["qid"],
        "question": question,
        "label": label,
        "source": source,
        "task": task,
        "routes": routes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="")
    parser.add_argument("--src", default="", choices=("", "formal"))
    args = parser.parse_args()
    task = assert_task(args.task)
    if args.src == "formal":
        refuse_index_build()
        src = formal_jsonl(task, "train_fit")
        expect = formal_expect(task, "train_fit")
        index = formal_index(task)
    else:
        src = public_jsonl(task, "train")
        expect = 500
        index = os.path.join(HERE, "index", task)
    rows = read_jsonl(src)
    if len(rows) != expect:
        raise SystemExit(f"expected {expect} train rows, found {len(rows)} in {src}")
    if args.limit:
        rows = rows[: args.limit]
    out = args.out or os.path.join(HERE, "labels", f"{task}_train.jsonl")
    if args.src == "formal":
        root = os.path.abspath(os.path.join(HERE, "formal-v1", "labels"))
        out_abs = os.path.abspath(out)
        if not out_abs.startswith(root + os.sep):
            raise SystemExit(f"refuse formal out {out}")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = {rec["qid"] for rec in read_jsonl(out)}
    todo = [row for row in rows if row["qid"] not in done]
    print(f"LABEL task={task} todo={len(todo)} done={len(done)} out={out}", flush=True)
    if not todo:
        return
    fmt = info(task)["answer_format"]
    corpus = None if args.src == "formal" else corpus_dir(task)
    retriever = LockedRetriever(Retriever(corpus, index=index, device="cpu"))
    started = time.time()
    finished = 0
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool, open(out, "a") as handle:
        futures = [pool.submit(label_row, row, task, fmt, retriever) for row in todo]
        for future in as_completed(futures):
            rec = future.result()
            with lock:
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
                handle.flush()
                finished += 1
                step = finished
            if step == 1 or step % 20 == 0 or step == len(todo):
                rate = step / max(time.time() - started, 1e-6)
                left_min = (len(todo) - step) / max(rate, 1e-6) / 60
                print(
                    f"LABEL progress {step}/{len(todo)} label={rec.get('label')} "
                    f"source={rec.get('source')} eta_min={left_min:.1f}",
                    flush=True,
                )
    print(f"LABEL wrote {out}", flush=True)


if __name__ == "__main__":
    main()
