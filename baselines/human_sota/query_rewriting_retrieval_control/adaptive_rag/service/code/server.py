"""Adaptive-RAG service. GET /health, POST /answer.

The final answer is trimmed generator text. The classifier only picks A, B or C.
"""
import json
import os
import threading

import uvicorn
from fastapi import FastAPI

from classify import Classifier
from guard import assert_task, corpus_dir, info
from llm import chat, set_question_clock
from ragcore import Retriever
from strategies import Budget, run_a, run_b, run_c

HERE = os.path.dirname(os.path.abspath(__file__))
ARAG = os.path.dirname(HERE)


def task_name():
    forced = os.environ.get("ARAG_TASK")
    if forced:
        return assert_task(forced)
    corpus = os.environ.get("CORPUS_DIR", "")
    parts = os.path.abspath(corpus).split(os.sep)
    if "public" in parts:
        name = parts[parts.index("public") - 1]
        return assert_task(name)
    raise SystemExit("cannot tell which task this corpus is")


TASK = task_name()
INFO = info(TASK)
INDEX = os.environ.get("ARAG_INDEX") or os.path.join(HERE, "index", TASK)
CLF_PATH = os.environ.get("ARAG_CLASSIFIER") or os.path.join(ARAG, "models", "classifier")
CORPUS = os.environ.get("CORPUS_DIR") or corpus_dir(TASK)
RETRIEVER = Retriever(CORPUS, index=INDEX, device="cpu")
CLF = Classifier(CLF_PATH, TASK)
TRACE = os.environ.get("ARAG_TRACE", "")
LOCK = threading.Lock()
app = FastAPI()
from fe_corpus import corpus_evidence
_FE_EVIDENCE = corpus_evidence()  # Startup, outside question budget.


def trace(row):
    if not TRACE:
        return
    with LOCK:
        with open(TRACE, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


@app.get("/health")
def health():
    return {"ok": True, "task": TASK, "classifier_params": CLF.params}


@app.post("/answer")
def answer(req: dict):
    question = req["question"]
    meta = req.get("meta") or {}
    doc_id = meta.get("doc_id")
    qid = req.get("qid")
    route, fell_back = CLF.predict(question)
    budget = Budget()
    # Same clock the 60s budget uses, so a retry can see time left.
    set_question_clock(budget.t0)
    try:
        fmt = INFO["answer_format"]
        if route == "A":
            text, _raw = run_a(chat, budget, fmt, question)
        elif route == "B":
            text, _raw = run_b(chat, RETRIEVER, budget, fmt, question, doc_id)
        else:
            route = "C"
            text, _raw = run_c(chat, RETRIEVER, budget, fmt, question, doc_id)
        snap = budget.snapshot()
        row = {
            "qid": qid,
            "route": route,
            "classifier_fallback": fell_back,
            "answer": text,
            **snap,
        }
        trace(row)
        return {
            "answer": text,
            "route": route,
            "calls": snap["calls"],
            "tokens": snap["tokens"],
            "seconds": snap["seconds"],
            "over_budget": snap["over_budget"],
            "error": snap["error"],
        }
    finally:
        set_question_clock(None)


@app.post("/retrieve")
def retrieve(req: dict):
    from fe_corpus import corpus_evidence, check_limits
    from strategies import TOP_K, MAX_REASON, MAX_PARAS, _call, reason_prompt, retrieval_query, ANSWER_RE, clean_answer
    question=req['question'];doc_id=(req.get('meta') or {}).get('doc_id')
    route,_=CLF.predict(question);budget=Budget();set_question_clock(budget.t0)
    collected=[];ids=[];reasoning=[];query=question
    try:
        if route=='A':return {'passages':[]}
        for _ in range(1 if route=='B' else MAX_REASON):
            if route!='B' and not budget.can_call(reserve_s=8):break
            for i in RETRIEVER.search(query,k=TOP_K,doc_id=doc_id):
                if i not in collected:collected.append(i)
                if i not in ids:ids.append(i)
            ids=ids[-MAX_PARAS:]
            if route=='B':break
            last=_call(chat,budget,reason_prompt(INFO['answer_format'],question,RETRIEVER.fmt(ids),reasoning),64)
            if not last:break
            reasoning.append(clean_answer(last) or last.strip().split('\n')[0])
            if ANSWER_RE.search(last):break
            query=retrieval_query(last,question)
        evidence=corpus_evidence();passages=[]
        for i in collected:
            _title,text=RETRIEVER.store.get(i)
            did=RETRIEVER.doc_list[RETRIEVER.doc_of[i]]
            if text.strip():passages.append(evidence.passage(text,doc_id=did))
        return {'passages':check_limits(passages)}
    finally:set_question_clock(None)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ["PORT"]), log_level="warning")
