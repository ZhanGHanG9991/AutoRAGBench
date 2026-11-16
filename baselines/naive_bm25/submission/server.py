"""Naive RAG baseline (s_naive): BM25 top-5 passages + one generator call."""
import os

import uvicorn
from fastapi import FastAPI

from ragcore import Retriever, answer_prompt, clean_answer, generate

R = Retriever(os.environ["CORPUS_DIR"])
app = FastAPI()


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/retrieve")
def retrieve(req: dict):
    q = req["question"]
    idx = R.search(q, k=5, doc_id=(req.get("meta") or {}).get("doc_id"))
    return {"passages": [{"doc_id": str(R.doc_list[int(R.doc_of[i])]),
                           "title": R.store.get(i)[0], "text": R.store.get(i)[1]} for i in idx]}


@app.post("/answer")
def answer(req: dict):
    q = req["question"]
    idx = R.search(q, k=5, doc_id=(req.get("meta") or {}).get("doc_id"))
    text = clean_answer(generate(answer_prompt(q, R.fmt(idx))))
    return {"answer": text, "citations": R.doc_ids(idx)}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ["PORT"]), log_level="warning")
