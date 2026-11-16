"""Reference RAG baseline (provisional s_ref): hybrid BM25 + bge-base dense retrieval (RRF) and bge-reranker
re-ranking, top-8 passages, one generator call. Multi-hop tasks (task.json "multi_hop") add one follow-up
retrieval round: the generator names the missing fact, its query retrieves 4 more passages (two calls)."""
import os

import uvicorn
from fastapi import FastAPI

from ragcore import TASK, Retriever, answer_prompt, clean_answer, generate

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = Retriever(os.environ["CORPUS_DIR"], dense_model=os.path.join(WS, "models", "bge-base-en-v1.5"),
              reranker=os.path.join(WS, "models", "bge-reranker-base"))
app = FastAPI()

FOLLOWUP = """Question: {q}

Passages found so far:
{ctx}

To answer the question, what single piece of information is still missing? Write one short search query
for it (a few words, e.g. an entity name plus the property). If nothing is missing, repeat the question.
Query:"""


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/retrieve")
def retrieve(req: dict):
    q = req["question"]
    doc = (req.get("meta") or {}).get("doc_id")
    idx = R.search(q, k=8, doc_id=doc)
    if TASK.get("multi_hop"):
        follow = clean_answer(generate(FOLLOWUP.format(q=q, ctx=R.fmt(idx[:5])), max_tokens=32))
        if follow:
            idx = idx[:6] + [i for i in R.search(follow, k=8, doc_id=doc) if i not in idx[:6]][:4]
    return {"passages": [{"doc_id": str(R.doc_list[int(R.doc_of[i])]),
                           "title": R.store.get(i)[0], "text": R.store.get(i)[1]} for i in idx]}


@app.post("/answer")
def answer(req: dict):
    q = req["question"]
    doc = (req.get("meta") or {}).get("doc_id")
    idx = R.search(q, k=8, doc_id=doc)
    if TASK.get("multi_hop"):
        follow = clean_answer(generate(FOLLOWUP.format(q=q, ctx=R.fmt(idx[:5])), max_tokens=32))
        if follow:
            idx = idx[:6] + [i for i in R.search(follow, k=8, doc_id=doc) if i not in idx[:6]][:4]
    text = clean_answer(generate(answer_prompt(q, R.fmt(idx)), max_tokens=64))
    return {"answer": text, "citations": R.doc_ids(idx)}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ["PORT"]), log_level="warning")
