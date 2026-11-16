import asyncio
import json
import os
import sys
import time

import uvicorn
from fastapi import FastAPI

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
TASK = json.load(open(os.path.join(HERE, "task.json")))
CFG = json.load(open(os.path.join(HERE, "cluerag.json")))
from cluerag_text import ClueRAG, SCOPE  # noqa: E402

RAG = ClueRAG(os.path.join(WS, "index"), os.path.join(WS, "models"), os.environ["LLM_BASE_URL"], TASK,
              os.path.join(os.environ.get("SCRATCH_DIR", "/tmp"), "cluerag"),
              max_tokens=CFG.get("max_tokens", 1024))
app = FastAPI()


async def retrieve_chunks(question, doc_id):
    token = SCOPE.set(doc_id)
    try:
        RAG.trace.clear()
        started = time.time()
        qid = "q"
        ner, _ = await RAG.retriever._ner_by_llm_async(texts=[(qid, question)])
        embedding = await asyncio.to_thread(RAG.retriever.embedder.get_embedding, [question])
        result = await RAG.retriever._process_single_query(
            qid, {"question": question, "answer": "", "ground_truth_chunks": []}, {qid: embedding[0]}, ner)
        passages, seen = [], set()
        for chunk_id in result["chunks"]:
            if not dict.__contains__(RAG.chunks, chunk_id):
                continue
            row = dict.__getitem__(RAG.chunks, chunk_id)
            text = row.get("text") or row.get("content") or ""
            prefixes = (f"Title: {row.get('title', '')}\nContent: ", "Content: ")
            for prefix in prefixes:
                if text.startswith(prefix):
                    text = text[len(prefix):]
                    break
            key = (str(row["doc_id"]), text)
            if not text or key in seen:
                continue
            seen.add(key)
            passages.append({"doc_id": str(row["doc_id"]), "title": row.get("title", "") or "",
                             "text": text})
        if not passages:
            raise RuntimeError("ClueRAG returned empty evidence")
        info = {"ner": len(ner.get(qid, [])), "layers": list(RAG.trace), "passages": len(passages),
                "seconds": round(time.time() - started, 2), "alpha": CFG.get("alpha", 1.0)}
        return passages, info
    finally:
        SCOPE.reset(token)


@app.get("/health")
def health():
    return {"ok": True, "method": "ClueRAG", "protocol": "fixed-evidence-v1"}


@app.post("/retrieve")
async def retrieve(req: dict):
    doc = (req.get("meta") or {}).get("doc_id") if TASK.get("uses_meta_doc_id") else None
    passages, info = await retrieve_chunks(req["question"], doc)
    print("CLUE_RETRIEVE " + json.dumps(info), file=sys.stderr, flush=True)
    return {"passages": passages}


@app.post("/answer")
async def answer(req: dict):
    doc = (req.get("meta") or {}).get("doc_id") if TASK.get("uses_meta_doc_id") else None
    ans, chunks, _, info = await RAG._answer(req["question"], doc)
    print("CLUE " + json.dumps(info), file=sys.stderr, flush=True)
    return {"answer": ans, "citations": RAG.doc_ids(chunks)}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ["PORT"]), log_level="warning")
