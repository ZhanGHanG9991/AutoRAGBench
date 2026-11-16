import os
import json
import time
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn
from runtime import Engine
from fe_corpus import corpus_evidence, check_limits

engine = Engine(Path(__file__).resolve().parents[1])
app = FastAPI()

class Request(BaseModel):
    question: str
    meta: dict = Field(default_factory=dict)

@app.get("/health")
def health():
    return {"status": "ok", "method": "RAPTOR"}

@app.post("/answer")
def answer(request: Request):
    return engine.answer(request.question, request.meta)

@app.post("/retrieve")
def retrieve(request: Request):
    """Expand ranked RAPTOR nodes to original corpus documents in rank order."""
    started = time.monotonic()
    _context, selected, tree = engine.retrieve(request.question, request.meta)
    evidence = corpus_evidence()
    passages = []
    seen = set()
    chars = 0
    capped = False
    for item in selected:
        node = tree.all_nodes[item["node_index"]]
        for raw_doc_id in getattr(node, "doc_ids", []):
            doc_id = str(raw_doc_id)
            if doc_id in seen:
                continue
            seen.add(doc_id)
            row = evidence.document(doc_id=doc_id)
            title = str(row.get("title") or "")
            text = str(row.get("text") or "")
            remaining = 500000 - chars - len(title)
            if len(passages) >= 256 or remaining <= 0:
                capped = True
                break
            if len(text) > remaining:
                text = text[:remaining]
                capped = True
            if text.strip():
                passages.append(evidence.passage(text, doc_id=doc_id))
                chars += len(title) + len(text)
            if capped:
                break
        if capped:
            break
    print("RAPTOR_FIXED_EVIDENCE " + json.dumps({
        "selected_nodes": len(selected),
        "summary_nodes": sum(x["layer_number"] > 0 for x in selected),
        "passages": len(passages),
        "capped": capped,
        "seconds": time.monotonic() - started,
    }), flush=True)
    return {"passages": check_limits(passages)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ["PORT"]))
