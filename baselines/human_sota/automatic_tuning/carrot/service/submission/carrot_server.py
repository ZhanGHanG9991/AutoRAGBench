import json
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from carrot_runtime_fe import Pipeline

WS = Path(__file__).resolve().parent.parent
CFG = json.loads((WS / "artifact.json").read_text())
P = Pipeline(WS, CFG["task"])
app = FastAPI()


@app.get("/health")
def health():
    return {"ok": True, "method": "CARROT-six-reranker", "protocol": "fixed-evidence-v1"}


@app.post("/retrieve")
def retrieve(req: dict):
    passages, stats = P.retrieve_passages(req["question"], req.get("meta") or {},
                                          top_k=CFG.get("top_k", 60), budget=CFG.get("budget", 1024))
    print("CARROT_RETRIEVE " + json.dumps(stats), flush=True)
    return {"passages": passages}


@app.post("/answer")
def answer(req: dict):
    # Kept for compatibility only; the formal fixed-evidence evaluator never calls this route.
    passages, stats = P.retrieve_passages(req["question"], req.get("meta") or {},
                                          top_k=CFG.get("top_k", 60), budget=CFG.get("budget", 1024))
    context = "\n\n".join(p["text"] for p in passages)
    text, usage = P.generate(req["question"], context, os.environ["LLM_BASE_URL"])
    stats["generator_usage"] = usage
    return {"answer": text, "citations": stats["selected_doc_ids"]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ["PORT"]), log_level="warning")
