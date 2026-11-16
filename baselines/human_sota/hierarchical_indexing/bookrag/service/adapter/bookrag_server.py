import json
import os
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from runtime import FixedLLM
from bookrag_engine_fe import FixedEvidenceEngine

W = Path(__file__).resolve().parent.parent
scratch = Path(os.environ["SCRATCH_DIR"])
scratch.mkdir(exist_ok=True, parents=True)
llm = FixedLLM(os.environ["LLM_BASE_URL"], output_tokens=1200)
engine = FixedEvidenceEngine(W / "index", W / "models", llm,
                             json.loads((W / "submission/task.json").read_text()))
app, lock = FastAPI(), threading.Lock()


@app.get("/health")
def health():
    return {"ok": True, "method": "BookRAG", "protocol": "fixed-evidence-v1"}


@app.post("/retrieve")
def retrieve(body: dict):
    with lock:
        try:
            passages, trace = engine.retrieve_passages(body["question"], body.get("meta") or {})
            with (scratch / "bookrag_retrieve_trace.jsonl").open("a") as stream:
                stream.write(json.dumps(trace, ensure_ascii=False) + "\n")
            print("BOOKRAG_RETRIEVE " + json.dumps(trace, ensure_ascii=False), flush=True)
            return {"passages": passages}
        except Exception as exc:
            import traceback
            traceback.print_exc()
            return JSONResponse(status_code=500, content={"error": str(exc)})


@app.post("/answer")
def answer(body: dict):
    with lock:
        try:
            result, trace = engine.answer(body["question"], body.get("meta") or {})
            return {"answer": result, "citations": []}
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ["PORT"]), log_level="warning")
