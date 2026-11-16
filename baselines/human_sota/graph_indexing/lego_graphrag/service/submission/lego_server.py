import json
import os

import uvicorn
from fastapi import FastAPI

from lego_text import INSTANCES, GenLLM, LegoText

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
TASK = json.load(open(os.path.join(HERE, "task.json")))
CFG = json.load(open(os.path.join(HERE, "lego.json")))
INST = INSTANCES[CFG["instance"]]
LLM = GenLLM(os.environ["LLM_BASE_URL"])
KW = {k: CFG[k] for k in ("ppr_nodes", "damping", "sae_window", "path_num", "n_passages", "llm_fallback") if k in CFG}
P = LegoText(os.path.join(WS, "index"), TASK, LLM, **KW)
app = FastAPI()


@app.get("/health")
def health():
    return {"ok": True, "method": "LEGO-GraphRAG", "protocol": "fixed-evidence-v1"}


@app.post("/retrieve")
def retrieve(req: dict):
    LLM.reset()
    paths, info = P.retrieve(req["question"], req.get("meta") or {}, INST)
    pids = P.path_passages(paths)
    passages, seen = [], set()
    for pid in pids:
        row = P.passages[pid]
        key = (str(row["doc_id"]), row["text"])
        if not row["text"] or key in seen:
            continue
        seen.add(key)
        passages.append({"doc_id": str(row["doc_id"]), "title": row.get("title", "") or "",
                         "text": row["text"]})
    if not passages:
        raise RuntimeError("LEGO-GraphRAG returned empty evidence")
    info.update({"passages": len(passages), "llm_calls": LLM.calls,
                 "llm_tokens": getattr(LLM.local, "tokens", 0), "instance": CFG["instance"]})
    print("LEGO_RETRIEVE " + json.dumps(info, ensure_ascii=False), flush=True)
    return {"passages": passages}


@app.post("/answer")
def answer(req: dict):
    LLM.reset()
    paths, _ = P.retrieve(req["question"], req.get("meta") or {}, INST)
    text, docs = P.generate(req["question"], paths, CFG["ctx"])
    return {"answer": text, "citations": docs}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ["PORT"]), log_level="warning")
