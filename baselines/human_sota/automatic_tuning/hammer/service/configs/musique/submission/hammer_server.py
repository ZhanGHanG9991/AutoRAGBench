"""HAMMER service exposing both the legacy answer path and fixed evidence."""
import json
import os
import sys
import threading
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(WORKSPACE / "submission")]
import hammer_rageng as hr  # noqa: E402

ARTIFACT = json.loads((WORKSPACE / "artifact.json").read_text())
TASK = ARTIFACT["task"]
hr.setup(WORKSPACE / "upstream", os.environ["LLM_BASE_URL"], WORKSPACE / "index", TASK)
hr._state["require_cache"] = True

from hammer import huggingface_helper as _hh  # noqa: E402
_loaded, _load = {}, _hh.get_hf_embedding_model


def _shared_embedding_model(name, *args, **kwargs):
    key = (name, str(kwargs.get("device")))
    if key not in _loaded:
        _loaded[key] = _load(name, *args, **kwargs)
    return _loaded[key]


_hh.get_hf_embedding_model = _shared_embedding_model

import uvicorn  # noqa: E402
from fastapi import FastAPI, Request  # noqa: E402
from hammer.tuner.main_tuner_tpe import FlowBuilder  # noqa: E402
from hammer.tuner import main_tuner_mcts as mt  # noqa: E402
from hammer.utils import optimized_rag_prompt_builder as orpb  # noqa: E402


from hammer.mcts.mcts_dataset_loader import SimpleDataset  # noqa: E402


FROZEN_DATASET = SimpleDataset(
    str(WORKSPACE / "data/corpus.json"),
    str(WORKSPACE / "data/qa.jsonl"),
    TASK,
)


class _Study:
    dataset = FROZEN_DATASET
    search_space = mt.SimpleSearchSpace()
    model_config = {"extra": "forbid", "yaml_file": None}
    toy_mode = False
    timeouts = mt.SimpleTimeoutConfig()
    optimization = mt.SimpleOptimizationConfig()


FLOW = FlowBuilder(_Study()).build_flow(ARTIFACT["params"])
BUILDER = orpb.create_optimized_rag_prompt_builder(FLOW, max_workers=1)
LOCK = threading.Lock()
app = FastAPI()


def _build(question, meta):
    if TASK == "qasper" and not meta.get("doc_id"):
        raise ValueError("QASPER requires meta.doc_id")
    started = time.monotonic()
    hr.set_question_metas([meta])
    built = BUILDER.batch_build_prompts([question])
    nodes = built.processed_nodes_list[0] or []
    return built, nodes, started


def _node_passages(nodes):
    passages, seen = [], set()
    for scored in nodes:
        node = scored.node
        meta = node.metadata or {}
        doc_id = meta.get("id") or meta.get("doc_id")
        text = node.text
        if doc_id is None or not text:
            continue
        key = (str(doc_id), text)
        if key in seen:
            continue
        seen.add(key)
        passages.append({"doc_id": str(doc_id), "title": meta.get("title", "") or "", "text": text})
        if len(passages) == 256:
            break
    if not passages:
        raise RuntimeError("HAMMER returned empty evidence")
    return passages


@app.get("/health")
def health():
    return {"ok": True, "method": "HAMMER", "protocol": "fixed-evidence-v1"}


@app.post("/retrieve")
async def retrieve(request: Request):
    payload = await request.json()
    question, meta = payload["question"], payload.get("meta") or {}
    with LOCK:
        _, nodes, started = _build(question, meta)
        passages = _node_passages(nodes)
        trace = {"seconds": time.monotonic() - started, "n_context": len(nodes),
                 "passages": len(passages), "citations": list(dict.fromkeys(p["doc_id"] for p in passages))}
        with open(Path(os.environ["SCRATCH_DIR"]) / "retrieve_trace.jsonl", "a") as stream:
            stream.write(json.dumps(trace, ensure_ascii=False) + "\n")
        print("HAMMER_RETRIEVE " + json.dumps(trace), flush=True)
    return {"passages": passages}


@app.post("/answer")
async def answer(request: Request):
    payload = await request.json()
    question, meta = payload["question"], payload.get("meta") or {}
    with LOCK:
        built, nodes, started = _build(question, meta)
        _, response, _, _ = hr.call_generator(built.final_prompts[0])
        result = {"answer": hr.final_answer(response),
                  "citations": list(dict.fromkeys(n.node.metadata.get("id") for n in nodes
                                                   if n.node.metadata.get("id")))}
        with open(Path(os.environ["SCRATCH_DIR"]) / "answer_trace.jsonl", "a") as stream:
            stream.write(json.dumps({"seconds": time.monotonic() - started, "n_context": len(nodes),
                                     "citations": result["citations"]}, ensure_ascii=False) + "\n")
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ["PORT"]), log_level="warning")
