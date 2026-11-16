"""AutoRAGBench service: GET /health, POST /answer. Final answer comes from the generator."""

def _release_path(value):
    """Resolve portable build/workspace paths without changing their layout."""
    import os
    from pathlib import Path
    workspace = os.environ.get("WORKSPACE", str(Path(__file__).resolve().parents[1]))
    root = os.environ.get("AUTORAGBENCH_ROOT", workspace)
    model_dir = os.environ.get("MODEL_DIR", str(Path(workspace) / "models"))
    for key, replacement in (("@WORKSPACE@", workspace), ("@AUTORAGBENCH_ROOT@", root),
                             ("@MODEL_DIR@", model_dir), ("@BUILD_TMP@", os.environ.get("BUILD_TMP", "/tmp"))):
        value = value.replace(key, replacement)
    return value

import json
import os
import sys
import time

import uvicorn
from fastapi import FastAPI
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bm25_index import build_or_load, search  # noqa: E402
from crag_corpus import ANSWER_PROMPT, REWRITE_PROMPT, build_knowledge, chat, decide_action, load_evaluator  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get(
    "CRAG_ROOT",
    _release_path("@AUTORAGBENCH_ROOT@/humansota/CorrectiveRAG"),
)
CFG_PATH = os.environ.get("CRAG_CONFIG") or os.path.join(HERE, "crag_config.json")
CFG = json.load(open(CFG_PATH, encoding="utf-8"))
CORPUS = os.environ["CORPUS_DIR"]
INDEX = os.environ.get("CRAG_INDEX") or os.path.join(REPO, "rageng", "index", "musique-pilot")
KIND = os.environ.get("CRAG_EVALUATOR_KIND", "t5")
EVAL = os.environ.get("CRAG_EVALUATOR") or os.path.join(REPO, "models", "evaluator")
FIRST = os.environ.get("CRAG_FIRST_RETRIEVER", "bm25")
DOCS, RETRIEVER = None, None
CONTRIEVER = None
if FIRST == "bm25":
    DOCS, RETRIEVER = build_or_load(CORPUS, INDEX)
elif FIRST == "contriever":
    from contriever_index import load_built
    CONTRIEVER = load_built(
        os.environ.get("CRAG_CONTRIEVER_INDEX") or os.path.join(REPO, "rageng", "index", "musique-contriever"),
        os.environ.get("CRAG_CONTRIEVER") or os.path.join(REPO, "models", "contriever-msmarco"),
    )
else:
    raise SystemExit(f"unknown CRAG_FIRST_RETRIEVER {FIRST}")
EV = load_evaluator(KIND, EVAL)
print(f"EVALUATOR tag={EV.tag} temporary={EV.temporary} path={EVAL}", file=sys.stderr, flush=True)
print(f"FIRST_RETRIEVER {FIRST}", file=sys.stderr, flush=True)
print(
    "REWRITE_RETRIEVER " + ("contriever" if CONTRIEVER is not None else "bm25"),
    file=sys.stderr, flush=True,
)
print(
    "EVALUATOR_INPUT sep=' [SEP] ' max_length=512 truncation=True score=raw_logit empty_doc=-1",
    file=sys.stderr, flush=True,
)
if os.environ.get("CRAG_UPPER"):
    CFG["upper_threshold"] = float(os.environ["CRAG_UPPER"])
if os.environ.get("CRAG_LOWER"):
    CFG["lower_threshold"] = float(os.environ["CRAG_LOWER"])
if os.environ.get("CRAG_REQUIRE_THRESHOLD") == "1":
    if CFG.get("threshold_status") != "ok" or CFG.get("evaluator") != "t5" or CFG.get("evaluator_temporary"):
        raise SystemExit(
            "refusing official run: crag_config.json is not an accepted T5 train fit "
            f"(status={CFG.get('threshold_status')})"
        )
if os.environ.get("CRAG_CALIBRATE") == "1":
    if FIRST != "bm25" or RETRIEVER is None:
        raise SystemExit("CRAG_CALIBRATE stays unset for contriever runs")
    train = os.environ["CRAG_CALIBRATE_TRAIN"]
    n = int(os.environ.get("CRAG_CALIBRATE_N", "16"))
    qs = [json.loads(line) for line in open(train, encoding="utf-8")][:n]
    rows = []
    for q in qs:
        hits = search(DOCS, RETRIEVER, q["question"], int(CFG["n_docs"]))
        rows.append(EV.score_docs(q["question"], [h["passage"] for h in hits]))
    maxes = sorted(max(r) for r in rows)
    upper = maxes[min(len(maxes) - 1, int(round(0.66 * (len(maxes) - 1))))]
    lower = maxes[min(len(maxes) - 1, int(round(0.33 * (len(maxes) - 1))))]
    if not lower < upper:
        lower = upper - 1.0
    CFG["upper_threshold"] = upper
    CFG["lower_threshold"] = lower
    counts = {0: 0, 1: 0, 2: 0}
    for r in rows:
        action, _ = decide_action(r, int(CFG["n_docs"]), upper, lower)
        counts[action] += 1
    print("CALIBRATE_TEMPORARY", json.dumps({
        "evaluator": EV.tag, "temporary": EV.temporary, "n_train": len(qs),
        "upper": upper, "lower": lower,
        "counts": {"correct": counts[2], "ambiguous": counts[1], "incorrect": counts[0]},
    }), file=sys.stderr, flush=True)
CLIENT = OpenAI(
    base_url=os.environ["LLM_BASE_URL"],
    api_key=os.environ.get("VLLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY",
)
app = FastAPI()
from fe_corpus import corpus_evidence
_FE_EVIDENCE = corpus_evidence()  # Startup, outside question budget.


def _retrieved(hits, scores=None):
    rows = []
    for i, hit in enumerate(hits):
        row = {
            "doc_id": hit["doc_id"],
            "title": hit["title"],
        }
        if "bm25" in hit:
            row["bm25"] = round(float(hit["bm25"]), 4)
        if "dot" in hit:
            row["dot"] = round(float(hit["dot"]), 4)
        if "row" in hit:
            row["row"] = int(hit["row"])
        if scores is not None and i < len(scores):
            row["t5"] = scores[i]
        rows.append(row)
    return rows


def _top_n():
    raw = os.environ.get("CRAG_TOP_N")
    return int(raw) if raw else None


def _first(question, k, doc_id):
    if CONTRIEVER is not None:
        return CONTRIEVER.search(question, k, only_doc_id=doc_id)
    return search(DOCS, RETRIEVER, question, k, only_doc_id=doc_id)


def second_search(query, k, doc_id):
    """Rewrite retrieval. Contriever mode reuses that dense index. BM25 mode stays BM25."""
    if CONTRIEVER is not None:
        return CONTRIEVER.search(query, k, only_doc_id=doc_id)
    return search(DOCS, RETRIEVER, query, k, only_doc_id=doc_id)


def answer_one(question, qid=None, doc_id=None):
    t0 = time.time()
    tokens = 0
    n_docs = int(CFG["n_docs"])
    top_n = _top_n()
    first = _first(question, n_docs, doc_id)
    info = build_knowledge(
        EV, question, [h["passage"] for h in first], None,
        CFG["upper_threshold"], CFG["lower_threshold"], n_docs, CFG["decompose_mode"], top_n,
    )
    calls = 0
    rewrite = None
    second = []
    citations = [h["doc_id"] for h in first]
    if info.get("need_rewrite"):
        rewrite, used = chat(CLIENT, REWRITE_PROMPT.format(question=question), 64)
        tokens += used
        calls += 1
        second = second_search(rewrite, int(CFG["rewrite_k"]), doc_id)
        info = build_knowledge(
            EV, question, [h["passage"] for h in first], [h["passage"] for h in second],
            CFG["upper_threshold"], CFG["lower_threshold"], n_docs, CFG["decompose_mode"], top_n,
        )
        if info["action_name"] == "incorrect":
            citations = [h["doc_id"] for h in second]
        else:
            citations = citations + [h["doc_id"] for h in second]
    prompt = ANSWER_PROMPT.format(docs=info.get("knowledge") or "", question=question)
    answer_format = os.environ.get("CRAG_ANSWER_FORMAT")
    if answer_format:
        prompt += "\nAnswer format: " + answer_format
    ans, used = chat(CLIENT, prompt, 128)
    tokens += used
    calls += 1
    seconds = round(time.time() - t0, 3)
    trace = {
        "qid": qid,
        "question": question,
        "evaluator": EV.tag,
        "temporary_evaluator": EV.temporary,
        "first_retriever": FIRST,
        "second_retriever": "contriever" if CONTRIEVER is not None else "bm25",
        "top_n": top_n if top_n is not None else (3 if CFG.get("decompose_mode") == "selection" else 6),
        "doc_id": doc_id,
        "upper": CFG["upper_threshold"],
        "lower": CFG["lower_threshold"],
        "retrieval": _retrieved(first, info.get("scores")),
        "action": info["action_name"],
        "rewrite": rewrite,
        "second_retrieval": _retrieved(second),
        "knowledge": info.get("knowledge") or "",
        "answer": ans,
        "calls": calls,
        "tokens": tokens,
        "seconds": seconds,
    }
    line = {
        "evaluator": EV.tag, "temporary_evaluator": EV.temporary,
        "action": info["action_name"], "calls": calls, "tokens": tokens,
        "max_score": info.get("max_score"), "seconds": seconds,
        "over_calls": calls > 4, "over_tokens": tokens > 32000, "over_seconds": seconds > 60,
    }
    print("CRAG", json.dumps(line), file=sys.stderr, flush=True)
    trace_path = os.environ.get("CRAG_TRACE")
    if trace_path:
        with open(trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")
    return {"answer": ans, "citations": citations}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/answer")
def answer(req: dict):
    meta = req.get("meta") or {}
    return answer_one(req["question"], req.get("qid"), doc_id=meta.get("doc_id") or None)


@app.post("/retrieve")
def retrieve(req: dict):
    from fe_refine import refine_hits
    from fe_corpus import check_limits
    question=req['question'];doc_id=(req.get('meta') or {}).get('doc_id') or None
    n_docs=int(CFG['n_docs']);first=_first(question,n_docs,doc_id)
    scores=EV.score_docs(question,[h['passage'] for h in first][:n_docs])
    action,_=decide_action(scores,n_docs,CFG['upper_threshold'],CFG['lower_threshold'])
    top_n=_top_n()
    if top_n is None:top_n=3 if CFG['decompose_mode']=='selection' else 6
    if action==2:
        passages=refine_hits(EV,question,first[:n_docs],CFG['decompose_mode'],top_n)
    else:
        rewrite,_=chat(CLIENT,REWRITE_PROMPT.format(question=question),64)
        second=second_search(rewrite,int(CFG['rewrite_k']),doc_id)
        # Original build_knowledge refines external first, then internal.
        external=refine_hits(EV,question,second,CFG['decompose_mode'],top_n)
        if action==0:passages=external
        else:passages=refine_hits(EV,question,first[:n_docs],CFG['decompose_mode'],top_n)+external
    return {'passages':check_limits(passages)}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ["PORT"]), log_level="warning")
