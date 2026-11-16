"""AutoRAGBench service for SubgraphRAG. Scheme (a) artifacts only.

One generator call per question. The answer text is the model's first line,
with an ``ans:`` / ``answer:`` prefix removed. Nothing is rewritten beyond that.
"""
from __future__ import annotations

import json
import os
import re
import time
import sys

# Resolve this source tree even when invoked through a tokenizer runpy wrapper.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from encode import sha_names
from encoder import embed_texts, load_encoder
from graph import KG, localize
from model import Retriever
from qwen_tok import OUTPUT_CAP, QwenPieceTokenizer, build_prompt, guard_triples

K = 100
CHAR_BUDGET = 24000
app = FastAPI()
from fe_corpus import corpus_evidence
_FE_EVIDENCE = corpus_evidence()  # Startup, outside question budget.
S = {}
GUARD = {"questions": 0, "triggered": 0, "dropped_triples": 0}
QWEN = None


def configured_k() -> int:
    raw = os.environ.get("SGRAG_K", str(K))
    k = int(raw)
    if k < 1:
        raise SystemExit(f"bad K {raw}")
    return k


class Query(BaseModel):
    question: str
    meta: dict | None = None
    qid: str | None = None


def _clean(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.S | re.I)
    text = text.strip()
    if not text:
        return ""
    line = text.splitlines()[0].strip()
    line = re.sub(r"^(ans|answer)\s*:\s*", "", line, flags=re.I).strip()
    return line.strip().strip('"').strip()


def _prompt(question: str, lines: list[str]) -> str:
    return build_prompt(question, lines)


def _select(kg: KG, loc: dict, scores: torch.Tensor, k: int, char_budget: int | None):
    order = torch.argsort(scores, descending=True).tolist()
    lines = []
    endpoints = []
    seen = set()
    used = 0
    n_keep = 0
    triples = loc["global_triples"]
    for idx in order[:k]:
        h, r, t = triples[idx]
        hn, rn, tn = kg.names[h], kg.rel_names[r], kg.names[t]
        line = f"({hn},{rn},{tn})"
        extra = len(line) + (1 if lines else 0)
        if char_budget is not None and lines and used + extra > char_budget:
            break
        lines.append(line)
        used += extra
        n_keep += 1
        for name in (hn, tn):
            if name not in seen:
                seen.add(name)
                endpoints.append(name)
    return lines, endpoints, n_keep


def _llm(prompt: str, question_t0: float | None = None) -> tuple[str, dict, float]:
    import httpx

    from retry_http import post_with_retry

    url = os.environ["LLM_BASE_URL"].rstrip("/") + "/chat/completions"
    t0 = time.time()
    payload = {
        "model": "generator",
        "max_tokens": 128,
        "temperature": 0,
        "frequency_penalty": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": prompt}],
    }

    def send():
        # Each attempt is its own POST through the metering gateway.
        return httpx.post(url, timeout=65, json=payload)

    response = post_with_retry(send, question_t0=question_t0)
    elapsed = time.time() - t0
    data = response.json()
    message = data["choices"][0]["message"]
    content = message.get("content") or ""
    if not content.strip():
        reasoning = message.get("reasoning_content") or ""
        if reasoning.strip():
            content = reasoning.strip().splitlines()[-1]
    usage = data.get("usage") or {}
    return _clean(content), usage, elapsed


def _trace(row: dict) -> None:
    path = os.path.join(os.environ["SCRATCH_DIR"], "trace.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def load() -> None:
    task = os.environ["SGRAG_TASK"]
    if task not in ("webqsp", "cwq"):
        raise SystemExit(task)
    art = os.environ["SGRAG_ARTIFACTS"]
    meta = json.load(open(os.path.join(art, "meta.json")))
    expect = os.environ.get("SGRAG_SCHEME", "a-train500")
    n_train = int(os.environ.get("SGRAG_N_TRAIN", "500"))
    if meta.get("scheme") != expect or meta.get("n_train") != n_train or meta.get("task") != task:
        raise SystemExit(f"refuse artifacts {meta}")
    ckpt = torch.load(os.path.join(art, "ckpt.pt"), map_location="cpu", weights_only=False)
    if ckpt.get("scheme") != expect or ckpt.get("task") != task:
        raise SystemExit("refuse checkpoint")
    triples = os.path.join(os.environ["CORPUS_DIR"], "triples.tsv")
    print("load graph", triples, flush=True)
    kg = KG(triples)
    text_names = [kg.names[i] for i in kg.text_ids]
    rel_names = sorted(kg.rel_names)
    if sha_names(text_names) != meta["text_sha256"] or sha_names(rel_names) != meta["rel_sha256"]:
        raise SystemExit("embedding rows do not match this corpus")
    device = "cpu"
    if torch.cuda.is_available():
        free, _total = torch.cuda.mem_get_info()
        if free > 4 * 1024**3:
            device = "cuda"
    print("encoder device", device, flush=True)
    tok, enc, n_params = load_encoder(os.environ["SGRAG_ENCODER"], device)
    if n_params > 1_000_000_000:
        raise SystemExit(f"encoder params {n_params}")
    model = Retriever(int(meta["emb_dim"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    n_ret = sum(p.numel() for p in model.parameters())
    total = n_params + n_ret
    print(f"params encoder {n_params} retriever {n_ret} dde 0 total {total}", flush=True)
    if total > 1_000_000_000:
        raise SystemExit(f"params {total}")
    S.update(
        kg=kg,
        tok=tok,
        enc=enc,
        model=model,
        device=device,
        ent=np.load(os.path.join(art, "ent_emb.f16.npy")),
        rel=np.load(os.path.join(art, "rel_emb.f16.npy")),
        hub=int(ckpt["hub"]),
        hop=int(ckpt.get("hop", 2)),
        ready=True,
    )
    print("READY", flush=True)


@app.get("/health")
def health():
    if not S.get("ready"):
        return JSONResponse({"ok": False}, status_code=503)
    return {"ok": True}


@app.post("/answer")
def answer(q: Query):
    if not S.get("ready"):
        return JSONResponse({"error": "not ready"}, status_code=503)
    t0 = time.time()
    question_t0 = time.monotonic()
    meta = q.meta or {}
    topics = meta.get("topic_entities") or []
    kg = S["kg"]
    extracted = kg.extract(topics, S["hub"], hop=S["hop"])
    lines: list[str] = []
    endpoints: list[str] = []
    n_cand = 0
    n_keep = 0
    capped = False
    if extracted is not None:
        loc = localize(kg, extracted, answer_names=None)
        n_cand = int(loc["n_cand"])
        capped = bool(loc["capped"])
        if loc["h"].size:
            device = S["device"]
            ent = torch.from_numpy(S["ent"][loc["text_rows"]].astype(np.float32)).to(device)
            rel = torch.from_numpy(S["rel"][loc["rel_rows"]].astype(np.float32)).to(device)
            q_emb = embed_texts(S["tok"], S["enc"], [q.question], batch=1, max_length=256)[0]
            with torch.no_grad():
                scores = S["model"](
                    torch.from_numpy(loc["h"]).to(device),
                    torch.from_numpy(loc["r"]).to(device),
                    torch.from_numpy(loc["t"]).to(device),
                    torch.from_numpy(q_emb.astype(np.float32)).to(device),
                    ent,
                    loc["n_non_text"],
                    rel,
                    torch.from_numpy(loc["topic_oh"]).to(device),
                ).squeeze(-1)
            k = configured_k()
            # SGRAG_EXACT_K=1 sends exactly K triples. The 24000-char cut would hide K>100.
            char_budget = None if os.environ.get("SGRAG_EXACT_K") == "1" else CHAR_BUDGET
            lines, endpoints, n_keep = _select(kg, loc, scores, k, char_budget)
    global QWEN
    if QWEN is None:
        QWEN = QwenPieceTokenizer()
    n_before = len(lines)
    tokens_before_guard = QWEN.count_chat(
        [{"role": "user", "content": build_prompt(q.question, lines)}]
    )
    lines, dropped, budget_est = guard_triples(QWEN, q.question, lines)
    n_keep = len(lines)
    tokens_before_send = budget_est - OUTPUT_CAP
    GUARD["questions"] += 1
    GUARD["dropped_triples"] += dropped
    if dropped:
        GUARD["triggered"] += 1
    if os.environ.get("SGRAG_GUARD_EACH") == "1" or dropped or GUARD["questions"] % 50 == 0:
        print(
            f"SGRAG_GUARD q={GUARD['questions']} triggered={GUARD['triggered']} "
            f"dropped_triples={GUARD['dropped_triples']} "
            f"this_before={n_before} this_dropped={dropped} "
            f"tokens_before_guard={tokens_before_guard} "
            f"tokens_before_send={tokens_before_send} budget_est={budget_est}",
            flush=True,
        )
    prompt = _prompt(q.question, lines)
    try:
        text, usage, llm_seconds = _llm(prompt, question_t0=question_t0)
        error = None
    except Exception as exc:
        text, usage, llm_seconds = "", {}, 0.0
        error = f"{type(exc).__name__}: {exc}"
    row = {
        "qid": q.qid,
        "k": configured_k(),
        "n_cand": n_cand,
        "n_keep": n_keep,
        "n_keep_before_guard": n_before,
        "guard_dropped": dropped,
        "truncated": dropped > 0,
        "tokens_before_guard": tokens_before_guard,
        "tokens_before_send": tokens_before_send,
        "guard_budget_est": budget_est,
        "capped": capped,
        "endpoints": endpoints,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
        "llm_seconds": round(llm_seconds, 3),
        "seconds": round(time.time() - t0, 3),
        "error": error,
    }
    _trace(row)
    if error:
        return JSONResponse({"answer": "", "error": error}, status_code=502)
    return {"answer": text, "citations": []}


@app.post("/retrieve")
def retrieve(q: Query):
    if not S.get("ready"):
        return JSONResponse({"error": "not ready"}, status_code=503)
    t0 = time.time()
    question_t0 = time.monotonic()
    meta = q.meta or {}
    topics = meta.get("topic_entities") or []
    kg = S["kg"]
    extracted = kg.extract(topics, S["hub"], hop=S["hop"])
    lines: list[str] = []
    endpoints: list[str] = []
    n_cand = 0
    n_keep = 0
    capped = False
    if extracted is not None:
        loc = localize(kg, extracted, answer_names=None)
        n_cand = int(loc["n_cand"])
        capped = bool(loc["capped"])
        if loc["h"].size:
            device = S["device"]
            ent = torch.from_numpy(S["ent"][loc["text_rows"]].astype(np.float32)).to(device)
            rel = torch.from_numpy(S["rel"][loc["rel_rows"]].astype(np.float32)).to(device)
            q_emb = embed_texts(S["tok"], S["enc"], [q.question], batch=1, max_length=256)[0]
            with torch.no_grad():
                scores = S["model"](
                    torch.from_numpy(loc["h"]).to(device),
                    torch.from_numpy(loc["r"]).to(device),
                    torch.from_numpy(loc["t"]).to(device),
                    torch.from_numpy(q_emb.astype(np.float32)).to(device),
                    ent,
                    loc["n_non_text"],
                    rel,
                    torch.from_numpy(loc["topic_oh"]).to(device),
                ).squeeze(-1)
            k = configured_k()
            # SGRAG_EXACT_K=1 sends exactly K triples. The 24000-char cut would hide K>100.
            char_budget = None if os.environ.get("SGRAG_EXACT_K") == "1" else CHAR_BUDGET
            lines, endpoints, n_keep = _select(kg, loc, scores, k, char_budget)
    source_triples=[]
    if extracted is not None and loc["h"].size:
        source_triples=[loc["global_triples"][i] for i in torch.argsort(scores,descending=True).tolist()[:len(lines)]]
    global QWEN
    if QWEN is None:
        QWEN = QwenPieceTokenizer()
    n_before = len(lines)
    tokens_before_guard = QWEN.count_chat(
        [{"role": "user", "content": build_prompt(q.question, lines)}]
    )
    lines, dropped, budget_est = guard_triples(QWEN, q.question, lines)
    n_keep = len(lines)
    tokens_before_send = budget_est - OUTPUT_CAP
    GUARD["questions"] += 1
    GUARD["dropped_triples"] += dropped
    if dropped:
        GUARD["triggered"] += 1
    if os.environ.get("SGRAG_GUARD_EACH") == "1" or dropped or GUARD["questions"] % 50 == 0:
        print(
            f"SGRAG_GUARD q={GUARD['questions']} triggered={GUARD['triggered']} "
            f"dropped_triples={GUARD['dropped_triples']} "
            f"this_before={n_before} this_dropped={dropped} "
            f"tokens_before_guard={tokens_before_guard} "
            f"tokens_before_send={tokens_before_send} budget_est={budget_est}",
            flush=True,
        )
    from fe_corpus import corpus_evidence, check_limits
    evidence=corpus_evidence();passages=[]
    # platform cap (256 passages) < configured K.
    # Preserve the method-selected, post-guard order; no re-ranking or K change.
    for h,r,t in source_triples[:min(len(lines), 256)]:
        hn,rn,tn=kg.names[h],kg.rel_names[r],kg.names[t]
        passages.append(evidence.passage(f"{hn} | {rn} | {tn}",title=hn))
    return {'passages':check_limits(passages)}


def main():
    for key in ("PORT", "LLM_BASE_URL", "CORPUS_DIR", "SCRATCH_DIR", "SGRAG_TASK", "SGRAG_ARTIFACTS", "SGRAG_ENCODER"):
        if not os.environ.get(key):
            raise SystemExit(f"missing {key}")
    os.makedirs(os.environ["SCRATCH_DIR"], exist_ok=True)
    load()
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ["PORT"]), log_level="info")


if __name__ == "__main__":
    main()
