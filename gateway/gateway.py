"""Metering gateway in front of vLLM (OpenAI-compatible).

build mode: unlimited calls, every call logged (gateway_build.jsonl).
eval  mode: calls are only accepted while the evaluator has a question open
            (POST /admin/open, /admin/close with X-Admin-Token). The (calls+1)-th call
            of a question gets 429; calls with no open question get 403. /admin/close
            returns the per-question settlement (calls, usage tokens, seconds, outputs).

Run: python gateway.py --mode eval --port 18300 --upstream http://127.0.0.1:18100 --log x.jsonl [--uds path]
Secrets: VLLM_API_KEY (upstream key) from env; the eval admin token from --admin-token-file (read, then
deleted) or GATEWAY_ADMIN_TOKEN.
"""
import argparse
import asyncio
import json
import os
import time

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


def extract_outputs(resp):
    outs = []
    for c in resp.get("choices", []) or []:
        if "message" in c and c["message"]:
            m = c["message"]
            outs.append(m.get("content") or "")
            for k in ("reasoning", "reasoning_content"):  # field name differs across vLLM versions
                if m.get(k):
                    outs.append(m[k])
        elif "text" in c:
            outs.append(c.get("text") or "")
    return outs


def build_app(mode, upstream, upstream_key, admin_token, log_path, budget):
    app = FastAPI()
    state = {"open": None}  # open question: {qid, t0, calls, rejected, prompt, completion, outputs}
    lock = asyncio.Lock()
    client = httpx.AsyncClient(base_url=upstream, timeout=httpx.Timeout(300.0))
    logf = open(log_path, "a", buffering=1)

    def log(rec):
        rec["ts"] = time.time()
        logf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    auth = {"Authorization": f"Bearer {upstream_key}"} if upstream_key else {}  # "Bearer " alone is an illegal header

    def admin_ok(request):
        return admin_token and request.headers.get("x-admin-token") == admin_token

    @app.get("/health")
    async def health():
        return {"ok": True, "mode": mode}

    @app.get("/v1/models")
    async def models():
        r = await client.get("/v1/models", headers=auth)
        return JSONResponse(r.json(), status_code=r.status_code)

    async def forward(request: Request, path: str):
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        if body.get("stream"):
            return JSONResponse({"error": "stream=true is not supported by this gateway"}, status_code=400)
        qid = None
        call_idx = None
        if mode == "eval":
            async with lock:
                q = state["open"]
                if q is None:
                    log({"event": "rejected_closed", "path": path, "request": body})
                    return JSONResponse({"error": "no question is open; generator calls are only allowed while answering"},
                                        status_code=403)
                if q["calls"] >= budget["calls"]:
                    q["rejected"] += 1
                    log({"event": "rejected_budget", "qid": q["qid"], "path": path})
                    return JSONResponse({"error": f"per-question call budget ({budget['calls']}) exhausted"},
                                        status_code=429)
                q["calls"] += 1
                qid, call_idx = q["qid"], q["calls"]
        t0 = time.time()
        r = await client.post(path, json=body, headers=auth)
        latency = time.time() - t0
        try:
            resp = r.json()
        except Exception:
            resp = {"raw": r.text}
        usage = resp.get("usage") or {}
        outs = extract_outputs(resp)
        if mode == "eval":
            async with lock:
                q = state["open"]
                if q is not None and q["qid"] == qid:
                    q["prompt"] += usage.get("prompt_tokens", 0) or 0
                    q["completion"] += usage.get("completion_tokens", 0) or 0
                    q["outputs"].extend(outs)
        log({"event": "call", "mode": mode, "qid": qid, "call_idx": call_idx, "path": path,
             "status": r.status_code, "latency": latency, "usage": usage, "request": body, "outputs": outs})
        return JSONResponse(resp, status_code=r.status_code)

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        return await forward(request, "/v1/chat/completions")

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await forward(request, "/v1/completions")

    @app.post("/admin/open")
    async def admin_open(request: Request):
        if mode != "eval" or not admin_ok(request):
            return JSONResponse({"error": "forbidden"}, status_code=403)
        qid = (await request.json())["qid"]
        async with lock:
            if state["open"] is not None:
                return JSONResponse({"error": f"question {state['open']['qid']} still open"}, status_code=409)
            state["open"] = {"qid": qid, "t0": time.time(), "calls": 0, "rejected": 0,
                             "prompt": 0, "completion": 0, "outputs": []}
        log({"event": "open", "qid": qid})
        return {"ok": True}

    @app.post("/admin/close")
    async def admin_close(request: Request):
        if mode != "eval" or not admin_ok(request):
            return JSONResponse({"error": "forbidden"}, status_code=403)
        qid = (await request.json())["qid"]
        async with lock:
            q = state["open"]
            if q is None or q["qid"] != qid:
                return JSONResponse({"error": "question not open"}, status_code=409)
            state["open"] = None
        seconds = time.time() - q["t0"]
        total = q["prompt"] + q["completion"]
        stats = {"qid": qid, "calls": q["calls"], "rejected_calls": q["rejected"],
                 "prompt_tokens": q["prompt"], "completion_tokens": q["completion"], "total_tokens": total,
                 "seconds": seconds, "outputs": q["outputs"],
                 "over_tokens": total > budget["tokens"], "over_seconds": seconds > budget["seconds"]}
        log({"event": "close", **stats})
        return stats

    @app.on_event("shutdown")
    async def _shutdown():
        await client.aclose()
        logf.close()

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["build", "eval"], required=True)
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--uds", help="listen on this unix socket instead of host:port")
    ap.add_argument("--admin-token-file")
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--calls", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=32000)
    ap.add_argument("--seconds", type=float, default=60)
    a = ap.parse_args()
    token = os.environ.pop("GATEWAY_ADMIN_TOKEN", "")
    if a.admin_token_file:
        token = open(a.admin_token_file).read().strip()
        os.remove(a.admin_token_file)
    app = build_app(a.mode, a.upstream, os.environ.get("VLLM_API_KEY", ""), token, a.log,
                    {"calls": a.calls, "tokens": a.tokens, "seconds": a.seconds})
    if a.uds:
        if os.path.exists(a.uds):
            os.remove(a.uds)
        uvicorn.run(app, uds=a.uds, log_level="warning")
    else:
        uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
