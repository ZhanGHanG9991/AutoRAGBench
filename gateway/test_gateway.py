"""Offline tests for gateway + evaluator (no GPU): a fake vLLM upstream, the real gateway
in eval mode, and a stdlib toy service that crashes on demand.

Run on the login node:  envs/vllm/bin/python gateway/test_gateway.py
"""
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gateway"))
import gateway  # noqa: E402


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def serve(app, port):
    cfg = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    srv = uvicorn.Server(cfg)
    threading.Thread(target=srv.run, daemon=True).start()
    for _ in range(100):
        try:
            httpx.get(f"http://127.0.0.1:{port}/health", timeout=1)
            return srv
        except httpx.HTTPError:
            time.sleep(0.1)
    raise RuntimeError("server did not start")


def fake_upstream():
    app = FastAPI()

    @app.get("/health")
    def h():
        return {"ok": True}

    @app.post("/v1/chat/completions")
    def chat(body: dict):
        q = body["messages"][-1]["content"]
        return {"choices": [{"message": {"role": "assistant", "content": f"ANSWER({q})"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}}
    return app


TOY_SERVICE = r'''
import json, os, sys, urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
BASE = os.environ["LLM_BASE_URL"]
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
    def do_POST(self):
        q = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["question"]
        if "CRASH" in q:
            os._exit(1)
        n = 5 if "GREEDY" in q else 1
        out = None
        for _ in range(n):
            req = urllib.request.Request(BASE + "/chat/completions", method="POST",
                data=json.dumps({"model": "generator", "messages": [{"role": "user", "content": q}]}).encode(),
                headers={"Content-Type": "application/json"})
            try:
                out = json.loads(urllib.request.urlopen(req).read())["choices"][0]["message"]["content"]
            except Exception as e:
                out = out or "budget"
        body = json.dumps({"answer": out}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(body)
HTTPServer(("127.0.0.1", int(os.environ["PORT"])), H).serve_forever()
'''


def main():
    tmp = Path(tempfile.mkdtemp(prefix="gwtest-"))
    up_port, gw_port, svc_port = free_port(), free_port(), free_port()
    serve(fake_upstream(), up_port)
    token = "test-admin"
    app = gateway.build_app("eval", f"http://127.0.0.1:{up_port}", "k", token, str(tmp / "gw.jsonl"),
                            {"calls": 4, "tokens": 32000, "seconds": 60})
    serve(app, gw_port)
    gw = f"http://127.0.0.1:{gw_port}"
    adm = {"X-Admin-Token": token}
    msg = {"model": "generator", "messages": [{"role": "user", "content": "hi"}]}

    # 1. no open question -> 403
    assert httpx.post(f"{gw}/v1/chat/completions", json=msg).status_code == 403
    # 2. wrong admin token -> 403
    assert httpx.post(f"{gw}/admin/open", json={"qid": "x"}, headers={"X-Admin-Token": "bad"}).status_code == 403
    # 3. 4 calls OK, 5th -> 429; close settles usage
    assert httpx.post(f"{gw}/admin/open", json={"qid": "q1"}, headers=adm).status_code == 200
    codes = [httpx.post(f"{gw}/v1/chat/completions", json=msg).status_code for _ in range(5)]
    assert codes == [200, 200, 200, 200, 429], codes
    st = httpx.post(f"{gw}/admin/close", json={"qid": "q1"}, headers=adm).json()
    assert st["calls"] == 4 and st["rejected_calls"] == 1 and st["total_tokens"] == 440, st
    # 4. after close -> 403 again; stream -> 400
    assert httpx.post(f"{gw}/v1/chat/completions", json=msg).status_code == 403
    httpx.post(f"{gw}/admin/open", json={"qid": "q2"}, headers=adm)
    assert httpx.post(f"{gw}/v1/chat/completions", json={**msg, "stream": True}).status_code == 400
    httpx.post(f"{gw}/admin/close", json={"qid": "q2"}, headers=adm)
    print("gateway: 403-closed, bad-token, 429-on-5th, settlement, stream-400 OK")

    # 5. evaluator with crash + restart
    ws = tmp / "ws"
    (ws / "submission").mkdir(parents=True)
    (ws / "submission" / "svc.py").write_text(TOY_SERVICE)
    (ws / "submission" / "start.sh").write_text("exec /usr/bin/python3 submission/svc.py\n")
    qs = [{"qid": "a", "question": "first"}, {"qid": "b", "question": "please CRASH"},
          {"qid": "c", "question": "GREEDY one"}, {"qid": "d", "question": "last"}]
    (tmp / "q.jsonl").write_text("".join(json.dumps(q) + "\n" for q in qs))
    env = {**os.environ, "GATEWAY_ADMIN_TOKEN": token}
    r = subprocess.run([sys.executable, str(ROOT / "evaluator.py"), "--workspace", str(ws),
                        "--questions", str(tmp / "q.jsonl"), "--out", str(tmp / "pred.jsonl"),
                        "--gateway", gw, "--llm-base-url", f"{gw}/v1", "--corpus-dir", str(tmp),
                        "--scratch-dir", str(tmp / "scratch"), "--service-log", str(tmp / "svc.log"),
                        "--port", str(svc_port), "--startup-timeout", "30", "--answer-timeout", "10"],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    preds = {p["qid"]: p for p in map(json.loads, open(tmp / "pred.jsonl"))}
    summ = json.load(open(tmp / "pred.jsonl.summary.json"))
    assert preds["a"]["answer"] == "ANSWER(first)", preds["a"]
    assert preds["b"]["error"] and summ["restarts"] == 1, (preds["b"], summ)
    assert preds["c"]["gateway"]["calls"] == 4 and preds["c"]["gateway"]["rejected_calls"] == 1, preds["c"]
    assert preds["d"]["answer"] == "ANSWER(last)", preds["d"]
    print("evaluator: answer, crash->restart, budget-in-service OK", json.dumps(summ))
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
