"""Tiny /health and /retrieve server used by every framework wrapper."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from exp3lib.budget import BudgetError
from exp3lib.passages import EvidenceError, build_passages


def make_server(retrieve, host="127.0.0.1", port=0):
    """retrieve(question, meta) -> list of hit dicts. corpus is closed over by the caller."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            return

        def _send(self, status, payload):
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.split("?", 1)[0] != "/health":
                self._send(404, {"error": "not_found"})
                return
            self._send(200, {"ready": True})

        def do_POST(self):
            if self.path.split("?", 1)[0] != "/retrieve":
                self._send(404, {"error": "not_found"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 1_000_000:
                    raise ValueError("invalid request size")
                request = json.loads(self.rfile.read(size))
                question = request["question"]
                meta = request.get("meta") or {}
                if not isinstance(question, str) or not isinstance(meta, dict):
                    raise ValueError("invalid request")
                hits, corpus = retrieve(question, meta)
                passages = build_passages(hits, corpus, meta)
                self._send(200, {"passages": passages})
            except BudgetError:
                self._send(429, {"error": "over_budget"})
            except (EvidenceError, ValueError, KeyError, TypeError):
                self._send(400, {"error": "invalid_request_or_evidence"})

    server = ThreadingHTTPServer((host, port), Handler)
    server.retrieve = retrieve
    return server
