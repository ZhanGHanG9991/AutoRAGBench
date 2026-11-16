"""Start the shared /health and /retrieve server.

"""

from __future__ import annotations

import os

from exp3lib.http_retrieve import make_server


def load_corpus(path):
    from exp3lib.jsonl_corpus import load_jsonl_corpus

    rows = load_jsonl_corpus(path)
    return {row["doc_id"]: {"title": row["title"], "text": row["text"]} for row in rows}


def serve(retrieve) -> None:
    host = os.environ.get("EXP3_BIND", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    server = make_server(retrieve, host=host, port=port)
    server.serve_forever()
