"""Read a corpus jsonl without copying it into one file per document.

Each line is one document: doc_id, text, and an optional title. The text is
what gets chunked. The title is returned beside it and is not prepended, so
a chunk stays a contiguous slice of the original text.
"""

from __future__ import annotations

import json
from pathlib import Path


def is_jsonl_corpus(path) -> bool:
    return Path(path).is_file() and Path(path).suffix == ".jsonl"


def load_jsonl_corpus(path):
    path = Path(path)
    rows = []
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            doc_id = row.get("doc_id")
            text = row.get("text")
            if not isinstance(doc_id, str) or not doc_id:
                raise ValueError(f"line {line_no} missing doc_id")
            if not isinstance(text, str):
                raise ValueError(f"line {line_no} missing text")
            if doc_id in seen:
                raise ValueError(f"duplicate doc_id on line {line_no}")
            seen.add(doc_id)
            title = row.get("title") or ""
            if not isinstance(title, str):
                raise ValueError(f"line {line_no} title is not a string")
            rows.append({"doc_id": doc_id, "title": title, "text": text})
    if not rows:
        raise ValueError(f"no documents in {path}")
    return rows
