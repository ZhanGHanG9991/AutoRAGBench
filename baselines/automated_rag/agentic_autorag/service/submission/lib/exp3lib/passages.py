"""Turn retrieval hits into fixed-evidence passages.

Only contiguous slices of the corpus text are returned. The platform then
applies the 32_000 character and 12_000 Qwen-token cuts. This builder stops
at 256 passages and at the contract's 500_000 character ceiling, which is
the largest payload validate_passages will accept.
"""

from __future__ import annotations

MAX_PASSAGES = 256
MAX_EVIDENCE_CHARS = 500_000


class EvidenceError(ValueError):
    pass


def original_slice(source_text: str, start: int, end: int) -> str:
    if not (0 <= start < end <= len(source_text)):
        raise EvidenceError("invalid original-text offsets")
    return source_text[start:end]


def build_passages(hits, corpus, meta=None, *, max_passages=MAX_PASSAGES, max_chars=MAX_EVIDENCE_CHARS):
    """hits are dicts with doc_id, start, end, and text.

    text must equal corpus[doc_id]['text'][start:end]. Order is kept.
    A requested meta doc_id drops every other document.
    """
    only = None if not meta else meta.get("doc_id")
    if only is not None:
        only = str(only)
    passages = []
    used = 0
    for hit in hits:
        if len(passages) >= max_passages:
            break
        did = str(hit["doc_id"])
        if only is not None and did != only:
            continue
        source = corpus[did]
        start, end = int(hit["start"]), int(hit["end"])
        text = original_slice(source["text"], start, end)
        if text != hit["text"]:
            raise EvidenceError("chunk is not the original contiguous text")
        if not text.strip():
            continue
        title = source.get("title") or ""
        if not isinstance(title, str):
            raise EvidenceError("title must be a string")
        overhead = len(title) + len(text)
        if passages and used + overhead > max_chars:
            room = max_chars - used - len(title)
            if room <= 0 or not text[:room].strip():
                break
            text = text[:room]
            overhead = len(title) + len(text)
        elif not passages and overhead > max_chars:
            room = max_chars - len(title)
            if room <= 0 or not text[:room].strip():
                break
            text = text[:room]
            overhead = len(title) + len(text)
        passages.append({"doc_id": did, "title": title, "text": text})
        used += overhead
    return passages
