"""Frozen FlashRAG control flow for the registered candidates.

The tie-break uses the generator_calls field stored on the candidate. That
number is the maximum this loop is allowed to spend. A run that stops early
does not change the registered number.

"""

from __future__ import annotations

ITER_PROMPT = (
    "Write one search query that can find a missing piece of the claim. "
    "Return only the query.\nClaim: {question}\nPassages:\n{passages}\nQuery:"
)
IRCOT_PROMPT = (
    "Write the next search query for the claim, or the single word STOP when "
    "another search is not needed. Return only that query or STOP.\n"
    "Claim: {question}\nPassages:\n{passages}\nNext:"
)
ADAPTIVE_PROMPT = (
    "Write one follow-up search query for the claim. Return only the query.\n"
    "Claim: {question}\nPassages:\n{passages}\nQuery:"
)


def _dedup_extend(ordered, seen, hits):
    for hit in hits:
        key = (str(hit["doc_id"]), int(hit["start"]), int(hit["end"]))
        if key in seen:
            continue
        seen.add(key)
        ordered.append(hit)


def _passages(hits):
    return "\n".join(hit["text"] for hit in hits)


def run_candidate(candidate, question, search, complete, rerank=None, meta=None):
    """search(query, k, doc_id) -> hits. complete(prompt) -> str. rerank(question, hits) -> hits.

    doc_id comes from meta and is passed through so a qasper paper can be
    filtered by the retriever. The HTTP layer filters again.
    """
    loop = candidate["loop"]
    top_k = int(candidate["top_k"])
    doc_id = None if not meta else meta.get("doc_id")
    if doc_id is not None:
        doc_id = str(doc_id)

    if loop == "standard":
        fetch = int(candidate["fetch_k"]) if candidate.get("reranker") else top_k
        hits = list(search(question, fetch, doc_id))
        if candidate.get("reranker"):
            if rerank is None:
                raise ValueError("reranker candidate requires a rerank function")
            hits = list(rerank(question, hits))[: int(candidate["top_n"])]
        return hits

    if loop == "iter_retgen":
        ordered = []
        seen = set()
        query = question
        for round_index in range(3):
            _dedup_extend(ordered, seen, search(query, top_k, doc_id))
            if round_index == 2:
                break
            query = complete(ITER_PROMPT.format(question=question, passages=_passages(ordered))).strip() or question
        return ordered

    if loop == "ircot":
        ordered = []
        seen = set()
        for _step in range(int(candidate["generator_calls"])):
            raw = complete(IRCOT_PROMPT.format(question=question, passages=_passages(ordered))).strip()
            if raw.upper() == "STOP":
                break
            query = raw or question
            _dedup_extend(ordered, seen, search(query, top_k, doc_id))
        return ordered

    if loop == "adaptive_single":
        return list(search(question, top_k, doc_id))

    if loop == "adaptive_multi":
        ordered = []
        seen = set()
        _dedup_extend(ordered, seen, search(question, top_k, doc_id))
        query = complete(ADAPTIVE_PROMPT.format(question=question, passages=_passages(ordered))).strip() or question
        _dedup_extend(ordered, seen, search(query, top_k, doc_id))
        complete(ADAPTIVE_PROMPT.format(question=question, passages=_passages(ordered)))
        return ordered

    raise ValueError("unknown candidate loop")
