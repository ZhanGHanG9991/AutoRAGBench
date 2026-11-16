"""Query and passage prefixes the model card asks for.

Applied only when the loaded SentenceTransformer has no prompt of its own,
so a model that already ships prompts is not prefixed twice.
"""

from __future__ import annotations

E5_QUERY = "query: "
E5_PASSAGE = "passage: "
BGE_QUERY = "Represent this sentence for searching relevant passages: "


def prefix_kind(model_name: str):
    name = model_name.lower()
    if "e5" in name:
        return "e5"
    if "bge" in name and "reranker" not in name and "m3" not in name:
        return "bge"
    return "none"


def apply_prefix(text: str, model_name: str, role: str, model_has_prompts: bool = False) -> str:
    if model_has_prompts:
        return text
    kind = prefix_kind(model_name)
    if kind == "e5":
        head = E5_QUERY if role == "query" else E5_PASSAGE
        return head + text
    if kind == "bge" and role == "query":
        return BGE_QUERY + text
    return text
