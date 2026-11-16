"""Qwen3.5-4B tokenizer for the query adapter.

Encode and decode go through the local tokenizer.json. Chat-template counts
use the same string Qwen's apply_chat_template emits when thinking is off.
The renderer was checked against transformers on system/user messages.
"""

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
from pathlib import Path

from tokenizers import Tokenizer


MODEL_DIR = _release_path("@MODEL_DIR@/Qwen3.5-4B")
BUDGET_LIMIT = 32000
OUTPUT_CAP = 512


class QwenPieceTokenizer:
    """TokenizerInterface for lightrag.utils.Tokenizer.

    The Rust tokenizer is safe to call from LightRAG's worker threads.
    __deepcopy__ returns self so LightRAG's asdict copy does not pickle it.
    """

    def __init__(self, model_dir: str = MODEL_DIR):
        path = Path(model_dir) / "tokenizer.json"
        if not path.is_file():
            raise SystemExit("QWEN_TOKENIZER_MISSING")
        self._tok = Tokenizer.from_file(str(path))

    def __deepcopy__(self, memo):
        return self

    def encode(self, content: str):
        return self._tok.encode(content).ids

    def decode(self, tokens):
        return self._tok.decode(list(tokens))

    def count_chat(self, messages) -> int:
        return len(self.encode(render_chat(messages)))


def render_chat(messages, enable_thinking: bool = False) -> str:
    """Match Qwen3.5 apply_chat_template with thinking off and no tools."""
    parts = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, str):
            content = "" if content is None else str(content)
        role = str(message.get("role") or "user")
        parts.append("<|im_start|>" + role + "\n" + content + "<|im_end|>\n")
    if enable_thinking:
        parts.append("<|im_start|>assistant\n")
    else:
        parts.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return "".join(parts)


def _chunk_span(text: str):
    header = text.find("Document Chunks (")
    if header < 0:
        return None
    fence = text.find("```json", header)
    if fence < 0:
        return None
    newline = text.find("\n", fence)
    if newline < 0:
        return None
    body_start = newline + 1
    close = text.find("\n```", body_start)
    if close < 0:
        return None
    return body_start, close


def chunk_lines(text: str):
    span = _chunk_span(text)
    if span is None:
        return None
    body = text[span[0] : span[1]]
    if body == "":
        return []
    return body.split("\n")


def drop_last_chunk(text: str):
    """Remove the last Document Chunks JSON line and its reference line."""
    span = _chunk_span(text)
    if span is None:
        return text, False
    lines = chunk_lines(text)
    if not lines:
        return text, False
    last = lines.pop()
    try:
        ref_id = json.loads(last).get("reference_id")
    except json.JSONDecodeError:
        ref_id = None
    body = "\n".join(lines)
    text = text[: span[0]] + body + text[span[1] :]
    if ref_id is not None:
        section = text.find("Reference Document List (")
        needle = f"\n[{ref_id}] "
        idx = text.find(needle, section) if section >= 0 else -1
        if idx >= 0:
            end = text.find("\n", idx + 1)
            if end < 0:
                text = text[:idx]
            else:
                text = text[:idx] + text[end:]
    return text, True


def guard_messages(qwen: QwenPieceTokenizer, messages, kw_tokens: int):
    """Drop trailing chunks until keyword+answer chat tokens and two 512 caps fit."""
    index = None
    for i, message in enumerate(messages):
        content = message.get("content") or ""
        if "Document Chunks (" in content:
            index = i
            break
    content = messages[index]["content"] if index is not None else ""
    before_lines = chunk_lines(content) if index is not None else None
    chunks_before = 0 if before_lines is None else len(before_lines)
    dropped = 0
    while True:
        total = int(kw_tokens) + OUTPUT_CAP + qwen.count_chat(messages) + OUTPUT_CAP
        if total <= BUDGET_LIMIT or index is None:
            break
        updated, removed = drop_last_chunk(messages[index]["content"])
        if not removed:
            break
        messages[index]["content"] = updated
        dropped += 1
        if dropped > chunks_before:
            break
    after_lines = chunk_lines(messages[index]["content"]) if index is not None else None
    chunks_after = 0 if after_lines is None else len(after_lines)
    total = int(kw_tokens) + OUTPUT_CAP + qwen.count_chat(messages) + OUTPUT_CAP
    return chunks_before, chunks_after, dropped, total
