"""Qwen3.5-4B tokenizer for SubgraphRAG send-time budget guard.

Same constants and chat-template counting as LightRAG rageng/qwen_tok.py.
Drops trailing triple lines until chat tokens plus one 512 output cap fit in 32k.
"""
from __future__ import annotations

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


from pathlib import Path

from tokenizers import Tokenizer


MODEL_DIR = _release_path("@MODEL_DIR@/Qwen3.5-4B")
BUDGET_LIMIT = 32000
OUTPUT_CAP = 512


class QwenPieceTokenizer:
    def __init__(self, model_dir: str = MODEL_DIR):
        path = Path(model_dir) / "tokenizer.json"
        if not path.is_file():
            raise SystemExit("QWEN_TOKENIZER_MISSING")
        self._tok = Tokenizer.from_file(str(path))

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


def build_prompt(question: str, lines: list[str]) -> str:
    head = (
        "Answer the question with exactly one entity name. "
        "Do not explain.\n\n"
        f"Question: {question}\n\n"
    )
    if not lines:
        return head + "Answer:"
    return head + "Triples:\n" + "\n".join(lines) + "\n\nAnswer:"


def guard_triples(qwen: QwenPieceTokenizer, question: str, lines: list[str]):
    """Drop trailing triples until chat tokens + OUTPUT_CAP fit in BUDGET_LIMIT.

    Returns (kept_lines, dropped, total_est).
    """
    kept = list(lines)
    before = len(kept)
    dropped = 0
    while True:
        messages = [{"role": "user", "content": build_prompt(question, kept)}]
        total = qwen.count_chat(messages) + OUTPUT_CAP
        if total <= BUDGET_LIMIT or not kept:
            break
        kept.pop()
        dropped += 1
        if dropped > before:
            break
    messages = [{"role": "user", "content": build_prompt(question, kept)}]
    total = qwen.count_chat(messages) + OUTPUT_CAP
    return kept, dropped, total
