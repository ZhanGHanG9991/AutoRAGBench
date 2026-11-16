"""One chat call to the platform generator.

The endpoint and the key come from LLM_BASE_URL and LLM_API_KEY. This module
does not print either value, and it does not read a dotenv file or any auth.json.

The per-question token budget is enforced before
sending. The prompt is counted with the Qwen tokenizer at EXP3_QWEN_TOKENIZER (chat
template, thinking off, as the generator sees it). If tokens already used + prompt +
max_tokens would exceed the budget, BudgetError('tokens') is raised and nothing is sent.
An HTTP 400 whose body mentions the context length is also BudgetError('tokens').
In both cases the rejected call is not counted in the meter (after_call never runs).

"""

from __future__ import annotations

import json
import os
from pathlib import Path
import urllib.error
import urllib.request
from functools import lru_cache

from exp3lib.budget import BudgetError

_DEFAULT_TOKENIZER = str(Path(os.environ["EXP3_MODELS_DIR"]) / "Qwen3.5-4B")


@lru_cache(maxsize=1)
def _tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(os.environ.get("EXP3_QWEN_TOKENIZER", _DEFAULT_TOKENIZER),
                                         local_files_only=True)


def prompt_tokens(prompt: str) -> int:
    """Prompt tokens as the generator counts them: one user message, chat template, thinking off."""
    tok = _tokenizer()
    messages = [{"role": "user", "content": prompt}]
    try:
        ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    except Exception:  # noqa: BLE001 - tokenizer without a chat template
        ids = tok(prompt)["input_ids"]
    if isinstance(ids, dict) or hasattr(ids, "keys"):
        ids = ids["input_ids"]
    return len(ids)


def platform_complete(prompt: str, budget, *, max_tokens: int = 256) -> str:
    budget.before_call()
    used = int(getattr(budget, "tokens", 0))
    limit = getattr(budget, "max_tokens", None)
    if limit is not None and used + prompt_tokens(prompt) + int(max_tokens) > int(limit):
        # Rejected before sending: no request, and the meter is not charged.
        raise BudgetError("tokens")
    text, prompt_used, completion_tokens = _post(prompt, max_tokens=max_tokens)
    budget.after_call(prompt_used, completion_tokens)
    return text


def _post(prompt: str, *, max_tokens: int) -> tuple[str, int, int]:
    base = os.environ.get("LLM_BASE_URL", "").rstrip("/")
    if not base:
        raise RuntimeError("LLM_BASE_URL is not set")
    url = base if base.endswith("/chat/completions") else base + "/chat/completions"
    key = os.environ.get("LLM_API_KEY", "") or "EMPTY"
    body = {
        "model": os.environ.get("EXP3_GENERATOR_MODEL", "generator"),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            pass
        if exc.code == 400 and "context length" in detail.lower():
            raise BudgetError("tokens") from exc
        raise RuntimeError("platform generator request failed") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError("platform generator request failed") from exc
    try:
        text = payload["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("platform generator response had no message") from exc
    usage = payload.get("usage") or {}
    prompt_used = int(usage.get("prompt_tokens") or max(1, len(prompt) // 4))
    completion_tokens = int(usage.get("completion_tokens") or max(1, len(text) // 4))
    return text, prompt_used, completion_tokens
