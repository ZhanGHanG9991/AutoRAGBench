"""One chat call to the platform generator.

The endpoint and the key come from LLM_BASE_URL and LLM_API_KEY. This module
does not print either value, and it does not read a dotenv file or any auth.json.

"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


def platform_complete(prompt: str, budget, *, max_tokens: int = 256) -> str:
    budget.before_call()
    text, prompt_tokens, completion_tokens = _post(prompt, max_tokens=max_tokens)
    budget.after_call(prompt_tokens, completion_tokens)
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
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError("platform generator request failed") from exc
    try:
        text = payload["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("platform generator response had no message") from exc
    usage = payload.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or max(1, len(prompt) // 4))
    completion_tokens = int(usage.get("completion_tokens") or max(1, len(text) // 4))
    return text, prompt_tokens, completion_tokens
