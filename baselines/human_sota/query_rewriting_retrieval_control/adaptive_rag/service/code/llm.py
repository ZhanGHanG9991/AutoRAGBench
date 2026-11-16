"""One call to the fixed Qwen generator. Thinking stays off."""
import os
import re

import httpx

from retry_http import post_with_retry

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_question_t0 = None


def set_question_clock(t0):
    """Monotonic timestamp when the current question started. None outside one."""
    global _question_t0
    _question_t0 = t0


def _text(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        bits = []
        for part in content:
            if isinstance(part, str):
                bits.append(part)
            elif isinstance(part, dict):
                bits.append(part.get("text") or "")
        return "".join(bits)
    return str(content)


def chat(prompt, max_tokens=64, timeout=25):
    """Return (text, total_tokens). Text is the generator's visible answer."""
    base = os.environ["LLM_BASE_URL"].rstrip("/")
    headers = {}
    key = os.environ.get("VLLM_API_KEY") or os.environ.get("LLM_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    payload = {
        "model": "generator",
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": prompt}],
    }

    def send():
        # A new client per attempt. Each POST hits the metering gateway.
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                f"{base}/chat/completions", headers=headers, json=payload
            )
            response.read()
            return response

    response = post_with_retry(send, question_t0=_question_t0)
    body = response.json()
    choice = body["choices"][0]["message"]
    text = _THINK.sub("", _text(choice.get("content"))).strip()
    if not text:
        # Some Qwen builds still put the reply in reasoning_content. That text
        # is model output, so it is a legal answer source. Prefer content above.
        text = _THINK.sub("", _text(choice.get("reasoning_content"))).strip()
    usage = body.get("usage") or {}
    tokens = int(usage.get("total_tokens") or 0)
    if not tokens:
        tokens = int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0)
    if not tokens:
        tokens = max(1, (len(prompt) + len(text)) // 4)
    return text, tokens
