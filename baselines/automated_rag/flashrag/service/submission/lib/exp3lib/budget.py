"""Per-question cap on retrieval-time generator calls.

The platform gateway counts calls per open question id. The fixed-evidence
reader opens a different question id, so its answer call is not part of this
budget. This meter only wraps text generated while retrieving.
"""

from __future__ import annotations

import time


class BudgetError(RuntimeError):
    def __init__(self, kind: str):
        super().__init__(kind)
        self.kind = kind


class CallBudget:
    def __init__(self, max_calls: int = 4, max_tokens: int = 32000, max_seconds: float = 60.0):
        self.max_calls = max_calls
        self.max_tokens = max_tokens
        self.max_seconds = max_seconds
        self.reset()

    def reset(self) -> None:
        self.calls = 0
        self.tokens = 0
        self.started = time.monotonic()

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def before_call(self) -> None:
        if self.calls >= self.max_calls:
            raise BudgetError("calls")
        if self.elapsed() > self.max_seconds:
            raise BudgetError("seconds")

    def after_call(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.calls += 1
        self.tokens += int(prompt_tokens) + int(completion_tokens)
        if self.tokens > self.max_tokens:
            raise BudgetError("tokens")
        if self.elapsed() > self.max_seconds:
            raise BudgetError("seconds")
