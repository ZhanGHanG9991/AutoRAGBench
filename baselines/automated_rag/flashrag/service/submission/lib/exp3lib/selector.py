"""FlashRAG candidate picker. Tune scores only. Dev and test are refused.

Higher tune score wins. A tie goes to the candidate with fewer generator
calls, then to the earlier entry in the registered list.
"""

from __future__ import annotations


def select(candidates, tune_scores, *, split="tune"):
    if split != "tune":
        raise ValueError("selector only accepts the train tune split")
    if not candidates:
        raise ValueError("no candidates")
    missing = [c["id"] for c in candidates if c["id"] not in tune_scores]
    if missing:
        raise ValueError("tune score missing for " + ",".join(missing))
    order = {c["id"]: i for i, c in enumerate(candidates)}

    def key(candidate):
        score = tune_scores[candidate["id"]]
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("tune score must be a number")
        return (float(score), -int(candidate["generator_calls"]), -order[candidate["id"]])

    return max(candidates, key=key)
