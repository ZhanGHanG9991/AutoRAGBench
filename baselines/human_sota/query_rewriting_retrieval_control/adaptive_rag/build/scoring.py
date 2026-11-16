"""Correctness for silver labels, using the benchmark's own score_one."""
import sys

from guard import bench_root


def score_mod(task):
    root = bench_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    from tasks import registry
    return registry.load(task)


def is_correct(task, answer, gold):
    scores = score_mod(task).score_one(answer or "", gold)
    if "em" in scores:
        return float(scores["em"]) == 1.0, scores
    if "acc" in scores:
        return float(scores["acc"]) == 1.0, scores
    return float(scores.get("f1") or 0) >= 0.999, scores
