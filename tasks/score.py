"""Score predictions for any task.

  python -m tasks.score --task musique --pred predictions.jsonl --gold gold.jsonl [--out scores.json]

predictions.jsonl lines come from evaluator.py ({qid, answer, error, over_budget, ...}). A question
scores 0 on every metric if it is missing, errored, timed out or went over the per-question budget.
"""
import argparse
import json

from tasks import registry
from tasks.common import read_jsonl


def score(task, preds, golds):
    mod, inf = registry.load(task), registry.info(task)
    metrics = inf["metrics"]
    per_q = []
    for g in golds:
        p = preds.get(g["qid"])
        valid = p is not None and not p.get("error") and not p.get("over_budget")
        vals = mod.score_one(p.get("answer") or "", g) if valid else {m: 0.0 for m in metrics}
        per_q.append({"qid": g["qid"], "valid": valid, **vals,
                      "error": (p or {}).get("error", "missing" if p is None else None),
                      "over_budget": bool((p or {}).get("over_budget"))})
    n = len(per_q)
    res = {"task": task, "n": n, "primary_metric": inf["primary_metric"],
           **{m: sum(x[m] for x in per_q) / n for m in metrics},
           "n_invalid": sum(not x["valid"] for x in per_q), "per_question": per_q}
    res["primary"] = res[inf["primary_metric"]]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    res = score(a.task, {p["qid"]: p for p in read_jsonl(a.pred)}, read_jsonl(a.gold))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_question"}))


if __name__ == "__main__":
    main()
