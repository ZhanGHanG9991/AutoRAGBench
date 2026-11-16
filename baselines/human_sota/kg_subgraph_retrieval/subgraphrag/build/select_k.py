"""Pick K from a holdout summary. Does not read dev.

Rule, fixed before any dev run: among K that were not eliminated, take the
highest holdout Hits@1. Scores within 0.01 of the best are ties; take the
smaller K. Tasks are chosen separately.
"""
from __future__ import annotations

import argparse
import json


def choose(rows: list[dict]) -> int:
    alive = [r for r in rows if not r.get("skipped") and not r.get("eliminated")]
    if not alive:
        raise SystemExit("no legal K")
    best = max(r["hits1"] for r in alive)
    tied = [r for r in alive if best - r["hits1"] <= 0.01]
    return min(r["k"] for r in tied)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--webqsp", required=True)
    ap.add_argument("--cwq", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-n", type=int, default=50)
    ap.add_argument("--scheme", default="a-train500")
    args = ap.parse_args()
    chosen = {}
    detail = {}
    for task, path in (("webqsp", args.webqsp), ("cwq", args.cwq)):
        summary = json.load(open(path))
        if summary.get("n") != args.expect_n or summary.get("scheme") != args.scheme:
            raise SystemExit(f"refuse summary {path}")
        k = choose(summary["ks"])
        chosen[task] = k
        detail[task] = {
            "k": k,
            "hits_step": summary.get("hits_step"),
            "rows": [
                {
                    "k": r.get("k"),
                    "hits1": r.get("hits1"),
                    "eliminated": r.get("eliminated"),
                    "skipped": r.get("skipped"),
                }
                for r in summary["ks"]
            ],
        }
        print(f"SELECTED {task} K={k}", flush=True)
    json.dump({"webqsp": chosen["webqsp"], "cwq": chosen["cwq"], "detail": detail}, open(args.out, "w"), indent=2)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
