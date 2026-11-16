"""Design-space exploration on dev (what the LEGO-GraphRAG paper does across instances), under the AutoRAGBench
per-question budget (<= 4 generator calls; checked here, enforced again by the gateway in the official eval).

For each instance in lego_text.INSTANCES the retrieval runs once per question, then both generation contexts
("paths", "paths+passages") are generated; each variant is charged the shared calls + its own generation call.
Only train/dev are used. Writes <out>/{<inst>__<ctx>.jsonl, summary.json}.

  python devgrid.py --task musique --index <index_dir> --llm http://127.0.0.1:PORT/v1 --key KEY --out <dir>
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
import traceback

R = os.environ["AUTORAGBENCH_ROOT"]
sys.path.insert(0, f"{R}/humansota/LEGO-GraphRAG/rageng")
sys.path.insert(0, R)  # tasks.score
from lego_text import INSTANCES, BudgetExceeded, GenLLM, LegoText  # noqa: E402
from tasks.common import read_jsonl  # noqa: E402
from tasks.score import score  # noqa: E402

CTXS = ("paths", "paths+passages")


def run_one(P, llm, inst, row):
    llm.reset()
    t = time.time()
    res = {}
    try:
        paths, info = P.retrieve(row["question"], row.get("meta"), INSTANCES[inst])
        shared = llm.calls
        for ctx in CTXS:
            llm.local.calls = shared
            ans, _ = P.generate(row["question"], paths, ctx)
            res[ctx] = {"qid": row["qid"], "answer": ans, "calls": llm.calls, "error": None}
        info["paths"] = [str(p) for p in paths[:5]]
    except BudgetExceeded as ex:
        info = {"budget": str(ex)}
        res = {c: {"qid": row["qid"], "answer": "", "over_budget": True, "error": None} for c in CTXS}
    except Exception as ex:  # noqa: BLE001 -- recorded, scores 0 like an evaluator error
        info = {"exception": traceback.format_exc()[-800:]}
        res = {c: {"qid": row["qid"], "answer": "", "error": f"{type(ex).__name__}: {ex}"[:300]} for c in CTXS}
    for c in CTXS:
        res[c].update(seconds=round(time.time() - t, 2), info=info)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--llm", required=True)
    ap.add_argument("--key", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--questions", default="", help="explicit formal-v2 public dev JSONL")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--instances", default=",".join(INSTANCES))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    info = json.load(open(f"{R}/tasks/{a.task}/info.json"))
    rows = read_jsonl(a.questions or f"{R}/data/{a.task}/public/{a.split}.jsonl")
    rows = rows[:a.limit] if a.limit else rows
    llm = GenLLM(a.llm, a.key)
    t0 = time.time()
    P = LegoText(a.index, info, llm)
    print(f"loaded index in {time.time() - t0:.0f}s: {P.G.vcount()} nodes {P.G.ecount()} edges", flush=True)
    summary_path = os.path.join(a.out, "summary.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
    for inst in a.instances.split(","):
        t = time.time()
        with cf.ThreadPoolExecutor(a.workers) as ex:
            results = list(ex.map(lambda r: run_one(P, llm, inst, r), rows))
        for ctx in CTXS:
            preds = [r[ctx] for r in results]
            name = f"{inst}__{ctx}"
            with open(os.path.join(a.out, name.replace("+", "_") + ".jsonl"), "w") as f:
                for p in preds:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
            sc = score(a.task, {p["qid"]: p for p in preds}, rows)
            summary[name] = {"primary": round(sc["primary"], 4), "n_invalid": sc["n_invalid"],
                             "mean_calls": round(sum(p.get("calls", 0) for p in preds) / len(preds), 2),
                             "max_calls": max(p.get("calls", 0) for p in preds),
                             "mean_s": round(sum(p["seconds"] for p in preds) / len(preds), 2),
                             "n": len(preds), "split": a.split}
            print(f"{name:42} {a.split} {sc['primary']:.4f} invalid={sc['n_invalid']} "
                  f"calls={summary[name]['mean_calls']} wall={time.time() - t:.0f}s", flush=True)
        json.dump(summary, open(summary_path, "w"), indent=1)
    best = max(summary, key=lambda k: summary[k]["primary"])
    print("BEST", best, summary[best], flush=True)


if __name__ == "__main__":
    main()
