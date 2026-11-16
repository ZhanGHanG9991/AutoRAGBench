"""K sweep for formal-v2 CWQ on the frozen holdout of 200. Does not read dev or test.

Selection rule is v1, unchanged. This file imports it:

- sweep_holdout.BASE_KS (100, 200, 400), EXTRA_K (800), run_k, rank_one
- select_k.choose: highest holdout Hits@1 among K that are not skipped and not
  eliminated; scores within 0.01 of the best tie; take the smaller K.

800 is run only when 400 is not eliminated. A K is eliminated when any question
errors, is not exactly 1 call, is missing token counts, has prompt or total
tokens above 32000, or has latency above 60s. Latency is the shared top-800
retrieve_s plus that K's generator time. Generator settings are one_call's.

The questions are holdout.jsonl only (formal-v2-split/cwq-sgrag-train2000),
with holdout_q.f16.npy in the same order. train_fit is not opened.
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


import argparse
import hashlib
import json
import os
import time

import httpx
import numpy as np
import torch

from encode import sha_names
from graph import KG
from model import Retriever
from select_k import choose
from sweep_holdout import BASE_KS, EXTRA_K, rank_one, run_k

ROOT = _release_path("@AUTORAGBENCH_ROOT@")
HOLD_SUFFIX = "/humansota/formal-v2-split/cwq-sgrag-train2000/holdout.jsonl"
TRIPLES_SUFFIX = "/data_formal-v2/cwq/public/corpus/triples.tsv"
CKPT_SUFFIX = "/humansota/SubgraphRAG/rageng/artifacts-formalv2/cwq-train1800/ckpt.pt"
EMB_SUFFIX = "/humansota/SubgraphRAG/rageng/artifacts-formalv2/cwq"
Q_SUFFIX = "/humansota/SubgraphRAG/rageng/artifacts-formalv2/cwq-train1800"
OUT_SUFFIX = "/humansota/SubgraphRAG/rageng/results-formalv2/cwq-holdout-k"
SCHEME = "formalv2-train1800"
N_HOLD = 200


def rule_self_check() -> None:
    """Lock select_k.choose, including its float comparison.

    Hits@1 on 200 questions is c/200. A gap of one question (0.005) ties.
    A gap of two questions is 0.010000000000000009, so `best - hits <= 0.01`
    is false. That is the v1 function; do not widen it.
    """
    if BASE_KS != (100, 200, 400) or EXTRA_K != 800:
        raise SystemExit(f"K grid drifted: {BASE_KS} {EXTRA_K}")
    got = choose(
        [
            {"k": 100, "hits1": 99 / 200, "eliminated": False, "skipped": False},
            {"k": 200, "hits1": 100 / 200, "eliminated": False, "skipped": False},
            {"k": 400, "hits1": 98 / 200, "eliminated": False, "skipped": False},
            {"k": 800, "hits1": 1.0, "eliminated": True, "skipped": False},
            {"k": 1600, "hits1": 1.0, "skipped": True, "eliminated": True},
        ]
    )
    if got != 100:
        raise SystemExit(f"choose() drifted from the v1 rule: {got}")
    got_far = choose(
        [
            {"k": 100, "hits1": 98 / 200, "eliminated": False, "skipped": False},
            {"k": 200, "hits1": 100 / 200, "eliminated": False, "skipped": False},
        ]
    )
    if got_far != 200:
        raise SystemExit(f"choose() drifted on a 2-question gap: {got_far}")


def refuse(path: str) -> str:
    p = os.path.realpath(path)
    parts = p.split("/")
    if "hidden" in parts:
        raise SystemExit(f"refuse hidden {p}")
    base = os.path.basename(p)
    if base in {"dev.jsonl", "test.jsonl", "dev", "test"} or base.startswith("dev"):
        raise SystemExit(f"refuse {p}")
    if "test" in base and base.endswith((".jsonl", ".json", ".tsv")):
        raise SystemExit(f"refuse {p}")
    return p


def sha_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_holdout(path: str, meta_q_path: str) -> list[dict]:
    path = refuse(path)
    meta_q_path = refuse(meta_q_path)
    if not path.endswith(HOLD_SUFFIX):
        raise SystemExit(f"refuse holdout {path}")
    if not meta_q_path.endswith(Q_SUFFIX + "/meta_q.json"):
        raise SystemExit(f"refuse meta_q {meta_q_path}")
    meta_q = json.load(open(meta_q_path))
    if meta_q.get("scheme") != SCHEME:
        raise SystemExit(f"refuse meta_q scheme {meta_q.get('scheme')}")
    recorded = meta_q.get("holdout") or {}
    if int(recorded.get("n", -1)) != N_HOLD:
        raise SystemExit(f"refuse meta_q holdout n {recorded.get('n')}")
    if sha_file(path) != recorded.get("sha256"):
        raise SystemExit("holdout sha256 does not match meta_q.json")
    rows = [json.loads(line) for line in open(path) if line.strip()]
    if len(rows) != N_HOLD:
        raise SystemExit(f"expected {N_HOLD} holdout rows, got {len(rows)}")
    for row in rows:
        if "qid" not in row or "question" not in row or "answers" not in row:
            raise SystemExit("holdout row missing qid/question/answers")
        topics = (row.get("meta") or {}).get("topic_entities")
        if not isinstance(topics, list):
            raise SystemExit(f"holdout {row['qid']} missing topic_entities")
    print(
        f"holdout {len(rows)} sha_ok first_qids {[r['qid'] for r in rows[:3]]}",
        flush=True,
    )
    return rows


def check_ckpt(ckpt_path: str) -> dict:
    ckpt_path = refuse(ckpt_path)
    if not ckpt_path.endswith(CKPT_SUFFIX):
        raise SystemExit(f"refuse ckpt {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ckpt.get("scheme") != SCHEME or ckpt.get("task") != "cwq":
        raise SystemExit(f"refuse checkpoint scheme/task {ckpt.get('scheme')} {ckpt.get('task')}")
    if int(ckpt.get("best_epoch", -1)) != 11:
        raise SystemExit(f"refuse best_epoch {ckpt.get('best_epoch')}")
    if int(ckpt.get("hop", 2)) != 2:
        raise SystemExit(f"refuse hop {ckpt.get('hop')}")
    print(
        f"ckpt scheme={ckpt['scheme']} best_epoch={ckpt['best_epoch']} hub={ckpt['hub']} "
        f"total_params={ckpt.get('total_params')}",
        flush=True,
    )
    return ckpt


def load_kg_and_emb(triples: str, emb_dir: str, q_dir: str, ckpt: dict):
    triples = refuse(triples)
    emb_dir = refuse(emb_dir)
    q_dir = refuse(q_dir)
    if not triples.endswith(TRIPLES_SUFFIX):
        raise SystemExit(f"refuse triples {triples}")
    if not emb_dir.endswith(EMB_SUFFIX):
        raise SystemExit(f"refuse emb dir {emb_dir}")
    if not q_dir.endswith(Q_SUFFIX):
        raise SystemExit(f"refuse q dir {q_dir}")
    meta_path = os.path.join(emb_dir, "meta.json")
    meta = json.load(open(refuse(meta_path)))
    if meta.get("scheme") != "formalv2-kg-only" or meta.get("task") != "cwq":
        raise SystemExit(f"refuse emb meta {meta.get('scheme')} {meta.get('task')}")
    if int(meta.get("emb_dim", -1)) != 1024:
        raise SystemExit(f"refuse emb_dim {meta.get('emb_dim')}")
    if os.path.getsize(triples) != int(meta["triples_bytes"]):
        raise SystemExit("triples size does not match 713178 meta")
    if sha_file(triples) != meta["triples_sha256"]:
        raise SystemExit("triples sha256 does not match 713178 meta")
    print("load graph", time.strftime("%H:%M:%S"), flush=True)
    kg = KG(triples)
    print("graph loaded", time.strftime("%H:%M:%S"), flush=True)
    text_names = [kg.names[i] for i in kg.text_ids]
    rel_names = sorted(kg.rel_names)
    if sha_names(text_names) != meta["text_sha256"] or sha_names(rel_names) != meta["rel_sha256"]:
        raise SystemExit("embedding rows do not match this corpus")
    ent_emb = np.load(refuse(os.path.join(emb_dir, "ent_emb.f16.npy")))
    rel_emb = np.load(refuse(os.path.join(emb_dir, "rel_emb.f16.npy")))
    holdout_q = np.load(refuse(os.path.join(q_dir, "holdout_q.f16.npy")))
    if ent_emb.shape != (int(meta["n_text"]), 1024) or rel_emb.shape != (int(meta["n_rel"]), 1024):
        raise SystemExit(f"bad emb shapes {ent_emb.shape} {rel_emb.shape}")
    if holdout_q.shape != (N_HOLD, 1024):
        raise SystemExit(f"bad holdout_q shape {holdout_q.shape}")
    model = Retriever(1024)
    model.load_state_dict(ckpt["state_dict"])
    device = "cpu"
    if torch.cuda.is_available():
        free, _total = torch.cuda.mem_get_info()
        if free > 4 * 1024**3:
            device = "cuda"
    model.to(device)
    model.eval()
    print("rank device", device, flush=True)
    return kg, model, ent_emb, rel_emb, holdout_q, device


def out_dir_ok(path: str, limit: int) -> str:
    p = refuse(path)
    if limit:
        if "results-formalv2" in p or "artifacts" in p or "hidden" in p.split("/"):
            raise SystemExit(f"smoke must not write {p}")
        return p
    if not p.endswith(OUT_SUFFIX):
        raise SystemExit(f"official out dir refused {p}")
    return p


def rank_phase(args, rows_all: list[dict], ckpt: dict) -> None:
    rows = rows_all[: args.limit] if args.limit else rows_all
    kg, model, ent_emb, rel_emb, holdout_q, device = load_kg_and_emb(
        args.triples, args.emb_dir, args.q_dir, ckpt
    )
    hub = int(ckpt["hub"])
    hop = int(ckpt.get("hop", 2))
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "cwq_holdout_ranked.jsonl")
    if os.path.exists(path):
        raise SystemExit(f"refuse overwrite {path}")
    print(f"rank holdout n={len(rows)} hub={hub} device={device}", flush=True)
    t0 = time.time()
    with open(path, "w") as handle:
        for i, row in enumerate(rows, 1):
            items, n_cand, seconds = rank_one(
                kg, model, holdout_q[i - 1], ent_emb, rel_emb, device, row, hub, hop, EXTRA_K
            )
            handle.write(
                json.dumps(
                    {
                        "qid": row["qid"],
                        "index": i - 1,
                        "n_cand": n_cand,
                        "retrieve_s": seconds,
                        "items": items,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            handle.flush()
            if args.limit or i % 10 == 0:
                print(f"  ranked {i}/{len(rows)} cand~{n_cand} sec={seconds:.2f}", flush=True)
    print(
        f"wrote {path} elapsed_s={time.time() - t0:.1f} device={device}",
        flush=True,
    )
    if args.limit:
        print("SMOKE_RANK_DONE", flush=True)


def budget_lines(out_dir: str, summaries: list[dict]) -> None:
    for row in summaries:
        k = row.get("k")
        if row.get("skipped"):
            print(f"BUDGET k={k} skipped=1", flush=True)
            continue
        path = os.path.join(out_dir, f"cwq_holdout_k{k}.jsonl")
        max_calls = 0
        n = 0
        with open(path) as handle:
            for line in handle:
                pred = json.loads(line)
                max_calls = max(max_calls, int((pred.get("gateway") or {}).get("calls") or 0))
                n += 1
        print(
            f"BUDGET k={k} n={n} max_calls={max_calls} "
            f"max_prompt={row.get('max_prompt_tokens')} max_total={row.get('max_total_tokens')} "
            f"max_latency={None if row.get('max_latency') is None else round(row['max_latency'], 3)} "
            f"eliminated={row.get('eliminated')} n_violations={row.get('n_violations')}",
            flush=True,
        )


def generate_phase(args, rows: list[dict]) -> None:
    if args.limit:
        raise SystemExit("generate refuses --limit")
    ranked_path = refuse(args.ranked)
    if not ranked_path.endswith(OUT_SUFFIX + "/cwq_holdout_ranked.jsonl"):
        raise SystemExit(f"refuse ranked {ranked_path}")
    packed = [json.loads(line) for line in open(ranked_path) if line.strip()]
    if len(packed) != len(rows) or [p["qid"] for p in packed] != [r["qid"] for r in rows]:
        raise SystemExit("ranked file does not match this holdout")
    ranked = [(p["items"], p["n_cand"], p["retrieve_s"]) for p in packed]
    os.makedirs(args.out_dir, exist_ok=True)
    gold_path = os.path.join(args.out_dir, "cwq_holdout_gold.jsonl")
    summary_path = os.path.join(args.out_dir, "cwq_holdout_summary.json")
    chosen_path = os.path.join(args.out_dir, "selected_k.json")
    for path in (gold_path, summary_path, chosen_path):
        if os.path.exists(path):
            raise SystemExit(f"refuse overwrite {path}")
    with open(gold_path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summaries = []
    with httpx.Client(timeout=65) as client:
        blocked = False
        for k in BASE_KS:
            summaries.append(run_k(client, args.llm, "cwq", rows, ranked, k, args.out_dir))
            if k == 400 and summaries[-1]["eliminated"]:
                blocked = True
        if blocked:
            summaries.append(
                {
                    "k": EXTRA_K,
                    "skipped": True,
                    "eliminated": True,
                    "reason": "400 eliminated, so 800 is not run",
                }
            )
            print("HOLDOUT cwq K=800 skipped because 400 was eliminated", flush=True)
        else:
            summaries.append(run_k(client, args.llm, "cwq", rows, ranked, EXTRA_K, args.out_dir))
    summary = {
        "task": "cwq",
        "scheme": SCHEME,
        "n": N_HOLD,
        "holdout": "frozen formal-v2-split/cwq-sgrag-train2000/holdout.jsonl (200); not in the loss",
        "hits_step": 1 / N_HOLD,
        "latency": "retrieve_s is the shared top-800 rank; each K adds its own generator time",
        "ks": summaries,
    }
    json.dump(summary, open(summary_path, "w"), indent=2)
    print("wrote", summary_path, flush=True)
    budget_lines(args.out_dir, summaries)
    if summary.get("n") != N_HOLD or summary.get("scheme") != SCHEME:
        raise SystemExit("refuse summary")
    k = choose(summary["ks"])
    detail = {
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
    json.dump({"cwq": k, "scheme": SCHEME, "n": N_HOLD, "detail": {"cwq": detail}}, open(chosen_path, "w"), indent=2)
    print(f"SELECTED cwq K={k}", flush=True)
    print("wrote", chosen_path, flush=True)


def main():
    rule_self_check()
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["rank", "generate"])
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--triples", default=ROOT + TRIPLES_SUFFIX)
    ap.add_argument("--ckpt", default=ROOT + CKPT_SUFFIX)
    ap.add_argument("--emb-dir", default=ROOT + EMB_SUFFIX)
    ap.add_argument("--q-dir", default=ROOT + Q_SUFFIX)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ranked", default="")
    ap.add_argument("--llm", default="")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if np.__version__ != "2.4.6":
        if not args.limit:
            raise SystemExit(f"numpy {np.__version__} is not the pin 2.4.6")
        print(f"SMOKE_NUMPY_WARN {np.__version__} official job pins 2.4.6", flush=True)
    if args.limit and not 1 <= args.limit <= 5:
        raise SystemExit("--limit must be 1..5")
    if args.limit and args.phase != "rank":
        raise SystemExit("smoke is rank only")
    args.out_dir = out_dir_ok(args.out_dir, args.limit)
    meta_q = os.path.join(args.q_dir, "meta_q.json")
    rows = load_holdout(args.holdout, meta_q)
    if args.phase == "rank":
        ckpt = check_ckpt(args.ckpt)
        rank_phase(args, rows, ckpt)
        return
    if not args.llm or not args.ranked:
        raise SystemExit("generate needs --llm and --ranked")
    generate_phase(args, rows)


if __name__ == "__main__":
    main()
