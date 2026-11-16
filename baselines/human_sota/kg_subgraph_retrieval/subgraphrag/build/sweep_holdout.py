"""Sweep K on the scheme-(a) early-stop holdout. Does not read dev or test.

The 50 questions are the same split as train.py: seed 42, first 50 of a
permutation of the 500 train rows. They were used for early stopping and did
not enter the loss. Hits@1 on 50 questions moves in steps of 0.02.
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
import json
import os
import subprocess
import time

import httpx
import numpy as np
import torch

from encode import sha_names
from graph import KG, localize
from model import Retriever
from server import _clean, _prompt

RAGENG = os.environ.get(
    "RAGENG_ROOT", _release_path("@AUTORAGBENCH_ROOT@")
)
RAGENG_PY = os.environ.get("RAGENG_PY", os.path.join(RAGENG, "envs/rag-base/bin/python"))
BASE_KS = (100, 200, 400)
EXTRA_K = 800


def holdout_indices(n: int, n_holdout: int) -> list[int]:
    if np.__version__ != "2.4.6":
        raise SystemExit(f"numpy {np.__version__} is not the pin 2.4.6")
    if (n, n_holdout) not in ((500, 50), (1800, 200)):
        raise SystemExit(f"unexpected split n={n} holdout={n_holdout}")
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    return [int(i) for i in perm[:n_holdout]]


def rank_one(kg, model, q_emb, ent_emb, rel_emb, device, row, hub, hop, top):
    t0 = time.time()
    topics = (row.get("meta") or {}).get("topic_entities") or []
    extracted = kg.extract(topics, hub, hop=hop)
    if extracted is None:
        return [], 0, time.time() - t0
    loc = localize(kg, extracted, answer_names=None)
    if loc["h"].size == 0:
        return [], int(loc["n_cand"]), time.time() - t0
    ent = torch.from_numpy(ent_emb[loc["text_rows"]].astype(np.float32)).to(device)
    rel = torch.from_numpy(rel_emb[loc["rel_rows"]].astype(np.float32)).to(device)
    with torch.no_grad():
        scores = model(
            torch.from_numpy(loc["h"]).to(device),
            torch.from_numpy(loc["r"]).to(device),
            torch.from_numpy(loc["t"]).to(device),
            torch.from_numpy(np.asarray(q_emb, dtype=np.float32)).to(device),
            ent,
            loc["n_non_text"],
            rel,
            torch.from_numpy(loc["topic_oh"]).to(device),
        ).squeeze(-1)
    order = torch.argsort(scores, descending=True).tolist()
    items = []
    triples = loc["global_triples"]
    for idx in order[:top]:
        h, r, t = triples[idx]
        hn, rn, tn = kg.names[h], kg.rel_names[r], kg.names[t]
        items.append((f"({hn},{rn},{tn})", hn, tn))
    return items, int(loc["n_cand"]), time.time() - t0


def one_call(client, llm, question, lines):
    url = llm.rstrip("/") + "/chat/completions"
    t0 = time.time()
    try:
        response = client.post(
            url,
            json={
                "model": "generator",
                "max_tokens": 128,
                "temperature": 0,
                "frequency_penalty": 0,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": _prompt(question, lines)}],
            },
        )
    except Exception as exc:
        return "", {}, time.time() - t0, f"{type(exc).__name__}: {exc}"
    elapsed = time.time() - t0
    if response.status_code != 200:
        return "", {}, elapsed, f"http {response.status_code}: {response.text[:300]}"
    data = response.json()
    message = data["choices"][0]["message"]
    content = message.get("content") or ""
    if not content.strip():
        reasoning = message.get("reasoning_content") or ""
        if reasoning.strip():
            content = reasoning.strip().splitlines()[-1]
    return _clean(content), data.get("usage") or {}, elapsed, None


def score_file(task, pred, gold, out):
    env = os.environ.copy()
    env["PYTHONPATH"] = RAGENG + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.check_call(
        [RAGENG_PY, "-m", "tasks.score", "--task", task, "--pred", pred, "--gold", gold, "--out", out],
        env=env,
    )
    scored = json.load(open(out))
    return float(scored["hits1"]), float(scored["f1"])


def run_k(client, llm, task, rows, ranked, k, out_dir):
    pred_path = os.path.join(out_dir, f"{task}_holdout_k{k}.jsonl")
    gold_path = os.path.join(out_dir, f"{task}_holdout_gold.jsonl")
    reasons = []
    recalls = []
    prompt_tokens = []
    total_tokens = []
    latencies = []
    retrieve_times = []
    llm_times = []
    n_keep = []
    with open(pred_path, "w") as pred_f:
        for row, (items, _n_cand, retrieve_s) in zip(rows, ranked):
            kept = items[:k]
            lines = [item[0] for item in kept]
            endpoints = []
            seen = set()
            for _line, hn, tn in kept:
                for name in (hn, tn):
                    if name not in seen:
                        seen.add(name)
                        endpoints.append(name)
            answer, usage, llm_s, error = one_call(client, llm, row["question"], lines)
            latency = retrieve_s + llm_s
            ptok = usage.get("prompt_tokens")
            ttok = usage.get("total_tokens")
            calls = 1
            why = []
            if error:
                why.append(error)
            if calls != 1:
                why.append(f"calls {calls}")
            if ptok is None or ttok is None:
                why.append("missing token count")
            else:
                if int(ptok) > 32000 or int(ttok) > 32000:
                    why.append(f"tokens prompt={ptok} total={ttok}")
            if latency > 60:
                why.append(f"latency {latency:.1f}")
            hit = any(a in endpoints for a in row.get("answers") or [])
            recalls.append(hit)
            latencies.append(latency)
            retrieve_times.append(retrieve_s)
            llm_times.append(llm_s)
            n_keep.append(len(lines))
            if ptok is not None:
                prompt_tokens.append(int(ptok))
            if ttok is not None:
                total_tokens.append(int(ttok))
            if why:
                reasons.append({"qid": row["qid"], "why": why})
            pred = {
                "qid": row["qid"],
                "answer": answer,
                "error": error,
                "over_budget": bool(why),
                "latency": round(latency, 3),
                "n_keep": len(lines),
                "answer_in_subgraph": hit,
                "gateway": {
                    "calls": calls,
                    "prompt_tokens": int(ptok or 0),
                    "completion_tokens": int(usage.get("completion_tokens") or 0),
                    "total_tokens": int(ttok or 0),
                    "seconds": round(latency, 3),
                },
            }
            pred_f.write(json.dumps(pred, ensure_ascii=False) + "\n")
            pred_f.flush()
    hits1, f1 = score_file(task, pred_path, gold_path, os.path.join(out_dir, f"{task}_holdout_k{k}_score.json"))
    summary = {
        "k": k,
        "skipped": False,
        "n": len(rows),
        "hits1": hits1,
        "f1": f1,
        "answer_recall": sum(recalls) / len(rows),
        "mean_prompt_tokens": (sum(prompt_tokens) / len(prompt_tokens)) if prompt_tokens else None,
        "max_prompt_tokens": max(prompt_tokens) if prompt_tokens else None,
        "mean_total_tokens": (sum(total_tokens) / len(total_tokens)) if total_tokens else None,
        "max_total_tokens": max(total_tokens) if total_tokens else None,
        "mean_latency": sum(latencies) / len(latencies),
        "max_latency": max(latencies),
        "mean_retrieve_seconds": sum(retrieve_times) / len(retrieve_times),
        "max_retrieve_seconds": max(retrieve_times),
        "mean_llm_seconds": sum(llm_times) / len(llm_times),
        "max_llm_seconds": max(llm_times),
        "mean_kept": sum(n_keep) / len(n_keep),
        "eliminated": bool(reasons),
        "n_violations": len(reasons),
        "violations": reasons,
    }
    print(
        f"HOLDOUT {task} K={k} hits1={hits1:.4f} f1={f1:.4f} "
        f"recall={summary['answer_recall']:.4f} "
        f"prompt_mean={summary['mean_prompt_tokens']} prompt_max={summary['max_prompt_tokens']} "
        f"lat_mean={summary['mean_latency']:.3f} lat_max={summary['max_latency']:.3f} "
        f"eliminated={summary['eliminated']}",
        flush=True,
    )
    return summary


def load_holdout(task: str, train_path: str, n_train: int, n_holdout: int):
    if "/hidden/" in train_path or train_path.endswith("/test.jsonl") or train_path.endswith("/dev.jsonl"):
        raise SystemExit(f"refuse path {train_path}")
    if n_train == 500 and not train_path.endswith("/public/train.jsonl"):
        raise SystemExit(train_path)
    if n_train == 1800 and not train_path.endswith(f"/formal-v1-split/{task}/train_fit.jsonl"):
        raise SystemExit(train_path)
    rows_all = [json.loads(line) for line in open(train_path)]
    if len(rows_all) != n_train:
        raise SystemExit(f"expected {n_train} train rows, got {len(rows_all)}")
    indices = holdout_indices(len(rows_all), n_holdout)
    rows = [rows_all[i] for i in indices]
    print(f"numpy {np.__version__} holdout {len(rows)} first_qids {[r['qid'] for r in rows[:3]]}", flush=True)
    return indices, rows


def rank_phase(task: str, artifacts: str, out_dir: str, train_path: str, triples_path: str, n_train: int, n_holdout: int, scheme: str) -> None:
    indices, rows = load_holdout(task, train_path, n_train, n_holdout)
    meta = json.load(open(os.path.join(artifacts, "meta.json")))
    if meta.get("scheme") != scheme or meta.get("n_train") != n_train or meta.get("task") != task:
        raise SystemExit(f"refuse artifacts {meta}")
    ckpt = torch.load(os.path.join(artifacts, "ckpt.pt"), map_location="cpu", weights_only=False)
    if ckpt.get("scheme") != scheme or ckpt.get("task") != task or int(ckpt.get("holdout", -1)) != n_holdout:
        raise SystemExit("refuse checkpoint")
    if "/hidden/" in triples_path or triples_path.endswith("/test.jsonl"):
        raise SystemExit(f"refuse triples {triples_path}")
    if scheme == "formalv1-train1800" and not triples_path.endswith(f"/data_formal-v1/{task}/public/corpus/triples.tsv"):
        raise SystemExit(triples_path)
    if scheme == "a-train500" and not triples_path.endswith(f"/data/{task}/public/corpus/triples.tsv"):
        raise SystemExit(triples_path)
    kg = KG(triples_path)
    text_names = [kg.names[i] for i in kg.text_ids]
    rel_names = sorted(kg.rel_names)
    if sha_names(text_names) != meta["text_sha256"] or sha_names(rel_names) != meta["rel_sha256"]:
        raise SystemExit("embedding rows do not match this corpus")
    device = "cpu"
    if torch.cuda.is_available():
        free, _total = torch.cuda.mem_get_info()
        if free > 4 * 1024**3:
            device = "cuda"
    print("rank device", device, flush=True)
    # train_q is in train-file order, so row i is the same vector train.py scored.
    model = Retriever(int(meta["emb_dim"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    ent_emb = np.load(os.path.join(artifacts, "ent_emb.f16.npy"))
    rel_emb = np.load(os.path.join(artifacts, "rel_emb.f16.npy"))
    train_q = np.load(os.path.join(artifacts, "train_q.f16.npy"))
    if train_q.shape[0] != n_train or train_q.shape[1] != int(meta["emb_dim"]):
        raise SystemExit(f"bad train_q shape {train_q.shape}")
    hub = int(ckpt["hub"])
    hop = int(ckpt.get("hop", 2))
    print(f"rank holdout hub={hub} train_q={tuple(train_q.shape)}", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{task}_holdout_ranked.jsonl")
    with open(path, "w") as f:
        for i, (idx, row) in enumerate(zip(indices, rows), 1):
            items, n_cand, seconds = rank_one(
                kg, model, train_q[idx], ent_emb, rel_emb, device, row, hub, hop, EXTRA_K
            )
            f.write(json.dumps({
                "qid": row["qid"],
                "index": idx,
                "n_cand": n_cand,
                "retrieve_s": seconds,
                "items": items,
            }, ensure_ascii=False) + "\n")
            f.flush()
            if i % 10 == 0:
                print(f"  ranked {i}/50 cand~{n_cand} sec={seconds:.2f}", flush=True)
    print("wrote", path, flush=True)


def generate_phase(task: str, llm: str, ranked_path: str, out_dir: str, train_path: str, n_train: int, n_holdout: int, scheme: str) -> None:
    _indices, rows = load_holdout(task, train_path, n_train, n_holdout)
    if "/hidden/" in ranked_path or ranked_path.endswith("/test.jsonl") or ranked_path.endswith("/dev.jsonl"):
        raise SystemExit(f"refuse ranked path {ranked_path}")
    packed = [json.loads(line) for line in open(ranked_path)]
    if len(packed) != len(rows) or [p["qid"] for p in packed] != [r["qid"] for r in rows]:
        raise SystemExit("ranked file does not match this holdout")
    ranked = [(p["items"], p["n_cand"], p["retrieve_s"]) for p in packed]
    os.makedirs(out_dir, exist_ok=True)
    gold_path = os.path.join(out_dir, f"{task}_holdout_gold.jsonl")
    with open(gold_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summaries = []
    with httpx.Client(timeout=65) as client:
        blocked = False
        for k in BASE_KS:
            summaries.append(run_k(client, llm, task, rows, ranked, k, out_dir))
            if k == 400 and summaries[-1]["eliminated"]:
                blocked = True
        if blocked:
            summaries.append({
                "k": EXTRA_K,
                "skipped": True,
                "eliminated": True,
                "reason": "400 eliminated, so 800 is not run",
            })
            print(f"HOLDOUT {task} K=800 skipped because 400 was eliminated", flush=True)
        else:
            summaries.append(run_k(client, llm, task, rows, ranked, EXTRA_K, out_dir))
    out = {
        "task": task,
        "scheme": scheme,
        "n": n_holdout,
        "holdout": f"seed 42, first {n_holdout} of permutation({n_train}), early-stop split, not in the loss",
        "hits_step": 1 / n_holdout,
        "latency": "retrieve_s is the shared top-800 rank; each K adds its own generator time",
        "ks": summaries,
    }
    path = os.path.join(out_dir, f"{task}_holdout_summary.json")
    json.dump(out, open(path, "w"), indent=2)
    print("wrote", path, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["webqsp", "cwq"])
    ap.add_argument("--phase", required=True, choices=["rank", "generate"])
    ap.add_argument("--artifacts", default="")
    ap.add_argument("--llm", default="")
    ap.add_argument("--ranked", default="")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--train", default="")
    ap.add_argument("--triples", default="")
    ap.add_argument("--scheme", default="a-train500")
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--holdout", type=int, default=50)
    args = ap.parse_args()
    train_path = args.train or os.path.join(RAGENG, "data", args.task, "public", "train.jsonl")
    triples_path = args.triples or os.path.join(RAGENG, "data", args.task, "public", "corpus", "triples.tsv")
    if args.phase == "rank":
        if not args.artifacts:
            raise SystemExit("rank needs --artifacts")
        rank_phase(
            args.task, args.artifacts, args.out_dir, train_path, triples_path, args.n_train, args.holdout, args.scheme
        )
    else:
        if not args.llm or not args.ranked:
            raise SystemExit("generate needs --llm and --ranked")
        generate_phase(
            args.task, args.llm, args.ranked, args.out_dir, train_path, args.n_train, args.holdout, args.scheme
        )


if __name__ == "__main__":
    main()
