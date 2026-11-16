"""Train one SubgraphRAG retriever on AutoRAGBench train 500. Scheme (a) only."""
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
import time

import numpy as np
import torch
import torch.nn.functional as F

from graph import HUB_CANDIDATES, KG, localize
from model import Retriever

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAGENG = os.environ.get(
    "RAGENG_ROOT", _release_path("@AUTORAGBENCH_ROOT@")
)


def refuse(path: str) -> None:
    p = os.path.abspath(path)
    if "/hidden/" in p or p.endswith("/test.jsonl"):
        raise SystemExit(f"refuse test path {p}")


def read_jsonl(path):
    refuse(path)
    return [json.loads(line) for line in open(path)]


def build_samples(kg, rows, hub, q_emb):
    samples = []
    n_path = 0
    n_capped = 0
    for i, row in enumerate(rows):
        extracted = kg.extract(row["meta"]["topic_entities"], hub)
        if extracted is None:
            samples.append(None)
            continue
        loc = localize(kg, extracted, row["answers"])
        loc["q"] = q_emb[i]
        loc["qid"] = row["qid"]
        samples.append(loc)
        n_path += int(loc["has_path"])
        n_capped += int(loc["capped"])
        if (i + 1) % 100 == 0:
            print(f"  built {i+1}/{len(rows)} path {n_path} capped {n_capped}", flush=True)
    return samples, n_path, n_capped


def batches(samples, device, ent_emb, rel_emb):
    for loc in samples:
        if loc is None or loc["h"].size == 0 or not loc["has_path"]:
            continue
        text = torch.from_numpy(ent_emb[loc["text_rows"]].astype(np.float32)).to(device)
        rel = torch.from_numpy(rel_emb[loc["rel_rows"]].astype(np.float32)).to(device)
        yield {
            "h": torch.from_numpy(loc["h"]).to(device),
            "r": torch.from_numpy(loc["r"]).to(device),
            "t": torch.from_numpy(loc["t"]).to(device),
            "q": torch.from_numpy(loc["q"].astype(np.float32)).to(device),
            "ent": text,
            "n_non": loc["n_non_text"],
            "rel": rel,
            "oh": torch.from_numpy(loc["topic_oh"]).to(device),
            "y": torch.from_numpy(loc["labels"]).to(device),
            "ans": loc["ans_local"],
        }


@torch.no_grad()
def evaluate(model, samples, device, ent_emb, rel_emb):
    model.eval()
    triple_rec, ans_rec = [], []
    for batch in batches(samples, device, ent_emb, rel_emb):
        logits = model(
            batch["h"], batch["r"], batch["t"], batch["q"], batch["ent"], batch["n_non"], batch["rel"], batch["oh"]
        ).squeeze(-1)
        y = batch["y"]
        pos = (y > 0.5).nonzero(as_tuple=False).flatten()
        if pos.numel() == 0:
            continue
        order = torch.argsort(logits, descending=True)
        ranks = torch.empty_like(order)
        ranks[order] = torch.arange(order.shape[0], device=device)
        triple_rec.append(float((ranks[pos] < 100).float().mean()))
        if batch["ans"]:
            k = min(100, order.shape[0])
            top = order[:k]
            hit = torch.zeros(batch["oh"].shape[0], device=device)
            hit[batch["h"][top]] = 1
            hit[batch["t"][top]] = 1
            ans = torch.tensor(batch["ans"], device=device)
            ans_rec.append(float(hit[ans].mean()))
    return {
        "triple_recall@100": float(np.mean(triple_rec)) if triple_rec else 0.0,
        "ans_recall@100": float(np.mean(ans_rec)) if ans_rec else 0.0,
        "n": len(triple_rec),
    }


def train_epoch(model, opt, samples, device, ent_emb, rel_emb):
    model.train()
    losses = []
    for batch in batches(samples, device, ent_emb, rel_emb):
        logits = model(
            batch["h"], batch["r"], batch["t"], batch["q"], batch["ent"], batch["n_non"], batch["rel"], batch["oh"]
        )
        loss = F.binary_cross_entropy_with_logits(logits.squeeze(-1), batch["y"])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    return float(np.mean(losses)) if losses else 0.0


def _shape_check() -> None:
    model = Retriever(32)
    h = torch.tensor([0, 1])
    r = torch.tensor([0, 0])
    t = torch.tensor([1, 0])
    q = torch.randn(32)
    ent = torch.randn(2, 32)
    rel = torch.randn(1, 32)
    oh = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    y = model(h, r, t, q, ent, 0, rel, oh)
    if tuple(y.shape) != (2, 1) or model.pred_in != 4 * 32 + 4 + 16:
        raise SystemExit(f"shape check failed {tuple(y.shape)} pred_in={model.pred_in}")
    print(f"shape_check ok pred_in={model.pred_in}", flush=True)


def main():
    _shape_check()
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["webqsp", "cwq"])
    ap.add_argument("--artifacts", default="")
    ap.add_argument("--minutes", type=float, default=25)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--train", default="")
    ap.add_argument("--triples", default="")
    ap.add_argument("--scheme", default="a-train500")
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--holdout", type=int, default=50)
    args = ap.parse_args()
    if np.__version__ != "2.4.6" and args.scheme == "formalv1-train1800":
        raise SystemExit(f"numpy {np.__version__} is not the pin 2.4.6")
    art = args.artifacts or os.path.join(ROOT, "rageng", "artifacts", args.task)
    art_abs = os.path.abspath(art)
    if "/artifacts-persist/" in art_abs or art_abs.rstrip("/").endswith("/artifacts-persist"):
        raise SystemExit(f"refuse artifacts-persist {art}")
    if "/rageng/artifacts/" in (art_abs + "/") or art_abs.rstrip("/").endswith("/rageng/artifacts"):
        raise SystemExit(f"refuse pilot artifacts {art}")
    if args.scheme == "a-train500":
        train_path = args.train or os.path.join(RAGENG, "data", args.task, "public", "train.jsonl")
        triples_path = args.triples or os.path.join(RAGENG, "data", args.task, "public", "corpus", "triples.tsv")
        if not train_path.endswith("/public/train.jsonl") or args.n_train != 500 or args.holdout != 50:
            raise SystemExit(train_path)
    elif args.scheme == "formalv1-train1800":
        train_path = args.train
        triples_path = args.triples
        if not train_path.endswith(f"/formal-v1-split/{args.task}/train_fit.jsonl"):
            raise SystemExit(train_path)
        if not triples_path.endswith(f"/data_formal-v1/{args.task}/public/corpus/triples.tsv"):
            raise SystemExit(triples_path)
        if args.n_train != 1800 or args.holdout != 200:
            raise SystemExit("formal train is 1800 with holdout 200")
    else:
        raise SystemExit(args.scheme)
    rows = read_jsonl(train_path)
    if len(rows) != args.n_train:
        raise SystemExit(f"expected {args.n_train} train rows, got {len(rows)}")
    meta = json.load(open(os.path.join(art, "meta.json")))
    if meta.get("scheme") != args.scheme or meta.get("n_train") != args.n_train or meta.get("task") != args.task:
        raise SystemExit(f"artifacts do not match this run: {meta}")
    ent_emb = np.load(os.path.join(art, "ent_emb.f16.npy"))
    rel_emb = np.load(os.path.join(art, "rel_emb.f16.npy"))
    q_emb = np.load(os.path.join(art, "train_q.f16.npy"))
    if q_emb.shape[0] != len(rows):
        raise SystemExit(f"train_q rows {q_emb.shape[0]} != {len(rows)}")
    print("load graph", flush=True)
    kg = KG(triples_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hub = None
    samples = None
    n_path = 0
    n_capped = 0
    coverage = {}
    for cand in HUB_CANDIDATES:
        t0 = time.time()
        samples, n_path, n_capped = build_samples(kg, rows, cand, q_emb)
        rate = n_path / len(rows)
        coverage[cand] = {"path": n_path, "capped": n_capped, "seconds": round(time.time() - t0, 1)}
        print(f"hub {cand} train path {n_path}/{len(rows)} capped {n_capped}", flush=True)
        if rate >= 0.98:
            hub = cand
            break
    if hub is None:
        hub = HUB_CANDIDATES[-1]
        print(f"no hub reached 0.98, using {hub}", flush=True)
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(rows))
    val_ids = set(int(i) for i in perm[: args.holdout])
    print(f"numpy {np.__version__} holdout {args.holdout} of {len(rows)} loss {len(rows) - args.holdout}", flush=True)
    train_s = [samples[i] for i in range(len(samples)) if i not in val_ids]
    val_s = [samples[i] for i in range(len(samples)) if i in val_ids]
    dim = int(ent_emb.shape[1])
    model = Retriever(dim).to(device)
    n_ret = sum(p.numel() for p in model.parameters())
    total = int(meta["encoder_params"]) + n_ret
    print(f"params encoder {meta['encoder_params']} retriever {n_ret} dde 0 total {total}", flush=True)
    if total > 1_000_000_000:
        raise SystemExit(f"params {total} exceed 1B")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best = -1.0
    best_state = None
    patience = 0
    history = []
    deadline = time.time() + args.minutes * 60
    for epoch in range(args.epochs):
        if time.time() > deadline:
            print("hit train time cap", flush=True)
            break
        metrics = evaluate(model, val_s, device, ent_emb, rel_emb)
        print(f"epoch {epoch} val {metrics}", flush=True)
        history.append({"epoch": epoch, **metrics})
        if metrics["triple_recall@100"] > best:
            best = metrics["triple_recall@100"]
            patience = 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if patience >= args.patience:
            print("early stop", flush=True)
            break
        loss = train_epoch(model, opt, train_s, device, ent_emb, rel_emb)
        print(f"epoch {epoch} loss {loss:.4f}", flush=True)
        history[-1]["loss"] = loss
    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    ckpt = {
        "scheme": args.scheme,
        "task": args.task,
        "hub": hub,
        "hop": 2,
        "emb_dim": dim,
        "state_dict": best_state,
        "encoder_params": meta["encoder_params"],
        "retriever_params": n_ret,
        "total_params": total,
        "best_val_triple_recall@100": best,
        "train_path": n_path,
        "train_capped": n_capped,
        "coverage": {str(k): v for k, v in coverage.items()},
        "holdout": args.holdout,
        "seed": 42,
    }
    torch.save(ckpt, os.path.join(art, "ckpt.pt"))
    json.dump(history, open(os.path.join(art, "train_log.json"), "w"), indent=2)
    print("saved", os.path.join(art, "ckpt.pt"), "best", best, flush=True)


if __name__ == "__main__":
    main()
