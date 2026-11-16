"""Encode entity names, relation names, and the 500 train questions. No dev, no test."""
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

import numpy as np

from encoder import embed_texts, load_encoder
from graph import KG

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAGENG = os.environ.get(
    "RAGENG_ROOT", _release_path("@AUTORAGBENCH_ROOT@")
)


def refuse(path: str) -> None:
    p = os.path.abspath(path)
    if "/hidden/" in p or p.endswith("/test.jsonl") or "/test/" in p.split("/public/")[-1]:
        raise SystemExit(f"refuse test path {p}")


def sha_names(names) -> str:
    h = hashlib.sha256()
    for name in names:
        h.update(name.encode())
        h.update(b"\n")
    return h.hexdigest()


def refuse_out(path: str, scheme: str) -> None:
    p = os.path.abspath(path)
    if "/artifacts-persist/" in p or p.rstrip("/").endswith("/artifacts-persist"):
        raise SystemExit(f"refuse artifacts-persist {p}")
    if "/rageng/artifacts/" in (p + "/") or p.rstrip("/").endswith("/rageng/artifacts"):
        raise SystemExit(f"refuse pilot artifacts {p}")
    if scheme == "a-train500" and "/artifacts-formalv1/" in p:
        raise SystemExit(f"pilot output must not use artifacts-formalv1: {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["webqsp", "cwq"])
    ap.add_argument("--encoder", default=os.path.join(ROOT, "models", "gte-large-en-v1.5"))
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--train", default="")
    ap.add_argument("--triples", default="")
    ap.add_argument("--scheme", default="a-train500")
    ap.add_argument("--n-train", type=int, default=500)
    args = ap.parse_args()
    if args.scheme == "a-train500":
        train_path = args.train or os.path.join(RAGENG, "data", args.task, "public", "train.jsonl")
        triples_path = args.triples or os.path.join(RAGENG, "data", args.task, "public", "corpus", "triples.tsv")
        if not train_path.endswith("/public/train.jsonl") or args.n_train != 500:
            raise SystemExit(train_path)
        out = args.out or os.path.join(ROOT, "rageng", "artifacts", args.task)
    elif args.scheme == "formalv1-train1800":
        train_path = args.train
        triples_path = args.triples
        if not train_path.endswith(f"/formal-v1-split/{args.task}/train_fit.jsonl"):
            raise SystemExit(train_path)
        if not triples_path.endswith(f"/data_formal-v1/{args.task}/public/corpus/triples.tsv"):
            raise SystemExit(triples_path)
        if args.n_train != 1800 or not args.out:
            raise SystemExit("formal encode needs --n-train 1800 and --out")
        out = args.out
    else:
        raise SystemExit(args.scheme)
    refuse(train_path)
    refuse(triples_path)
    refuse_out(out, args.scheme)
    os.makedirs(out, exist_ok=True)
    print("load graph", triples_path, flush=True)
    kg = KG(triples_path)
    text_names = [kg.names[i] for i in kg.text_ids]
    rel_names = sorted(kg.rel_names)
    n_triples = int(kg.heads.shape[0])
    print(
        f"ENCODED_TRIPLES task={args.task} n_triples={n_triples} "
        f"n_text={len(text_names)} n_rel={len(rel_names)} n_cvt={int(kg.is_cvt.sum())}",
        flush=True,
    )
    rows = [json.loads(line) for line in open(train_path)]
    if len(rows) != args.n_train:
        raise SystemExit(f"expected {args.n_train} train questions, got {len(rows)}")
    questions = [r["question"] for r in rows]
    tok, model, n_params = load_encoder(args.encoder, args.device)
    print(f"encoder_params {n_params}", flush=True)
    if n_params > 1_000_000_000:
        raise SystemExit(f"encoder alone exceeds 1B: {n_params}")
    print("embed entities", flush=True)
    ent = embed_texts(tok, model, text_names, batch=args.batch, max_length=128)
    print("embed relations", flush=True)
    rel = embed_texts(tok, model, rel_names, batch=args.batch, max_length=128)
    print("embed train questions", flush=True)
    q = embed_texts(tok, model, questions, batch=32, max_length=256)
    np.save(os.path.join(out, "ent_emb.f16.npy"), ent)
    np.save(os.path.join(out, "rel_emb.f16.npy"), rel)
    np.save(os.path.join(out, "train_q.f16.npy"), q)
    meta = {
        "task": args.task,
        "encoder": args.encoder,
        "encoder_params": n_params,
        "emb_dim": int(ent.shape[1]),
        "n_text": len(text_names),
        "n_rel": len(rel_names),
        "n_train": args.n_train,
        "n_triples": n_triples,
        "text_sha256": sha_names(text_names),
        "rel_sha256": sha_names(rel_names),
        "scheme": args.scheme,
    }
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=2)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
