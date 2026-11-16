"""Encode formal-v2 CWQ entity texts and relation names.

No train questions, no dev split, no retriever training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time

import numpy as np

from encoder import embed_texts, load_encoder
from graph import KG

TRIPLES_SUFFIX = "/data_formal-v2/cwq/public/corpus/triples.tsv"
SCHEME = "formalv2-kg-only"
BATCH = 128
MAX_LENGTH = 128


def refuse_read(path: str) -> None:
    p = os.path.abspath(path)
    if "/hidden/" in p or p.endswith("/test.jsonl"):
        raise SystemExit(f"refuse test path {p}")
    tail = p.split("/public/")[-1]
    if "/test/" in tail:
        raise SystemExit(f"refuse test path {p}")


def refuse_out(path: str) -> None:
    p = os.path.abspath(path)
    banned = (
        "/hidden/",
        "/artifacts-persist/",
        "/artifacts-formalv1/",
        "/results-formalv1/",
        "/rageng/artifacts/",
        "/runs/",
        "/shared-indexes/",
        "/data_formal-v1/",
        "/data_formal-v2/",
    )
    probe = p if p.endswith("/") else p + "/"
    for item in banned:
        if item in probe or p.rstrip("/").endswith(item.rstrip("/")):
            raise SystemExit(f"refuse output path {p}")


def sha_names(names) -> str:
    h = hashlib.sha256()
    for name in names:
        h.update(name.encode())
        h.update(b"\n")
    return h.hexdigest()


def sha_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["cwq"])
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--triples", required=True)
    ap.add_argument("--scheme", default=SCHEME)
    args = ap.parse_args()
    if args.scheme != SCHEME:
        raise SystemExit(args.scheme)
    if args.batch != BATCH:
        raise SystemExit(f"batch must be {BATCH}")
    if not args.triples.endswith(TRIPLES_SUFFIX):
        raise SystemExit(args.triples)
    refuse_read(args.triples)
    refuse_read(args.encoder)
    refuse_out(args.out)
    os.makedirs(args.out, exist_ok=True)

    print("load graph", args.triples, flush=True)
    stop = threading.Event()
    started = time.time()

    def heartbeat():
        while not stop.wait(60):
            print(f"GRAPH_LOAD_WAIT seconds={int(time.time() - started)}", flush=True)

    beater = threading.Thread(target=heartbeat, daemon=True)
    beater.start()
    kg = KG(args.triples)
    stop.set()

    text_names = [kg.names[i] for i in kg.text_ids]
    rel_names = sorted(kg.rel_names)
    n_triples = int(kg.heads.shape[0])
    print(
        f"ENCODED_TRIPLES task={args.task} n_triples={n_triples} "
        f"n_text={len(text_names)} n_rel={len(rel_names)} n_cvt={int(kg.is_cvt.sum())}",
        flush=True,
    )
    st = os.stat(args.triples)
    triples_sha = sha_file(args.triples)
    print(f"TRIPLES_SHA256 {triples_sha} inode={st.st_ino} bytes={st.st_size}", flush=True)

    tok, model, n_params = load_encoder(args.encoder, args.device)
    print(f"encoder_params {n_params}", flush=True)
    if n_params > 1_000_000_000:
        raise SystemExit(f"encoder alone exceeds 1B: {n_params}")
    print("embed entities", flush=True)
    ent = embed_texts(tok, model, text_names, batch=BATCH, max_length=MAX_LENGTH)
    print("embed relations", flush=True)
    rel = embed_texts(tok, model, rel_names, batch=BATCH, max_length=MAX_LENGTH)
    if ent.dtype != np.float16 or ent.shape != (len(text_names), 1024):
        raise SystemExit(f"bad ent emb {ent.dtype} {ent.shape}")
    if rel.dtype != np.float16 or rel.shape != (len(rel_names), 1024):
        raise SystemExit(f"bad rel emb {rel.dtype} {rel.shape}")
    np.save(os.path.join(args.out, "ent_emb.f16.npy"), ent)
    np.save(os.path.join(args.out, "rel_emb.f16.npy"), rel)
    meta = {
        "task": args.task,
        "encoder": args.encoder,
        "encoder_name": "Alibaba-NLP/gte-large-en-v1.5",
        "encoder_params": n_params,
        "emb_dim": int(ent.shape[1]),
        "batch": BATCH,
        "max_length": MAX_LENGTH,
        "n_text": len(text_names),
        "n_rel": len(rel_names),
        "n_triples": n_triples,
        "n_cvt": int(kg.is_cvt.sum()),
        "text_sha256": sha_names(text_names),
        "rel_sha256": sha_names(rel_names),
        "triples_sha256": triples_sha,
        "triples_inode": int(st.st_ino),
        "triples_bytes": int(st.st_size),
        "triples_path": os.path.abspath(args.triples),
        "scheme": SCHEME,
    }
    with open(os.path.join(args.out, "meta.json"), "w") as handle:
        json.dump(meta, handle, indent=2)
        handle.write("\n")
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
