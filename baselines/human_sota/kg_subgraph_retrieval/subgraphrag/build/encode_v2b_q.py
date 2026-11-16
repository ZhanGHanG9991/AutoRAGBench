"""formal-v2 CWQ (train1800 scheme): encode train_fit 1,800 + holdout 200 questions (gte-large-en-v1.5, batch 32,
max_length 256 as encode.py). Entity / relation embeddings are reused from 713178. No dev, no test, no hidden/."""
from __future__ import annotations
import argparse, hashlib, json, os
import numpy as np
from encoder import embed_texts, load_encoder
SPLIT = "/humansota/formal-v2-split/cwq-sgrag-train2000/"
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True); ap.add_argument("--train", required=True)
    ap.add_argument("--holdout", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert a.train.endswith(SPLIT + "train_fit.jsonl") and a.holdout.endswith(SPLIT + "holdout.jsonl")
    for p in (a.train, a.holdout, a.out):
        assert "/hidden/" not in os.path.abspath(p) and "dev" not in os.path.basename(p) and "test" not in os.path.basename(p)
    tok, model, n_params = load_encoder(a.encoder, "cuda")
    meta = {"scheme": "formalv2-train1800", "encoder_params": n_params, "batch": 32, "max_length": 256}
    for name, path in (("train", a.train), ("holdout", a.holdout)):
        rows = [json.loads(l) for l in open(path) if l.strip()]
        np.save(os.path.join(a.out, f"{name}_q.f16.npy"), embed_texts(tok, model, [r["question"] for r in rows], batch=32, max_length=256))
        meta[name] = {"path": path, "sha256": sha(path), "n": len(rows)}
        print(f"ENCODED_Q {name} {len(rows)}", flush=True)
    json.dump(meta, open(os.path.join(a.out, "meta_q.json"), "w"), indent=2)
if __name__ == "__main__":
    main()
