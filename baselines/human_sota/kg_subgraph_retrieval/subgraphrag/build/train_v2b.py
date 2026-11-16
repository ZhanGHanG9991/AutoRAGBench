"""formal-v2 CWQ retriever training, scheme formalv2-train1800 (configuration; train.py unchanged).
  - train = 1,800 and early-stop holdout = 200, both sampled from data_formal-v2/cwq/public/train.jsonl
    (humansota/formal-v2-split/cwq-sgrag-train2000/, seed 20261022). The v2 dev is never read here.
  - graph = data_formal-v2 triples (713178 embeddings); hub rule computed on train_fit only, as train.py.
  - every improvement on the holdout writes ckpt.pt + train_log.json atomically to --persist (so a timeout
    keeps the best checkpoint).
  - sample building in a fork pool (identical samples), unused `global_triples` dropped.
Model, optimiser, lr, epochs 15, patience 5, labels: as train.py; time cap is a parameter (sbatch: 90 min)."""
from __future__ import annotations
import argparse, json, multiprocessing as mp, os, time
import numpy as np
import torch
from graph import HUB_CANDIDATES, KG, localize
from model import Retriever
from train import batches, evaluate, train_epoch, _shape_check

_KG = None
_ROWS = None
_Q = None

def _one(args):
    i, hub = args
    row = _ROWS[i]
    ext = _KG.extract(row["meta"]["topic_entities"], hub)
    if ext is None:
        return None
    loc = localize(_KG, ext, row["answers"])
    loc.pop("global_triples", None)
    loc["q"] = _Q[i]
    loc["qid"] = row["qid"]
    return loc

def build(kg, rows, q, hub, workers):
    global _KG, _ROWS, _Q
    _KG, _ROWS, _Q = kg, rows, q
    ctx = mp.get_context("fork")
    out = []
    with ctx.Pool(workers) as pool:
        for j, loc in enumerate(pool.imap(_one, [(i, hub) for i in range(len(rows))], chunksize=16)):
            out.append(loc)
            if (j + 1) % 1000 == 0:
                print(f"  built {j+1}/{len(rows)} hub {hub} {time.strftime('%H:%M:%S')}", flush=True)
    n_path = sum(1 for s in out if s is not None and s["has_path"])
    n_cap = sum(1 for s in out if s is not None and s["capped"])
    return out, n_path, n_cap

def save(a, state, hub, dim, meta, n_ret, total, best, n_path, n_capped, coverage, history, epoch):
    ckpt = {"scheme": "formalv2-train1800", "task": "cwq", "hub": hub, "hop": 2, "emb_dim": dim,
            "state_dict": state, "encoder_params": meta["encoder_params"], "retriever_params": n_ret,
            "total_params": total, "best_val_triple_recall@100": best, "val": "v2 train holdout 200 (seed 20261022)",
            "best_epoch": epoch, "train_path": n_path, "train_capped": n_capped,
            "coverage": {str(k): v for k, v in coverage.items()}, "seed": 42}
    os.makedirs(a.persist, exist_ok=True)

    def _atomic(name, dump):
        final = os.path.join(a.persist, name)
        tmp = final + ".tmp"
        with open(tmp, "wb") as f:
            dump(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, final)

    _atomic("ckpt.pt", lambda f: torch.save(ckpt, f))
    _atomic("train_log.json", lambda f: f.write(json.dumps(history, indent=2).encode()))
    print(f"CKPT_SAVED epoch={epoch} {time.strftime('%H:%M:%S')}", flush=True)


def refuse(p):
    p = os.path.abspath(p)
    if "/hidden/" in p or "test" in os.path.basename(p):
        raise SystemExit(f"refuse {p}")

def main():
    _shape_check()
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--persist", required=True)
    ap.add_argument("--triples", required=True)
    ap.add_argument("--minutes", type=float, default=25)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    assert a.train.endswith("/humansota/formal-v2-split/cwq-sgrag-train2000/train_fit.jsonl"), a.train
    assert a.holdout.endswith("/humansota/formal-v2-split/cwq-sgrag-train2000/holdout.jsonl"), a.holdout
    assert "/artifacts-formalv2/" in os.path.abspath(a.persist) and "/hidden/" not in os.path.abspath(a.persist)
    assert a.triples.endswith("/data_formal-v2/cwq/public/corpus/triples.tsv"), a.triples
    for p in (a.train, a.holdout, a.triples):
        refuse(p)
    if np.__version__ != "2.4.6":
        raise SystemExit(f"numpy {np.__version__} is not the pin 2.4.6")
    train_rows = [json.loads(l) for l in open(a.train) if l.strip()]
    dev_rows = [json.loads(l) for l in open(a.holdout) if l.strip()]
    if len(train_rows) != 1800 or len(dev_rows) != 200:
        raise SystemExit(f"rows {len(train_rows)} / {len(dev_rows)}")
    meta = json.load(open(os.path.join(a.artifacts, "meta.json")))
    if meta.get("scheme") != "formalv2-kg-only" or meta.get("task") != "cwq":
        raise SystemExit(f"artifacts mismatch {meta}")
    ent_emb = np.load(os.path.join(a.artifacts, "ent_emb.f16.npy"))
    rel_emb = np.load(os.path.join(a.artifacts, "rel_emb.f16.npy"))
    tq = np.load(os.path.join(a.artifacts, "train_q.f16.npy"))
    dq = np.load(os.path.join(a.artifacts, "holdout_q.f16.npy"))
    assert tq.shape[0] == len(train_rows) and dq.shape[0] == len(dev_rows)
    print("load graph", time.strftime("%H:%M:%S"), flush=True)
    kg = KG(a.triples)
    if len(kg.text_ids) != meta["n_text"] or len(kg.rel_names) != meta["n_rel"]:
        raise SystemExit("graph does not match 713178 embeddings")
    print("graph loaded", time.strftime("%H:%M:%S"), flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hub, samples, n_path, n_capped, coverage = None, None, 0, 0, {}
    for cand in HUB_CANDIDATES:
        t0 = time.time()
        samples = None
        samples, n_path, n_capped = build(kg, train_rows, tq, cand, a.workers)
        coverage[cand] = {"path": n_path, "capped": n_capped, "seconds": round(time.time() - t0, 1)}
        print(f"hub {cand} train path {n_path}/{len(train_rows)} capped {n_capped} {coverage[cand]['seconds']}s", flush=True)
        if n_path / len(train_rows) >= 0.98:
            hub = cand
            break
    if hub is None:
        hub = HUB_CANDIDATES[-1]
        print(f"no hub reached 0.98, using {hub}", flush=True)
    val_s, v_path, v_cap = build(kg, dev_rows, dq, hub, a.workers)
    print(f"holdout(val) hub {hub} path {v_path}/{len(dev_rows)} capped {v_cap}", flush=True)
    dim = int(ent_emb.shape[1])
    model = Retriever(dim).to(device)
    n_ret = sum(p.numel() for p in model.parameters())
    total = int(meta["encoder_params"]) + n_ret
    print(f"params encoder {meta['encoder_params']} retriever {n_ret} total {total}", flush=True)
    if total > 1_000_000_000:
        raise SystemExit("params exceed 1B")
    torch.manual_seed(42)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best, best_state, patience, history, best_ep = -1.0, None, 0, [], None
    deadline = time.time() + a.minutes * 60
    for epoch in range(a.epochs):
        if time.time() > deadline:
            print("hit train time cap", flush=True)
            break
        m = evaluate(model, val_s, device, ent_emb, rel_emb)
        print(f"epoch {epoch} val {m} {time.strftime('%H:%M:%S')}", flush=True)
        history.append({"epoch": epoch, **m})
        if m["triple_recall@100"] > best:
            best, patience, best_ep = m["triple_recall@100"], 0, epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            save(a, best_state, hub, dim, meta, n_ret, total, best, n_path, n_capped, coverage, history, epoch)
        else:
            patience += 1
        if patience >= a.patience:
            print("early stop", flush=True)
            break
        loss = train_epoch(model, opt, samples, device, ent_emb, rel_emb)
        print(f"epoch {epoch} loss {loss:.4f} {time.strftime('%H:%M:%S')}", flush=True)
        history[-1]["loss"] = loss
    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    save(a, best_state, hub, dim, meta, n_ret, total, best, n_path, n_capped, coverage, history, best_ep)
    print("saved", os.path.join(a.persist, "ckpt.pt"), flush=True)

if __name__ == "__main__":
    main()
