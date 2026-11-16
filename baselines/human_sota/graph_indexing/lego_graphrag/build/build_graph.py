"""Step 2 of the text adaptation: OpenIE triples -> the knowledge graph LEGO-GraphRAG retrieves from.

- Entity names are normalised (case, whitespace, outer punctuation); each node shows its most frequent surface form.
- Every OpenIE triple (s, r, o) becomes the edge s -r-> o plus the inverse edge o -"r (inverse)"-> s, so that
  LEGO's directed modules (PPR, shortest paths, beam search; all mode="out") can walk from either end, as they
  can on Freebase, where most relations have a reverse relation.
- Synonym edges ("synonym of", both directions) link node names whose all-MiniLM-L6-v2 cosine is >= --syn
  (HippoRAG-style; OpenIE from text spells the same entity in several ways, Freebase does not).
- Provenance: every edge keeps the passages it was extracted from (triple_pids.pkl), used to attach source
  passages to reasoning paths at generation time.

  python build_graph.py --index <index_dir> --emb <all-MiniLM-L6-v2 dir>
Writes graph.pkl (igraph), node_emb.npy, triple_pids.pkl, doc_nodes.pkl, graph_stats.json into <index_dir>.
"""
import argparse
import collections
import json
import os
import pickle
import re

import igraph as ig
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

INV = " (inverse)"
SYN = "synonym of"


def key(s):
    s = re.sub(r"\s+", " ", s.replace("->", "-")).strip().strip(".,;:!?\"'`()[]{} ").lower()
    return s


def clean(s):
    return re.sub(r"\s+", " ", s.replace("->", "-")).strip().strip(".,;:!?\"'` ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--syn", type=float, default=0.9)
    ap.add_argument("--syn-k", type=int, default=10)
    ap.add_argument("--max-len", type=int, default=100, help="drop entity strings longer than this (characters)")
    a = ap.parse_args()

    pid_doc = {}
    for line in open(os.path.join(a.index, "passages.jsonl")):
        p = json.loads(line)
        pid_doc[p["pid"]] = p["doc_id"]

    surface = collections.defaultdict(collections.Counter)  # key -> surface forms
    trip = collections.defaultdict(set)                      # (skey, rel, okey) -> pids
    node_pids = collections.defaultdict(set)
    n_raw = n_fail = 0
    seen = set()
    for line in open(os.path.join(a.index, "openie.jsonl")):
        r = json.loads(line)
        if "error" in r or r["pid"] in seen:
            n_fail += "error" in r
            continue
        seen.add(r["pid"])
        for s, rel, o in r["triples"]:
            n_raw += 1
            ks, ko, rel = key(s), key(o), clean(rel).lower()
            if not ks or not ko or not rel or ks == ko or len(ks) > a.max_len or len(ko) > a.max_len:
                continue
            surface[ks][clean(s)] += 1
            surface[ko][clean(o)] += 1
            trip[(ks, rel, ko)].add(r["pid"])
            node_pids[ks].add(r["pid"])
            node_pids[ko].add(r["pid"])

    keys = sorted(surface)
    kidx = {k: i for i, k in enumerate(keys)}
    names = [surface[k].most_common(1)[0][0] for k in keys]
    # names must be unique vertex ids for LEGO's G.vs.find(name=...); equal keys already merged, but two keys can
    # still share a surface form after cleaning -> disambiguate.
    cnt = collections.Counter(names)
    names = [n if cnt[n] == 1 else f"{n} [{i}]" for i, n in enumerate(names)]

    edges, rels, triple_pids = [], [], {}
    for (ks, rel, ko), pids in trip.items():
        s, o = kidx[ks], kidx[ko]
        edges += [(s, o), (o, s)]
        rels += [rel, rel + INV]
        triple_pids[(names[s], rel, names[o])] = sorted(pids)
        triple_pids[(names[o], rel + INV, names[s])] = sorted(pids)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(a.emb, device=dev)
    emb = model.encode(names, batch_size=1024, normalize_embeddings=True, convert_to_numpy=True,
                       show_progress_bar=False).astype(np.float32)
    E = torch.from_numpy(emb).to(dev)
    n_syn = 0
    for st in range(0, len(names), 1024):
        sims = E[st:st + 1024] @ E.T
        val, idx = sims.topk(min(a.syn_k + 1, len(names)), dim=1)
        for i, (vs, js) in enumerate(zip(val.tolist(), idx.tolist())):
            for v, j in zip(vs, js):
                if v < a.syn or j == st + i or j < st + i:
                    continue
                edges += [(st + i, j), (j, st + i)]
                rels += [SYN, SYN]
                n_syn += 1

    G = ig.Graph(n=len(names), edges=edges, directed=True)
    G.vs["name"] = names
    G.es["name"] = rels
    G.write_pickle(os.path.join(a.index, "graph.pkl"))
    np.save(os.path.join(a.index, "node_emb.npy"), emb.astype(np.float16))
    pickle.dump(triple_pids, open(os.path.join(a.index, "triple_pids.pkl"), "wb"))
    doc_nodes = collections.defaultdict(set)
    for k, pids in node_pids.items():
        for p in pids:
            doc_nodes[pid_doc[p]].add(kidx[k])
    pickle.dump({d: sorted(v) for d, v in doc_nodes.items()}, open(os.path.join(a.index, "doc_nodes.pkl"), "wb"))
    stats = {"passages": len(pid_doc), "passages_with_openie": len(seen), "openie_failures": n_fail,
             "raw_triples": n_raw, "unique_triples": len(trip), "nodes": len(names), "synonym_pairs": n_syn,
             "edges": G.ecount(), "isolated_nodes": sum(1 for d in G.degree() if d == 0)}
    json.dump(stats, open(os.path.join(a.index, "graph_stats.json"), "w"), indent=1)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
