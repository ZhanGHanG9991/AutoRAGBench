"""Candidate triples on AutoRAGBench's merged Freebase graph.

Official SubgraphRAG scores every triple in a per-question RoG subgraph.
Here each task is one merged graph. A question's candidates are the directed
triples among entities reached by a 2-hop walk from the topic entities.
CVT nodes (names starting with ``m.`` or ``g.``) are always expanded. Other
non-topic nodes whose degree exceeds ``hub`` stay in the ball, but the walk
does not go through them.
"""
from __future__ import annotations

from collections import deque

import numpy as np

HOP = 2
MAX_TRIPLES = 100_000
HUB_CANDIDATES = (2000, 5000, 10000)


def is_cvt_name(name: str) -> bool:
    return name.startswith("m.") or name.startswith("g.")


class KG:
    def __init__(self, triples_path: str):
        ent2id: dict[str, int] = {}
        rel2id: dict[str, int] = {}

        def eid(s: str) -> int:
            i = ent2id.get(s)
            if i is None:
                i = len(ent2id)
                ent2id[s] = i
            return i

        def rid(s: str) -> int:
            i = rel2id.get(s)
            if i is None:
                i = len(rel2id)
                rel2id[s] = i
            return i

        heads, rels, tails = [], [], []
        with open(triples_path) as f:
            for line in f:
                h, r, t = line.rstrip("\n").split("\t")
                heads.append(eid(h))
                rels.append(rid(r))
                tails.append(eid(t))
        self.heads = np.asarray(heads, np.int32)
        self.rels = np.asarray(rels, np.int32)
        self.tails = np.asarray(tails, np.int32)
        n_ent = len(ent2id)
        self.names = [""] * n_ent
        for name, i in ent2id.items():
            self.names[i] = name
        self.rel_names = [""] * len(rel2id)
        for name, i in rel2id.items():
            self.rel_names[i] = name
        self.ent2id = ent2id
        self.is_cvt = np.fromiter((is_cvt_name(n) for n in self.names), dtype=np.bool_, count=n_ent)
        self.deg = np.bincount(self.heads, minlength=n_ent).astype(np.int32)
        self.deg += np.bincount(self.tails, minlength=n_ent).astype(np.int32)
        out: list[list[int]] = [[] for _ in range(n_ent)]
        und_sets: list[set[int]] = [set() for _ in range(n_ent)]
        for i, (h, t) in enumerate(zip(self.heads.tolist(), self.tails.tolist())):
            out[h].append(i)
            if h != t:
                und_sets[h].add(t)
                und_sets[t].add(h)
        self.out = out
        self.und = [list(s) for s in und_sets]
        text_ids = [i for i, flag in enumerate(self.is_cvt.tolist()) if not flag]
        text_ids.sort(key=lambda i: self.names[i])
        self.text_ids = text_ids
        self.text_row = np.full(n_ent, -1, np.int32)
        for row, gid in enumerate(text_ids):
            self.text_row[gid] = row
        rel_order = list(range(len(self.rel_names)))
        rel_order.sort(key=lambda i: self.rel_names[i])
        self.rel_row = np.empty(len(rel_order), np.int32)
        for row, gid in enumerate(rel_order):
            self.rel_row[gid] = row

    def extract(self, topic_names, hub: int, hop: int = HOP, max_triples: int = MAX_TRIPLES):
        topics = []
        seen_topic = set()
        for name in topic_names:
            i = self.ent2id.get(name)
            if i is None or i in seen_topic:
                continue
            seen_topic.add(i)
            topics.append(i)
        if not topics:
            return None
        dist = {s: 0 for s in topics}
        q = deque(topics)
        while q:
            u = q.popleft()
            d = dist[u]
            if d >= hop:
                continue
            if u not in seen_topic and (not bool(self.is_cvt[u])) and int(self.deg[u]) > hub:
                continue
            nd = d + 1
            for v in self.und[u]:
                if v not in dist:
                    dist[v] = nd
                    q.append(v)
        items = []
        for u, du in dist.items():
            for ti in self.out[u]:
                v = int(self.tails[ti])
                dv = dist.get(v)
                if dv is None:
                    continue
                items.append((du if du > dv else dv, ti))
        items.sort()
        capped = len(items) > max_triples
        if capped:
            items = items[:max_triples]
        items.sort(key=lambda x: x[1])
        triples = [(int(self.heads[ti]), int(self.rels[ti]), int(self.tails[ti])) for _, ti in items]
        return {"triples": triples, "capped": capped, "topics": topics}


def localize(kg: KG, extracted: dict, answer_names=None):
    triples = extracted["triples"]
    topics = extracted["topics"]
    ents = set(topics)
    rels = set()
    for h, r, t in triples:
        ents.add(h)
        ents.add(t)
        rels.add(r)
    text, non = [], []
    for e in sorted(ents, key=lambda i: kg.names[i]):
        (non if kg.is_cvt[e] else text).append(e)
    ordered = text + non
    local_ent = {e: i for i, e in enumerate(ordered)}
    rel_list = sorted(rels, key=lambda i: kg.rel_names[i])
    local_rel = {r: i for i, r in enumerate(rel_list)}
    h_ids, r_ids, t_ids = [], [], []
    for h, r, t in triples:
        h_ids.append(local_ent[h])
        r_ids.append(local_rel[r])
        t_ids.append(local_ent[t])
    n_ent = len(ordered)
    topic_oh = np.zeros((n_ent, 2), np.float32)
    topic_oh[:, 0] = 1.0
    topic_local = []
    for g in topics:
        li = local_ent[g]
        topic_local.append(li)
        topic_oh[li, 0] = 0.0
        topic_oh[li, 1] = 1.0
    ans_local = []
    if answer_names:
        for name in answer_names:
            gid = kg.ent2id.get(name)
            if gid is not None and gid in local_ent:
                ans_local.append(local_ent[gid])
    labels, has_path = (None, False)
    if answer_names is not None:
        labels, has_path = label_shortest(h_ids, t_ids, n_ent, topic_local, ans_local)
    return {
        "h": np.asarray(h_ids, np.int64),
        "r": np.asarray(r_ids, np.int64),
        "t": np.asarray(t_ids, np.int64),
        "text_rows": np.asarray([int(kg.text_row[g]) for g in text], np.int64),
        "rel_rows": np.asarray([int(kg.rel_row[g]) for g in rel_list], np.int64),
        "n_non_text": len(non),
        "topic_oh": topic_oh,
        "topic_local": topic_local,
        "ans_local": ans_local,
        "labels": labels,
        "has_path": has_path,
        "n_cand": len(triples),
        "capped": bool(extracted["capped"]),
        "global_triples": triples,
    }


def label_shortest(h_ids, t_ids, n_ent, topics, answers):
    """Triples on a directed shortest path, matching the official NetworkX rule.

    Both directions are searched. If both exist, only the shorter direction is
    kept. Parallel edges collapse to the last triple id, as DiGraph.add_edge does.
    """
    n = len(h_ids)
    labels = np.zeros(n, np.float32)
    if n == 0 or not topics or not answers:
        return labels, False
    pair = {}
    for i, (h, t) in enumerate(zip(h_ids, t_ids)):
        pair[(int(h), int(t))] = i
    adj = [[] for _ in range(n_ent)]
    radj = [[] for _ in range(n_ent)]
    for (h, t), ti in pair.items():
        adj[h].append((t, ti))
        radj[t].append(h)

    def bfs(nbrs, src):
        dist = {src: 0}
        q = deque([src])
        while q:
            u = q.popleft()
            for item in nbrs[u]:
                v = item[0] if isinstance(item, tuple) else item
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    fwd_cache, rev_cache = {}, {}

    def dist_from(src):
        if src not in fwd_cache:
            fwd_cache[src] = bfs(adj, src)
        return fwd_cache[src]

    def dist_to(dst):
        if dst not in rev_cache:
            rev_cache[dst] = bfs(radj, dst)
        return rev_cache[dst]

    has = False
    for qid in topics:
        from_q = dist_from(qid)
        for aid in answers:
            if qid == aid:
                continue
            from_a = dist_from(aid)
            lf = from_q.get(aid)
            lb = from_a.get(qid)
            options = [x for x in (lf, lb) if x is not None]
            if not options:
                continue
            has = True
            best = min(options)
            if lf is not None and lf == best:
                to_a = dist_to(aid)
                for u, nbrs in enumerate(adj):
                    du = from_q.get(u)
                    if du is None or du >= lf:
                        continue
                    for v, ti in nbrs:
                        dv = to_a.get(v)
                        if dv is not None and du + 1 + dv == lf:
                            labels[ti] = 1.0
            if lb is not None and lb == best:
                to_q = dist_to(qid)
                for u, nbrs in enumerate(adj):
                    du = from_a.get(u)
                    if du is None or du >= lb:
                        continue
                    for v, ti in nbrs:
                        dv = to_q.get(v)
                        if dv is not None and du + 1 + dv == lb:
                            labels[ti] = 1.0
    return labels, has
