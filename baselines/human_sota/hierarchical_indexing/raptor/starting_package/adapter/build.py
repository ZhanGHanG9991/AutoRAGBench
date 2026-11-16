"""Build document-scoped RAPTOR trees with upstream semantic clustering.

The scheduling loop batches embeddings and propagates every future exception.
It preserves upstream stopping, chunking, soft GMM clustering and summary graph.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import time

import httpx
import numpy as np
import umap
from raptor import cluster_utils as cu
from raptor.tree_structures import Node, Tree
from raptor.utils import split_text, get_text
from runtime import SBERT, TOKENIZER, generate


def seeded_global(embeddings, dim, n_neighbors=None, metric="cosine"):
    neighbors = n_neighbors or max(2, int((len(embeddings) - 1) ** 0.5))
    return umap.UMAP(n_neighbors=neighbors, n_components=dim, metric=metric,
                     random_state=224, n_jobs=1).fit_transform(embeddings)


def seeded_local(embeddings, dim, num_neighbors=10, metric="cosine"):
    return umap.UMAP(n_neighbors=num_neighbors, n_components=dim, metric=metric,
                     random_state=224, n_jobs=1).fit_transform(embeddings)


cu.global_cluster_embeddings = seeded_global
cu.local_cluster_embeddings = seeded_local

# Upstream locates local members by broadcasting [cluster,N,768] equality.
# Integer position propagation is equivalent for distinct vectors and also avoids
# accidentally merging distinct nodes whose embeddings happen to be identical.
def position_clustering(embeddings, dim, threshold, verbose=False):
    # Recursive local clustering can produce an atomic two-node subcluster.
    # In that case len(embeddings)-2 is zero, which UMAP rejects.  Give each
    # remaining node its own cluster: the upstream recursion then reaches its
    # documented singleton base case instead of reclustering the same pair.
    if len(embeddings) == 0:
        return []
    if len(embeddings) <= 2:
        return [np.array([i], dtype=int) for i in range(len(embeddings))]
    reduced = seeded_global(embeddings, min(dim, len(embeddings)-2))
    global_labels, n_global = cu.GMM_cluster(reduced, threshold)
    result = [np.array([], dtype=int) for _ in embeddings]
    offset = 0
    for label in range(n_global):
        members = np.array([i for i, labels in enumerate(global_labels) if label in labels])
        if not len(members):
            continue
        if len(members) <= dim + 1:
            local_labels, n_local = [np.array([0]) for _ in members], 1
        else:
            local_labels, n_local = cu.GMM_cluster(seeded_local(embeddings[members], dim), threshold)
        for i, labels in zip(members, local_labels):
            result[i] = np.append(result[i], labels + offset)
        offset += n_local
    return result

cu.perform_clustering = position_clustering


def chunk_documents(docs, config):
    leaves = []
    for doc in docs:
        text = doc.get("title", "") + "\n" + doc["text"]
        # KG relations/entity identifiers contain dots; preserve whole TSV facts.
        if doc.get("kg"):
            chunks, pending = [], []
            for line in doc["text"].splitlines():
                if pending and len(TOKENIZER.encode("\n".join(pending+[line]))) > config["chunk_tokens"]:
                    chunks.append("\n".join(pending)); pending=[]
                pending.append(line)
            if pending: chunks.append("\n".join(pending))
        else:
            chunks = split_text(text, TOKENIZER, config["chunk_tokens"])
        for chunk in chunks:
            if chunk.strip():
                leaves.append((chunk, str(doc["doc_id"])))
    return leaves


def build_tree(docs, embedding, config, log):
    return build_chunks(chunk_documents(docs, config), embedding, config, log)


def build_chunks(leaves, embedding, config, log):
    if not leaves:
        raise ValueError("Empty corpus document")
    vectors = embedding.encode([x[0] for x in leaves])
    nodes = {}
    for i, ((text, doc_id), vector) in enumerate(zip(leaves, vectors)):
        node = Node(text, i, set(), {"SBERT": vector})
        node.doc_ids = [doc_id]
        nodes[i] = node
    layers = {0: list(nodes.values())}
    current = list(nodes.values())
    leaf_nodes = dict(nodes)
    return abstract_nodes(nodes, leaf_nodes, layers, current, embedding, config, log)


def abstract_nodes(nodes, leaf_nodes, layers, current, embedding, config, log):
    for layer in range(config["max_layers"]):
        if len(current) <= config["reduction_dimension"] + 1:
            break
        clusters = cu.RAPTOR_Clustering.perform_clustering(
            current, "SBERT", max_length_in_cluster=config["cluster_max_tokens"],
            tokenizer=TOKENIZER, reduction_dimension=config["reduction_dimension"],
            threshold=config["cluster_threshold"])
        covered = {n.index for cluster in clusters for n in cluster}
        if covered != {n.index for n in current}:
            raise ValueError("Soft clustering omitted nodes; refusing incomplete tree")
        # Soft overlap can yield as many clusters as input nodes. Like upstream,
        # permit this; max_layers bounds the recursion without changing memberships.

        def summarize(cluster):
            text = get_text(cluster)

            def call_summary(payload):
                """Generate one summary with the existing bounded transient retry policy."""
                prompt = "Write a summary of the following, including as many key details as possible: " + payload + ":"
                for attempt in range(6):
                    try:
                        return generate([
                            {"role": "system", "content": "You are a Summarizing Text Portal"},
                            {"role": "user", "content": prompt}],
                            config["summary_tokens"])
                    except httpx.HTTPStatusError as exc:
                        if exc.response.status_code < 500 and exc.response.status_code != 429:
                            raise
                        if attempt == 5:
                            raise
                        delay = min(2 ** (attempt + 1), 30)
                        print("RAPTOR_SUMMARY_RETRY", exc.response.status_code,
                              attempt + 1, "delay", delay, flush=True)
                        time.sleep(delay)
                    except httpx.TransportError as exc:
                        if attempt == 5:
                            raise
                        delay = min(2 ** (attempt + 1), 30)
                        print("RAPTOR_SUMMARY_RETRY", type(exc).__name__,
                              attempt + 1, "delay", delay, flush=True)
                        time.sleep(delay)
                raise AssertionError("unreachable")

            # Qwen3.5-4B is served with a 32,768-token context, while an
            # upstream RAPTOR soft cluster can occasionally be much larger in
            # the Qwen tokenizer even when its cl100k estimate is acceptable.
            # Opt-in map/reduce preserves the cluster and tree topology: it
            # summarizes deterministic contiguous segments, then summarizes
            # those partial summaries into the single original parent node.
            max_chars = int(os.environ.get("RAPTOR_SUMMARY_MAX_CHARS", "0"))
            if max_chars > 0 and len(text) > max_chars:
                parts = [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
                partials = []
                usages = []
                for part in parts:
                    summary, usage = call_summary(part)
                    partials.append(summary)
                    usages.append(usage)
                final, usage = call_summary("\n\n".join(partials))
                usages.append(usage)
                combined = {}
                for item in usages:
                    for key, value in item.items():
                        if isinstance(value, (int, float)):
                            combined[key] = combined.get(key, 0) + value
                print("RAPTOR_SUMMARY_MAP_REDUCE", "chars", len(text),
                      "parts", len(parts), "max_chars", max_chars, flush=True)
                return final, combined

            return call_summary(text)

        # executor.map retains deterministic cluster ordering.  Some shared
        # nodes can temporarily exhaust their per-user thread allowance after
        # long runs; reduce scheduling concurrency without changing inputs,
        # seeds, cluster membership, or the persisted canonical config.
        workers = max(1, min(config["summary_concurrency"], len(clusters)))
        while workers > 1:
            try:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    summaries = list(pool.map(summarize, clusters))
                break
            except RuntimeError as exc:
                if "can't start new thread" not in str(exc):
                    raise
                workers = max(1, workers // 2)
                print("RAPTOR_THREAD_RETRY workers", workers, flush=True)
                time.sleep(5)
        else:
            summaries = [summarize(cluster) for cluster in clusters]
        vectors = embedding.encode([x[0] for x in summaries])
        parents = []
        for cluster, (text, usage), vector in zip(clusters, summaries, vectors):
            parent = Node(text, len(nodes), {n.index for n in cluster}, {"SBERT": vector})
            parent.doc_ids = sorted({d for n in cluster for d in n.doc_ids})[:128]
            nodes[parent.index] = parent
            parents.append(parent)
            log.write(json.dumps({"node": parent.index, "layer": layer + 1,
                                  "doc_ids": parent.doc_ids, "children": sorted(parent.children),
                                  "usage": usage, "summary_sha256": hashlib.sha256(text.encode()).hexdigest()}) + "\n")
            log.flush()
        layers[layer + 1] = parents
        current = parents
    return Tree(nodes, {n.index: n for n in current}, leaf_nodes, max(layers), layers)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--models", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--questions", help="Public train/dev only, smoke scope")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    random.seed(config["seed"]); np.random.seed(config["seed"])
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if (out / "COMPLETE").exists():
        raise RuntimeError("Refusing to overwrite completed index")
    start = time.time()
    wanted = None
    if args.questions:
        questions = [json.loads(x) for x in Path(args.questions).read_text().splitlines()][:args.limit or 3]
        wanted = {str(q["meta"]["doc_id"]) for q in questions}
    docs = []
    corpus_digest = hashlib.sha256()
    with open(args.corpus, 'rb') as f:
        for line in f:
            corpus_digest.update(line)
            doc = json.loads(line)
            if wanted is None or str(doc["doc_id"]) in wanted:
                docs.append(doc)
                if wanted is None and args.limit and len(docs) >= args.limit:
                    break
    if wanted is not None:
        assert {str(d["doc_id"]) for d in docs} == wanted
    if not args.questions and not args.limit:
        manifest = json.loads((Path(args.corpus).parent.parent / 'data_manifest.json').read_text())
        assert corpus_digest.hexdigest() == manifest['corpus_sha256']
        assert len(docs) == manifest['n_corpus']
    embedding = SBERT(Path(args.models) / "multi-qa-mpnet-base-cos-v1", "cuda")
    catalog = {"task": args.task, "trees": {}, "scope": "smoke" if args.questions or args.limit else "full"}
    counts = {}
    per_tree = {}
    groups = [([d], str(d["doc_id"])) for d in docs] if args.task == "qasper" else [(docs, "corpus")]
    with (out / "summaries.jsonl").open("a") as log:
        for group, key in groups:
            file = hashlib.sha256(key.encode()).hexdigest() + ".pkl"
            metadata={'config':config,'input_sha256':hashlib.sha256(json.dumps(group,sort_keys=True).encode()).hexdigest()}
            meta_file=out/(file+'.meta.json')
            if (out/file).exists() and meta_file.exists():
                assert json.loads(meta_file.read_text())==metadata
                with (out/file).open('rb') as f:tree=pickle.load(f)
            else:
                tree = build_tree(group, embedding, config, log)
                with (out / (file+'.tmp')).open("wb") as f:
                    pickle.dump(tree, f, protocol=5)
                (out/(file+'.tmp')).replace(out/file)
                meta_file.write_text(json.dumps(metadata))
            with (out / file).open("rb") as f:
                restored = pickle.load(f)
            assert len(restored.all_nodes) == len(tree.all_nodes)
            assert all(n.children <= restored.all_nodes.keys() for n in restored.all_nodes.values())
            catalog["trees"][key] = file
            per_tree[key] = {str(k): len(v) for k, v in tree.layer_to_nodes.items()}
            for k, v in per_tree[key].items():
                counts[k] = counts.get(k, 0) + v
            print("TREE_BUILT", key, per_tree[key], flush=True)
            (out / "progress.json").write_text(json.dumps({"completed_trees": len(per_tree),
                "total_trees": len(groups), "seconds": time.time()-start, "layer_nodes": counts}))
    assert sum(v for k, v in counts.items() if int(k) > 0) > 0, "Not RAPTOR: no summary nodes"
    stats = {"task": args.task, "scope": catalog["scope"], "documents": len(docs),
             "layer_nodes": counts, "trees": per_tree, "seconds": time.time() - start,
             "corpus_sha256": corpus_digest.hexdigest(),
             "max_summary_depth": max(map(int, counts)),
             "trees_without_summaries": sum(not any(int(k)>0 for k in layers) for layers in per_tree.values()),
             "embedding_parameters": embedding.parameter_count, "config": config,
             "job_id": os.environ.get("SLURM_JOB_ID"), "serialization_verified": True}
    (out / "catalog.json").write_text(json.dumps(catalog, indent=2))
    (out / "index_stats.json").write_text(json.dumps(stats, indent=2))
    (out / "COMPLETE").write_text("Built and reloaded; evaluation still required.\n")


if __name__ == "__main__":
    main()
