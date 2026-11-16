"""Parallel, deterministic RAPTOR forest construction for formal-v1 corpora.

Tree workers receive disjoint global block IDs.  They retain the original
corpus-order, 1,024-leaf partitioning and write the same checkpoint payload as
build_collection.py.  A separate assembly phase runs only after every shard
has verified the full public corpus and completed its assigned blocks.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sqlite3
import time

import numpy as np

from build import build_chunks, chunk_documents
from build_collection import make_index
from runtime import SBERT


def corpus_manifest(corpus):
    path = Path(corpus)
    return json.loads((path.parent.parent / "data_manifest.json").read_text())


def assigned_count(total, shard_id, shard_count):
    if shard_id >= total:
        return 0
    return 1 + (total - 1 - shard_id) // shard_count


def build_shard(args, config):
    if not 0 <= args.shard_id < args.shard_count:
        raise ValueError("shard-id must be in [0, shard-count)")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "COMPLETE").exists():
        raise RuntimeError("Completed index must not be overwritten")
    spool = out / "trees"
    spool.mkdir(exist_ok=True)
    marker_dir = out / "shard_progress"
    marker_dir.mkdir(exist_ok=True)

    # Concurrency changes scheduling only.  Persisted checkpoints retain the
    # canonical experiment config so existing sequential checkpoints reload.
    runtime_config = dict(config)
    runtime_config["summary_concurrency"] = int(
        os.environ.get("RAPTOR_SUMMARY_CONCURRENCY", config["summary_concurrency"])
    )
    if runtime_config["summary_concurrency"] < 1:
        raise ValueError("RAPTOR_SUMMARY_CONCURRENCY must be positive")
    embedding = SBERT(Path(args.models) / "multi-qa-mpnet-base-cos-v1", "cuda")
    start = time.time()
    blocks = documents = leaves = built = reused = 0
    limit_reached = False
    pending = []
    digest = hashlib.sha256()
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    progress_path = marker_dir / (
        f"progress.shard-{args.shard_id:03d}-of-{args.shard_count:03d}.json"
    )

    def flush(items):
        nonlocal blocks, leaves, built, reused, limit_reached
        block_id = blocks
        if block_id % args.shard_count == args.shard_id:
            dest = spool / f"{block_id:07d}.pkl"
            item_hash = hashlib.sha256(
                json.dumps(items, ensure_ascii=False).encode()
            ).hexdigest()
            if dest.exists():
                with dest.open("rb") as handle:
                    saved = pickle.load(handle)
                assert saved["input_sha256"] == item_hash
                assert saved["config"] == config
                reused += 1
            else:
                calls = spool / f"{block_id:07d}.calls.jsonl"
                with calls.open("w") as log:
                    tree = build_chunks(items, embedding, runtime_config, log)
                saved = {"input_sha256": item_hash, "config": config, "tree": tree}
                temp = spool / f".{block_id:07d}.{job_id}.tmp"
                with temp.open("wb") as handle:
                    pickle.dump(saved, handle, protocol=5)
                os.replace(temp, dest)
                built += 1
            progress_path.write_text(json.dumps({
                "task": args.task,
                "stage": "trees",
                "shard_id": args.shard_id,
                "shard_count": args.shard_count,
                "documents_seen": documents,
                "global_blocks_seen": block_id + 1,
                "assigned_blocks_done": built + reused,
                "new_blocks": built,
                "reused_blocks": reused,
                "global_leaves_seen": leaves + len(items),
                "seconds_current_job": time.time() - start,
                "job_id": job_id,
                "summary_concurrency_runtime": runtime_config["summary_concurrency"],
            }))
            print(
                "RAPTOR_SHARD_BLOCK",
                block_id,
                "shard",
                f"{args.shard_id}/{args.shard_count}",
                "new",
                built,
                "reused",
                reused,
                "seconds",
                round(time.time() - start, 1),
                flush=True,
            )
            if args.max_new_blocks and built >= args.max_new_blocks:
                limit_reached = True
        blocks += 1
        leaves += len(items)

    completed_scan = True
    with open(args.corpus, "rb") as handle:
        for line in handle:
            digest.update(line)
            doc = json.loads(line)
            documents += 1
            if args.task in ("webqsp", "cwq"):
                doc["kg"] = True
            pending.extend(chunk_documents([doc], config))
            while len(pending) >= config["collection_leaf_block_size"]:
                size = config["collection_leaf_block_size"]
                flush(pending[:size])
                pending = pending[size:]
                if limit_reached:
                    completed_scan = False
                    break
            if limit_reached:
                break
    if not completed_scan:
        print("RAPTOR_SHARD_PARTIAL", json.dumps({
            "task": args.task,
            "shard_id": args.shard_id,
            "shard_count": args.shard_count,
            "new_blocks": built,
            "reused_blocks": reused,
            "global_blocks_seen": blocks,
            "documents_seen": documents,
            "reason": "bounded worker recycle before LLVM/JIT accumulation",
        }), flush=True)
        return
    if pending:
        flush(pending)

    manifest = corpus_manifest(args.corpus)
    assert digest.hexdigest() == manifest["corpus_sha256"]
    assert documents == manifest["n_corpus"]
    expected = assigned_count(blocks, args.shard_id, args.shard_count)
    assert built + reused == expected, (built, reused, expected)
    marker = {
        "task": args.task,
        "stage": "trees_complete",
        "shard_id": args.shard_id,
        "shard_count": args.shard_count,
        "documents": documents,
        "total_blocks": blocks,
        "total_leaves": leaves,
        "assigned_blocks": expected,
        "new_blocks": built,
        "reused_blocks": reused,
        "corpus_sha256": digest.hexdigest(),
        "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
        "summary_concurrency_runtime": runtime_config["summary_concurrency"],
        "embedding_parameters": embedding.parameter_count,
        "seconds_current_job": time.time() - start,
        "job_id": job_id,
    }
    marker_path = marker_dir / (
        f"done.shard-{args.shard_id:03d}-of-{args.shard_count:03d}.json"
    )
    marker_path.write_text(json.dumps(marker, indent=2))
    print("RAPTOR_SHARD_COMPLETE", json.dumps(marker), flush=True)


def validate_markers(out, task, shard_count, config_path):
    marker_dir = out / "shard_progress"
    markers = []
    for shard_id in range(shard_count):
        path = marker_dir / f"done.shard-{shard_id:03d}-of-{shard_count:03d}.json"
        if not path.exists():
            raise RuntimeError(f"Missing completed shard marker: {path}")
        markers.append(json.loads(path.read_text()))
    first = markers[0]
    config_sha = hashlib.sha256(Path(config_path).read_bytes()).hexdigest()
    for shard_id, marker in enumerate(markers):
        assert marker["task"] == task
        assert marker["shard_id"] == shard_id
        assert marker["shard_count"] == shard_count
        for key in ("documents", "total_blocks", "total_leaves", "corpus_sha256"):
            assert marker[key] == first[key], (key, marker[key], first[key])
        assert marker["config_sha256"] == config_sha
    assert sum(x["assigned_blocks"] for x in markers) == first["total_blocks"]
    return markers, first


def validate_kg(corpus, task):
    if task not in ("webqsp", "cwq"):
        return None
    corpus = Path(corpus)
    triples = corpus.with_name("triples.tsv")
    digest = hashlib.sha256()
    edges = 0
    with triples.open("rb") as handle:
        for line in handle:
            digest.update(line)
            assert len(line.rstrip(b"\n").split(b"\t")) == 3
            edges += 1
    corpus_edges = 0
    with corpus.open() as handle:
        for line in handle:
            corpus_edges += len(json.loads(line)["text"].splitlines())
    # Some official KG corpora contain rendered entity values with embedded
    # newlines.  The corpus is still the immutable, manifest-hashed retrieval
    # source, but splitlines() then exceeds the physical triples.tsv row count.
    # Keep strict equality by default and permit only an explicitly declared,
    # exactly matched compatibility delta for a diagnosed dataset version.
    expected_delta = int(os.environ.get("RAPTOR_EXPECTED_KG_LINE_DELTA", "0"))
    actual_delta = corpus_edges - edges
    assert actual_delta == expected_delta, (
        corpus_edges, edges, actual_delta, expected_delta
    )
    return {
        "triples": edges,
        "triples_sha256": digest.hexdigest(),
        "entity_text_lines": corpus_edges,
        "entity_text_line_delta": actual_delta,
        "expected_entity_text_line_delta": expected_delta,
    }


def assemble(args, config):
    out = Path(args.out)
    if (out / "COMPLETE").exists():
        raise RuntimeError("Completed index must not be overwritten")
    markers, totals = validate_markers(out, args.task, args.shard_count, args.config)
    spool = out / "trees"
    expected_names = [f"{block_id:07d}.pkl" for block_id in range(totals["total_blocks"])]
    actual_names = sorted(path.name for path in spool.glob("*.pkl"))
    assert actual_names == expected_names, (
        len(actual_names), len(expected_names),
        sorted(set(expected_names) - set(actual_names))[:10],
        sorted(set(actual_names) - set(expected_names))[:10],
    )
    manifest = corpus_manifest(args.corpus)
    assert manifest["n_corpus"] == totals["documents"]
    assert manifest["corpus_sha256"] == totals["corpus_sha256"]
    kg_info = validate_kg(args.corpus, args.task)

    start = time.time()
    dbpath = out / "nodes.sqlite"
    if dbpath.exists():
        dbpath.unlink()
    db = sqlite3.connect(dbpath)
    db.execute(
        "CREATE TABLE nodes(id INTEGER PRIMARY KEY,text TEXT,layer INTEGER,children TEXT,doc_ids TEXT)"
    )
    roots = []
    counts = {}
    offset = 0
    with (out / "vectors.f32").open("wb") as vector_file:
        for position, name in enumerate(expected_names):
            with (spool / name).open("rb") as handle:
                tree = pickle.load(handle)["tree"]
            mapping = {
                node.index: layer
                for layer, nodes in tree.layer_to_nodes.items()
                for node in nodes
            }
            records = []
            vectors = []
            for index, node in sorted(tree.all_nodes.items()):
                assert index == len(vectors)
                records.append((
                    offset + index,
                    node.text,
                    mapping[index],
                    json.dumps(sorted(offset + child for child in node.children)),
                    json.dumps(node.doc_ids),
                ))
                vectors.append(node.embeddings["SBERT"])
                level = str(mapping[index])
                counts[level] = counts.get(level, 0) + 1
            db.executemany("INSERT INTO nodes VALUES(?,?,?,?,?)", records)
            db.commit()
            np.asarray(vectors, dtype="float32").tofile(vector_file)
            roots.extend(offset + index for index in tree.root_nodes)
            offset += len(tree.all_nodes)
            if (position + 1) % 100 == 0:
                print("RAPTOR_ASSEMBLE_BLOCK", position + 1, "nodes", offset, flush=True)
    db.close()
    assert sum(value for key, value in counts.items() if int(key) > 0) > 0
    np.save(out / "roots.npy", np.asarray(roots, dtype="int64"))
    search = make_index(out, config)
    stats = {
        "task": args.task,
        "scope": "full",
        "documents": totals["documents"],
        "leaves": totals["total_leaves"],
        "trees": totals["total_blocks"],
        "layer_nodes": counts,
        "max_summary_depth": max(map(int, counts)),
        "corpus_sha256": totals["corpus_sha256"],
        "kg": kg_info,
        "config": config,
        "search": search,
        "seconds_current_job": time.time() - start,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "embedding_parameters": markers[0]["embedding_parameters"],
        "serialization_verified": True,
        "parallel_build": {
            "shard_count": args.shard_count,
            "tree_job_ids": [marker["job_id"] for marker in markers],
            "summary_concurrency_runtime": sorted(
                {marker["summary_concurrency_runtime"] for marker in markers}
            ),
        },
    }
    (out / "index_stats.json").write_text(json.dumps(stats, indent=2))
    (out / "catalog.json").write_text(json.dumps({
        "task": args.task,
        "scope": "full",
        "storage": "sqlite_faiss",
        "nodes": offset,
    }))
    (out / "COMPLETE").write_text(
        "Full corpus coverage, recursive summaries, deterministic shard merge and search reload checked.\n"
    )
    print("RAPTOR_ASSEMBLY_COMPLETE", json.dumps(stats), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("trees", "assemble"), required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--shard-id", type=int)
    parser.add_argument("--max-new-blocks", type=int, default=0)
    args = parser.parse_args()
    if args.shard_count < 1:
        raise ValueError("shard-count must be positive")
    if args.max_new_blocks < 0:
        raise ValueError("max-new-blocks must be non-negative")
    config = json.loads(Path(args.config).read_text())
    if args.phase == "trees":
        if args.shard_id is None:
            raise ValueError("trees phase requires --shard-id")
        build_shard(args, config)
    else:
        if args.shard_id is not None:
            raise ValueError("assemble phase does not accept --shard-id")
        assemble(args, config)


if __name__ == "__main__":
    main()
