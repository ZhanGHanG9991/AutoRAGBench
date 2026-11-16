#!/usr/bin/env python3
"""Rebuild MuSiQue vector stores with bge-m3.

Shard matrices are zeros. This reads the merged graph and text chunks and
writes vectors through LightRAG's own vector-store upsert. Entity text is
``name\\n description``. Relation text is ``keywords\\t src\\n tgt\\n description``
with sorted endpoints. Chunk ids are the keys already stored in text chunks.
"""
import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir("/tmp")

import numpy as np

from bge_client import embed_texts
from lightrag import LightRAG
from lightrag.operate import _truncate_vdb_content
from lightrag.utils import EmbeddingFunc, compute_mdhash_id
from nano_vectordb.dbs import buffer_string_to_array


def endpoint_order(src, tgt):
    src = str(src)
    tgt = str(tgt)
    if src > tgt:
        return tgt, src
    return src, tgt


def load_json(path: Path):
    return json.loads(path.read_text())


def vdb_count(path: Path) -> int:
    if not path.exists():
        return 0
    obj = json.loads(path.read_text())
    return len(obj.get("data") or [])


def _reopen_empty_vector_client(storage_obj, path: Path) -> None:
    """Drop rows loaded before this file was removed.

    NanoVectorDBStorage reads vdb_*.json in __post_init__. initialize() does
    not reopen it, so unlink then upsert writes the stale ids back out.
    """
    from nano_vectordb import NanoVectorDB

    storage_obj._client = NanoVectorDB(
        storage_obj.embedding_func.embedding_dim,
        storage_file=str(path),
    )
    storage_obj._pending_upserts.clear()
    storage_obj._pending_deletes.clear()
    storage_obj._unsaved_deletes.clear()
    storage_obj._unsaved_upserts.clear()
    storage_obj._client_dirty = False


def matrix_norm_stats(path: Path, expected: int) -> dict:
    """Row L2 stats. ok is false for NaN, all-zero, or a norm off the unit sphere.

    A matching row count is not enough. Index jobs write a placeholder matrix.
    """
    empty = {
        "n": 0,
        "nan_rows": -1,
        "zero_rows": -1,
        "min": None,
        "mean": None,
        "max": None,
        "ok": False,
    }
    if expected <= 0 or not path.is_file():
        return empty
    obj = json.loads(path.read_text())
    n = len(obj.get("data") or [])
    matrix = obj.get("matrix")
    if n != expected or not isinstance(matrix, str) or not matrix:
        empty["n"] = n
        return empty
    dim = int(obj.get("embedding_dim") or 1024)
    try:
        flat = np.asarray(buffer_string_to_array(matrix), dtype=np.float32).reshape(-1)
    except Exception:
        empty["n"] = n
        return empty
    if flat.size != expected * dim:
        empty["n"] = n
        return empty
    norms = np.linalg.norm(flat.reshape(expected, dim), axis=1)
    finite = np.isfinite(norms)
    nan_rows = int((~finite).sum())
    zero_rows = int(np.sum(finite & (norms < 1e-6)))
    stats = {
        "n": int(norms.size),
        "nan_rows": nan_rows,
        "zero_rows": zero_rows,
        "min": None,
        "mean": None,
        "max": None,
        "ok": False,
    }
    if finite.any():
        kept = norms[finite]
        stats["min"] = float(kept.min())
        stats["mean"] = float(kept.mean())
        stats["max"] = float(kept.max())
    stats["ok"] = (
        stats["n"] == expected
        and nan_rows == 0
        and zero_rows == 0
        and stats["min"] is not None
        and stats["min"] >= 0.99
        and stats["max"] <= 1.01
    )
    return stats


def print_norm(namespace: str, stats: dict) -> None:
    def fmt(value):
        return "na" if value is None else f"{value:.4f}"

    print(
        f"NORM {namespace} n={stats['n']} nan_rows={stats['nan_rows']} "
        f"zero_rows={stats['zero_rows']} min={fmt(stats['min'])} "
        f"mean={fmt(stats['mean'])} max={fmt(stats['max'])} ok={int(stats['ok'])}",
        flush=True,
    )


def matrix_is_unit(path: Path, expected: int) -> bool:
    return matrix_norm_stats(path, expected)["ok"]


def paper_storages(roots):
    found = []
    for root in roots:
        base = Path(root)
        if not base.is_dir():
            print("PAPER_ROOT_MISSING", flush=True)
            raise SystemExit(5)
        for child in sorted(base.iterdir()):
            if not child.is_dir():
                continue
            storage = child / "storage"
            graph = storage / "graph_chunk_entity_relation.graphml"
            if graph.is_file():
                found.append(storage)
    return found


def in_part(key: str, part: int, parts: int) -> bool:
    if parts <= 1:
        return True
    digest = hashlib.md5(str(key).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % parts == part


async def rebuild_storage(storage: Path, sock: str, batch_size: int, expect_chunks, guard=None, part=0, parts=1):
    calls = {"n": 0, "texts": 0, "norm": 0.0, "min": None, "max": None}

    async def embed(texts, **kwargs):
        if guard is not None and guard.level == "immediate":
            print("DISK_GUARD action=exit_immediate", flush=True)
            raise SystemExit(74)
        if guard is not None and guard.level == "warn":
            print("DISK_GUARD action=exit_after_batch", flush=True)
            raise SystemExit(75)
        rows = embed_texts(sock, list(texts), timeout=600)
        if rows.shape != (len(texts), 1024):
            raise RuntimeError(f"bad shape {rows.shape}")
        norms = np.linalg.norm(rows, axis=1)
        if (
            np.any(~np.isfinite(norms))
            or float(norms.min()) < 0.99
            or float(norms.max()) > 1.01
        ):
            raise RuntimeError("embedding norm left the unit sphere")
        batch_min = float(norms.min())
        batch_max = float(norms.max())
        calls["n"] += 1
        calls["texts"] += len(texts)
        calls["norm"] += float(norms.mean())
        calls["min"] = batch_min if calls["min"] is None else min(calls["min"], batch_min)
        calls["max"] = batch_max if calls["max"] is None else max(calls["max"], batch_max)
        if calls["n"] % 50 == 0:
            print(
                f"EMBED_PROGRESS batches={calls['n']} texts={calls['texts']}",
                flush=True,
            )
        return rows

    async def unused_llm(*_args, **_kwargs):
        raise RuntimeError("embed rebuild does not call the generator")

    rag = LightRAG(
        working_dir=str(storage),
        llm_model_func=unused_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=embed,
            model_name="bge-m3",
        ),
        embedding_batch_num=batch_size,
        embedding_func_max_async=1,
        tiktoken_model_name="gpt-4o-mini",
    )
    if parts > 1:
        async def _keep_shared_files():
            # Part dirs symlink the merged graph and chunk JSON. Those callbacks
            # rewrite the path they were opened from, which would replace the merge.
            return False

        rag.chunk_entity_relation_graph.index_done_callback = _keep_shared_files
        rag.text_chunks.index_done_callback = _keep_shared_files
    await rag.initialize_storages()
    if parts > 1:
        rag.chunk_entity_relation_graph.index_done_callback = _keep_shared_files
        rag.text_chunks.index_done_callback = _keep_shared_files
    cfg = {
        "embedding_token_limit": 8192,
        "tokenizer": rag.tokenizer,
    }

    graph = await rag.chunk_entity_relation_graph._get_graph()
    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    chunks = load_json(storage / "kv_store_text_chunks.json")
    print(
        f"EMBED_INPUT nodes={node_count} edges={edge_count} chunks={len(chunks)}",
        flush=True,
    )

    async def rebuild(namespace, storage_obj, expected, payload):
        path = storage / f"vdb_{namespace}.json"
        if expected == 0:
            print(f"VDB_EMPTY {namespace}", flush=True)
            return 0
        existing = vdb_count(path)
        if existing == expected and expected > 0:
            stats = matrix_norm_stats(path, expected)
            if stats["ok"]:
                print(f"VDB_KEEP {namespace} n={existing}", flush=True)
                print_norm(namespace, stats)
                return existing
            print(f"VDB_REBUILD {namespace} n={existing} reason=bad_norm", flush=True)
        if path.exists():
            path.unlink()
            _reopen_empty_vector_client(storage_obj, path)
            await storage_obj.initialize()
        await storage_obj.upsert(payload)
        await storage_obj.index_done_callback()
        got = vdb_count(path)
        print(f"VDB_WROTE {namespace} n={got} expected={expected}", flush=True)
        stats = matrix_norm_stats(path, expected)
        print_norm(namespace, stats)
        if got != expected or not stats["ok"]:
            raise SystemExit(6)
        return got

    entities = {}
    for name, data in graph.nodes(data=True):
        description = str(data.get("description") or "")
        content = _truncate_vdb_content(f"{name}\n{description}", cfg, "entity")
        entities[compute_mdhash_id(str(name), prefix="ent-")] = {
            "entity_name": str(name),
            "content": content,
            "source_id": str(data.get("source_id") or ""),
            "file_path": str(data.get("file_path") or ""),
        }
    relations = {}
    for src, tgt, data in graph.edges(data=True):
        left, right = endpoint_order(src, tgt)
        keywords = str(data.get("keywords") or "")
        description = str(data.get("description") or "")
        content = _truncate_vdb_content(
            f"{keywords}\t{left}\n{right}\n{description}",
            cfg,
            "relation",
        )
        relations[compute_mdhash_id(left + right, prefix="rel-")] = {
            "src_id": left,
            "tgt_id": right,
            "content": content,
            "source_id": str(data.get("source_id") or ""),
            "file_path": str(data.get("file_path") or ""),
        }
    chunk_rows = {}
    for chunk_id, row in chunks.items():
        if not isinstance(row, dict):
            continue
        content = _truncate_vdb_content(str(row.get("content") or ""), cfg, "chunk")
        chunk_rows[str(chunk_id)] = {
            "content": content,
            "full_doc_id": str(row.get("full_doc_id") or ""),
            "file_path": str(row.get("file_path") or ""),
        }

    if parts > 1:
        entities = {key: value for key, value in entities.items() if in_part(key, part, parts)}
        relations = {key: value for key, value in relations.items() if in_part(key, part, parts)}
        chunk_rows = {key: value for key, value in chunk_rows.items() if in_part(key, part, parts)}
    elif len(entities) != node_count or len(relations) != edge_count:
        raise SystemExit(6)
    if parts <= 1:
        if expect_chunks is None:
            if len(chunk_rows) <= 0:
                raise SystemExit(6)
        elif len(chunk_rows) != expect_chunks:
            raise SystemExit(6)

    await rebuild("entities", rag.entities_vdb, len(entities), entities)
    await rebuild("relationships", rag.relationships_vdb, len(relations), relations)
    await rebuild("chunks", rag.chunks_vdb, len(chunk_rows), chunk_rows)
    mean_norm = calls["norm"] / calls["n"] if calls["n"] else 0.0
    if calls["n"]:
        print(
            f"EMBED_DONE batches={calls['n']} texts={calls['texts']} "
            f"mean_norm={mean_norm:.4f} min_norm={calls['min']:.4f} "
            f"max_norm={calls['max']:.4f}",
            flush=True,
        )
        if not (0.99 <= mean_norm <= 1.01 and 0.99 <= calls["min"] and calls["max"] <= 1.01):
            raise SystemExit(6)
    else:
        print(
            "EMBED_DONE batches=0 texts=0 mean_norm=na min_norm=na max_norm=na",
            flush=True,
        )
    return {
        "batches": calls["n"],
        "texts": calls["texts"],
        "norm": calls["norm"],
        "kept": calls["n"] == 0,
    }


def prepare_storage(storage: Path, part_dir: str, parts: int) -> Path:
    if parts <= 1:
        return storage
    if not part_dir:
        raise SystemExit("EMBED_PART_DIR")
    dest = Path(part_dir)
    dest.mkdir(parents=True, exist_ok=True)
    for name in (
        "graph_chunk_entity_relation.graphml",
        "kv_store_text_chunks.json",
    ):
        src = storage / name
        link = dest / name
        if not src.is_file():
            raise SystemExit(5)
        if not link.exists():
            link.symlink_to(src)
    return dest


def join_parts(part_root: Path, storage: Path, parts: int):
    from merge_shards import endpoint_key, merge_vdb

    storage.mkdir(parents=True, exist_ok=True)
    specs = (
        ("entities", lambda item: item.get("entity_name")),
        (
            "relationships",
            lambda item: endpoint_key(item.get("src_id"), item.get("tgt_id")),
        ),
        ("chunks", lambda item: item.get("__id__")),
    )
    for namespace, key_fn in specs:
        paths = [part_root / f"part{i}" / f"vdb_{namespace}.json" for i in range(parts)]
        present = [path for path in paths if path.is_file()]
        if len(present) != parts:
            missing = parts - len(present)
            print(f"JOIN_MISSING namespace={namespace} missing={missing}", flush=True)
            raise SystemExit(5)
        out_path = storage / f"vdb_{namespace}.json"
        count = merge_vdb(paths, key_fn, out_path)
        stats = matrix_norm_stats(out_path, count)
        print_norm(namespace, stats)
        print(f"JOIN_WROTE {namespace} n={count}", flush=True)
        if count <= 0 or not stats["ok"]:
            raise SystemExit(6)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage", default="")
    parser.add_argument("--bge-sock", default="")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--parts", type=int, default=1)
    parser.add_argument("--part-dir", default="")
    parser.add_argument("--join-parts", type=int, default=0)
    parser.add_argument("--part-root", default="")
    parser.add_argument("--disk-guard", action="store_true")
    parser.add_argument(
        "--expect-chunks",
        type=int,
        default=29750,
        help="Required chunk rows for one storage. Ignored for --paper-root.",
    )
    parser.add_argument(
        "--paper-root",
        action="append",
        default=[],
        help="Directory of per-document working dirs. Repeat for each shard.",
    )
    args = parser.parse_args()
    if args.join_parts:
        join_parts(Path(args.part_root), Path(args.storage), args.join_parts)
        return
    if bool(args.storage) == bool(args.paper_root) or not args.bge_sock:
        raise SystemExit("EMBED_MODE")
    from disk_guard import DiskGuard

    guard = DiskGuard(args.disk_guard)
    guard.start()
    if args.storage:
        storage = prepare_storage(Path(args.storage), args.part_dir, args.parts)
        result = await rebuild_storage(
            storage,
            args.bge_sock,
            args.batch_size,
            args.expect_chunks,
            guard,
            args.part,
            args.parts,
        )
        if result["batches"] == 0 and not result["kept"]:
            raise SystemExit(6)
        if result["kept"]:
            print("EMBED_KEPT", flush=True)
        return
    storages = paper_storages(args.paper_root)
    print(f"PAPER_STORAGES n={len(storages)}", flush=True)
    if not storages:
        raise SystemExit(5)
    failed = 0
    batches = 0
    texts = 0
    norm = 0.0
    for index, storage in enumerate(storages, start=1):
        try:
            result = await rebuild_storage(storage, args.bge_sock, args.batch_size, None)
        except (SystemExit, Exception) as exc:
            failed += 1
            print(f"PAPER_EMBED_FAIL err={type(exc).__name__}", flush=True)
            continue
        batches += result["batches"]
        texts += result["texts"]
        norm += result["norm"]
        if index % 25 == 0 or index == len(storages):
            print(
                f"PAPER_EMBED n={index} total={len(storages)} failed={failed}",
                flush=True,
            )
    mean_norm = norm / batches if batches else 0.0
    print(
        f"PAPER_EMBED_DONE n={len(storages)} failed={failed} "
        f"batches={batches} texts={texts} mean_norm={mean_norm:.4f}",
        flush=True,
    )
    if failed:
        raise SystemExit(6)
    if batches and not (0.99 <= mean_norm <= 1.01):
        raise SystemExit(6)


if __name__ == "__main__":
    asyncio.run(main())
