#!/usr/bin/env python3
"""Merge LightRAG shard storages in one process.

Each shard is a finished ainsert workspace. Node and edge identity follow the
library: entity name, and an undirected endpoint pair. Descriptions, source
ids, and file paths that differ are joined with the library separator ``<SEP>``.
When a joined description has at least 8 fragments (the library default
``force_llm_summary_on_merge``), this process asks the same generator for one
summary. Fewer fragments stay joined, which is what a single LightRAG process
does before that threshold.

Vector rows are concatenated from the shard matrices and de-duplicated by
entity name, relation endpoints, or chunk id. This probe's vectors are zeros,
so the merge does not re-embed.
"""

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
import asyncio
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir("/tmp")

import networkx as nx
import numpy as np
from nano_vectordb.dbs import array_to_buffer_string, buffer_string_to_array

SEP = "<SEP>"
FORCE_SUMMARY = 8
# Qwen3.5-4B is started with max_model_len 32768. A completion reserve of
# max_tokens plus the chat wrapper has to fit in that window. Prompts that
# do not are split before the request; vLLM answers those with HTTP 400.
SUMMARY_MODEL_LEN = 32768
SUMMARY_TOKEN_MARGIN = 256
QWEN_TOKENIZER = _release_path("@MODEL_DIR@/Qwen3.5-4B/tokenizer.json")
GRAPH_NAME = "graph_chunk_entity_relation.graphml"
KV_UNION = (
    "kv_store_full_docs.json",
    "kv_store_text_chunks.json",
    "kv_store_doc_status.json",
    "kv_store_full_entities.json",
    "kv_store_full_relations.json",
)
KV_CHUNK_IDS = (
    "kv_store_entity_chunks.json",
    "kv_store_relation_chunks.json",
)


def split_sep(value):
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    return [part for part in text.split(SEP) if part]


def join_unique(left, right):
    seen = set()
    ordered = []
    for part in split_sep(left) + split_sep(right):
        if part not in seen:
            seen.add(part)
            ordered.append(part)
    return SEP.join(ordered)


def load_graph(storage: Path):
    path = storage / GRAPH_NAME
    graph = nx.read_graphml(path)
    if graph.is_directed():
        graph = nx.Graph(graph)
    return graph


def merge_graphs(graphs):
    merged = nx.Graph()
    for graph in graphs:
        for node, data in graph.nodes(data=True):
            if node not in merged:
                merged.add_node(node, **data)
                continue
            current = merged.nodes[node]
            current["description"] = join_unique(
                current.get("description"), data.get("description")
            )
            current["source_id"] = join_unique(
                current.get("source_id"), data.get("source_id")
            )
            current["file_path"] = join_unique(
                current.get("file_path"), data.get("file_path")
            )
            if not current.get("entity_type") and data.get("entity_type"):
                current["entity_type"] = data.get("entity_type")
            current["created_at"] = min(
                int(current.get("created_at") or 0),
                int(data.get("created_at") or 0),
            )
        for src, tgt, data in graph.edges(data=True):
            if merged.has_edge(src, tgt):
                current = merged.edges[src, tgt]
                current["description"] = join_unique(
                    current.get("description"), data.get("description")
                )
                current["keywords"] = join_unique(
                    current.get("keywords"), data.get("keywords")
                )
                current["source_id"] = join_unique(
                    current.get("source_id"), data.get("source_id")
                )
                current["file_path"] = join_unique(
                    current.get("file_path"), data.get("file_path")
                )
                current["weight"] = float(current.get("weight") or 0) + float(
                    data.get("weight") or 0
                )
            else:
                merged.add_edge(src, tgt, **data)
    return merged


def _summary_budget(max_tokens: int) -> int:
    room = SUMMARY_MODEL_LEN - max(0, int(max_tokens)) - SUMMARY_TOKEN_MARGIN
    return max(1024, room)


def _render_summary_chat(content: str) -> str:
    """Match the thinking-off Qwen chat wrapper used for this request."""
    return (
        "<|im_start|>user\n"
        + content
        + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def _summary_prompt(template, kind, name, parts):
    listing = "\n".join(
        json.dumps({"Description": part}, ensure_ascii=False) for part in parts
    )
    return template.format(
        description_type=kind,
        description_name=name,
        description_list=listing,
        summary_length=600,
        language="English",
    )


def _chat_token_count(tokenizer, content: str) -> int:
    return len(tokenizer.encode(_render_summary_chat(content)).ids)


def _pack_summary_parts(tokenizer, template, kind, name, parts, budget: int):
    """Split descriptions so each summary request fits the model window.

    Returns (groups, changed). changed is true when a fragment was cut or
    the list no longer fits in one request.
    """

    def fits(candidate):
        prompt = _summary_prompt(template, kind, name, candidate)
        if len(prompt) <= 40000:
            return True
        return _chat_token_count(tokenizer, prompt) <= budget

    def shrink(text):
        lo, hi = 1, len(text)
        best = text[:1]
        while lo <= hi:
            mid = (lo + hi) // 2
            if fits([text[:mid]]):
                best = text[:mid]
                lo = mid + 1
            else:
                hi = mid - 1
        return best

    groups = []
    current = []
    changed = False
    for part in parts:
        if not current:
            if fits([part]):
                current = [part]
            else:
                groups.append([shrink(part)])
                changed = True
            continue
        if fits(current + [part]):
            current.append(part)
        else:
            groups.append(current)
            changed = True
            if fits([part]):
                current = [part]
            else:
                groups.append([shrink(part)])
                current = []
    if current:
        groups.append(current)
    if len(groups) != 1:
        changed = True
    return groups, changed


def needs_summary(graph):
    subjects = []
    for node, data in graph.nodes(data=True):
        parts = split_sep(data.get("description"))
        if len(parts) >= FORCE_SUMMARY:
            subjects.append(("entity", node, parts))
    for src, tgt, data in graph.edges(data=True):
        parts = split_sep(data.get("description"))
        if len(parts) >= FORCE_SUMMARY:
            subjects.append(("relation", f"{src}<->{tgt}", parts))
    return subjects


async def summarize_long(graph, llm_url, max_tokens, concurrency, guard=None):
    subjects = needs_summary(graph)
    print(f"NEEDS_SUMMARY n={len(subjects)}", flush=True)
    if not subjects:
        return 0
    if not llm_url:
        raise SystemExit(
            f"{len(subjects)} descriptions have >= {FORCE_SUMMARY} fragments "
            "and need the generator; pass --llm"
        )
    from openai import AsyncOpenAI
    from lightrag.prompt import PROMPTS

    from tokenizers import Tokenizer

    if not os.path.isfile(QWEN_TOKENIZER):
        raise SystemExit("QWEN_TOKENIZER_MISSING")
    tokenizer = Tokenizer.from_file(QWEN_TOKENIZER)
    budget = _summary_budget(max_tokens)
    client = AsyncOpenAI(
        base_url=llm_url,
        api_key=os.environ["LIGHTRAG_LLM_API_KEY"],
        timeout=600,
        max_retries=0,
    )
    template = PROMPTS["summarize_entity_descriptions"]
    sem = asyncio.Semaphore(max(1, concurrency))
    fitted = 0

    async def one(kind, name, parts):
        prompt = _summary_prompt(template, kind, name, parts)
        request = {
            "model": "generator",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
        if max_tokens > 0:
            request["max_tokens"] = max_tokens
        if guard is not None and guard.level:
            return "", "disk"
        last_error = ""
        for attempt in range(3):
            try:
                async with sem:
                    resp = await client.chat.completions.create(**request)
                return (resp.choices[0].message.content or "").strip(), ""
            except Exception as exc:
                last_error = type(exc).__name__
                await asyncio.sleep(1 + attempt)
        return "", last_error

    async def run_subject(kind, name, parts, depth=0):
        nonlocal fitted
        groups, changed = _pack_summary_parts(
            tokenizer, template, kind, name, parts, budget
        )
        if changed and depth == 0:
            fitted += 1
        if len(groups) == 1:
            return await one(kind, name, groups[0])
        partials = []
        for group in groups:
            text, err = await one(kind, name, group)
            if err:
                return "", err
            if text:
                partials.append(text)
        if len(partials) <= 1:
            return (partials[0] if partials else ""), ""
        if depth >= 3:
            groups, _changed = _pack_summary_parts(
                tokenizer, template, kind, name, partials, budget
            )
            return await one(kind, name, groups[0])
        return await run_subject(kind, name, partials, depth + 1)

    try:
        results = await asyncio.gather(
            *(run_subject(kind, name, parts) for kind, name, parts in subjects)
        )
    finally:
        await client.close()

    errors = Counter()
    empty = 0
    done = 0
    for (kind, name, _parts), (text, err) in zip(subjects, results):
        if err:
            errors[err] += 1
            continue
        if not text:
            empty += 1
            continue
        if kind == "entity":
            graph.nodes[name]["description"] = text
        else:
            src, tgt = name.split("<->", 1)
            graph.edges[src, tgt]["description"] = text
        done += 1
        if done % 50 == 0:
            print(f"SUMMARY_PROGRESS n={done}", flush=True)
    print(
        f"SUMMARY_FIT n={fitted} budget={budget}",
        flush=True,
    )
    print(
        f"SUMMARY_DONE n={done} empty={empty} "
        f"errors={json.dumps(dict(errors), sort_keys=True)}",
        flush=True,
    )
    if errors:
        raise SystemExit(1)
    return done


def _content_duplicate_ids(doc_keys, status):
    """Ids LightRAG refused because the same text is already in this shard.

    A content-hash duplicate is stored as a dup- row, not as its own document.
    Count it toward the corpus total when the original row is processed.
    """
    covered = set()
    for key, row in status.items():
        if not isinstance(row, dict) or not str(key).startswith(("dup-", "pad-")):
            continue
        meta = row.get("metadata") or {}
        if not isinstance(meta, dict) or meta.get("duplicate_kind") != "content_hash":
            continue
        src = str(row.get("file_path") or "")
        orig = str(meta.get("original_doc_id") or "")
        orig_row = status.get(orig)
        if (
            src
            and src not in doc_keys
            and src not in covered
            and orig
            and orig != src
            and orig in doc_keys
            and isinstance(orig_row, dict)
            and str(orig_row.get("status")) == "processed"
        ):
            covered.add(src)
    return covered


def load_doc_map(path):
    """Count repeated top-level doc_id keys before JSON collapses them."""
    dup_keys = 0

    def pairs_hook(pairs):
        nonlocal dup_keys
        data = dict(pairs)
        # The root object is decoded last; nested metadata keys do not count.
        dup_keys = len(pairs) - len(data)
        return data

    data = json.loads(path.read_text(), object_pairs_hook=pairs_hook)
    return data, dup_keys


def check_doc_uniqueness(storage, covered_ids):
    """Check shard input lists against full_docs plus content-hash duplicates."""
    reports = []
    for name in ("ids.txt", "doc_ids.json"):
        path = storage.parent / name
        if not path.is_file():
            continue
        if name == "ids.txt":
            ids = [line.strip() for line in path.read_text().splitlines()
                   if line.strip() and not line.strip().startswith("#")]
        else:
            ids = json.loads(path.read_text())
        unique = set(ids)
        reports.append({
            "path": str(path),
            "count": len(ids),
            "unique": len(unique),
            "duplicates": len(ids) - len(unique),
            "missing_from_storage": len(unique - covered_ids),
            "unexpected_in_storage": len(covered_ids - unique),
        })
    return reports


def check_full_docs(storages):
    sets = []
    dup_keys = []
    status_dup_keys = []
    input_docs = []
    for storage in storages:
        data, duplicates = load_doc_map(storage / "kv_store_full_docs.json")
        dup_keys.append(duplicates)
        keys = set(data)
        status, duplicates = load_doc_map(storage / "kv_store_doc_status.json")
        status_dup_keys.append(duplicates)
        keys |= _content_duplicate_ids(keys, status)
        sets.append(keys)
        if len(storages) == 1:
            input_docs.extend(check_doc_uniqueness(storage, keys))
    # One shard has no cross-shard overlap. set.intersection(only) returns
    # that set, because the first argument is taken as self.
    if len(sets) < 2:
        inter = set()
    else:
        inter = set.intersection(*sets)
    union = set.union(*sets) if sets else set()
    return {
        "sizes": [len(item) for item in sets],
        "overlap": len(inter),
        "union": len(union),
        "dup_keys": dup_keys,
        "status_dup_keys": status_dup_keys,
        "input_docs": input_docs,
    }


def check_doc_status(storages):
    reals = []
    not_processed = 0
    dup = 0
    for storage in storages:
        data = json.loads((storage / "kv_store_doc_status.json").read_text())
        real = 0
        for key, row in data.items():
            if str(key).startswith(("dup-", "pad-")):
                dup += 1
                continue
            real += 1
            status = str(row.get("status")) if isinstance(row, dict) else ""
            if status != "processed":
                not_processed += 1
        docs = json.loads((storage / "kv_store_full_docs.json").read_text())
        real += len(_content_duplicate_ids(set(docs), data))
        reals.append(real)
    return {"real": reals, "not_processed": not_processed, "dup": dup}


def graph_overlap(graphs):
    def pack(parts):
        if len(parts) < 2:
            inter = set()
        else:
            inter = set.intersection(*parts)
        union = set.union(*parts) if parts else set()
        return {
            "sizes": [len(part) for part in parts],
            "intersection": len(inter),
            "union": len(union),
            "jaccard": (len(inter) / len(union)) if union else 1.0,
            "intersection_over_each": [
                (len(inter) / len(part)) if part else 1.0 for part in parts
            ],
        }

    node_sets = [set(graph.nodes) for graph in graphs]
    edge_sets = [
        {tuple(sorted((str(src), str(tgt)))) for src, tgt in graph.edges}
        for graph in graphs
    ]
    return {"nodes": pack(node_sets), "edges": pack(edge_sets)}


def merge_object_maps(paths):
    merged = {}
    collisions = 0
    for path in paths:
        if not path.exists():
            continue
        incoming = json.loads(path.read_text())
        for key, value in incoming.items():
            if key not in merged:
                merged[key] = value
            elif merged[key] != value:
                collisions += 1
    return merged, collisions


def merge_chunk_maps(paths):
    merged = {}
    for path in paths:
        if not path.exists():
            continue
        incoming = json.loads(path.read_text())
        for key, value in incoming.items():
            if key not in merged:
                merged[key] = value
                continue
            current = merged[key]
            ids = []
            seen = set()
            for item in (current.get("chunk_ids") or []) + (value.get("chunk_ids") or []):
                if item not in seen:
                    seen.add(item)
                    ids.append(item)
            current["chunk_ids"] = ids
            current["count"] = len(ids)
    return merged


def merge_vdb(paths, key_fn, out_path):
    kept = {}
    order = []
    rows = []
    for path in paths:
        if not path.exists():
            continue
        obj = json.loads(path.read_text())
        matrix = np.asarray(buffer_string_to_array(obj["matrix"]), dtype=np.float32)
        if matrix.ndim == 1:
            if matrix.size % 1024 != 0:
                raise SystemExit(
                    f"{path} flat matrix size {matrix.size} is not divisible by 1024"
                )
            matrix = matrix.reshape(-1, 1024)
        data = obj["data"]
        if len(matrix) != len(data):
            raise SystemExit(f"{path} matrix rows {len(matrix)} != data {len(data)}")
        for row, item in zip(matrix, data):
            key = key_fn(item)
            if key in kept:
                continue
            kept[key] = True
            order.append(item)
            rows.append(row)
    if not order:
        return 0
    stacked = np.vstack(rows) if rows else np.zeros((0, 1024), dtype=np.float32)
    payload = {
        "embedding_dim": 1024,
        "data": order,
        "matrix": array_to_buffer_string(stacked.astype(np.float32)),
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False))
    return len(order)


def endpoint_key(src, tgt):
    return tuple(sorted((str(src), str(tgt))))


def compare(reference: Path, merged: nx.Graph):
    if reference.is_dir():
        ref = load_graph(reference)
    else:
        ref = nx.read_graphml(reference)
        if ref.is_directed():
            ref = nx.Graph(ref)
    ref_nodes = set(ref.nodes)
    mer_nodes = set(merged.nodes)
    ref_edges = {endpoint_key(u, v) for u, v in ref.edges}
    mer_edges = {endpoint_key(u, v) for u, v in merged.edges}
    node_union = ref_nodes | mer_nodes
    edge_union = ref_edges | mer_edges
    node_jaccard = (len(ref_nodes & mer_nodes) / len(node_union)) if node_union else 1
    edge_jaccard = (len(ref_edges & mer_edges) / len(edge_union)) if edge_union else 1
    node_rel = abs(len(mer_nodes) - len(ref_nodes)) / len(ref_nodes) if ref_nodes else 0
    edge_rel = abs(len(mer_edges) - len(ref_edges)) / len(ref_edges) if ref_edges else 0
    return {
        "reference_nodes": len(ref_nodes),
        "reference_edges": len(ref_edges),
        "merged_nodes": len(mer_nodes),
        "merged_edges": len(mer_edges),
        "node_jaccard": node_jaccard,
        "edge_jaccard": edge_jaccard,
        "node_only_reference": len(ref_nodes - mer_nodes),
        "node_only_merged": len(mer_nodes - ref_nodes),
        "edge_only_reference": len(ref_edges - mer_edges),
        "edge_only_merged": len(mer_edges - ref_edges),
        "close": node_rel <= 0.05
        and edge_rel <= 0.05
        and node_jaccard >= 0.90
        and edge_jaccard >= 0.90,
    }


def self_check(storage: Path):
    graph = load_graph(storage)
    nodes = list(graph.nodes)
    left_nodes = set(nodes[::2])
    left = graph.subgraph(left_nodes).copy()
    right = graph.subgraph(set(nodes) - left_nodes).copy()
    # Edges that crossed the cut are missing from both subgraphs. Put each
    # cross edge on the side of its first endpoint so the union still has it.
    for src, tgt, data in graph.edges(data=True):
        if left.has_edge(src, tgt) or right.has_edge(src, tgt):
            continue
        if src in left_nodes:
            left.add_edge(src, tgt, **data)
            if tgt not in left:
                left.add_node(tgt, **graph.nodes[tgt])
        else:
            right.add_edge(src, tgt, **data)
            if src not in right:
                right.add_node(src, **graph.nodes[src])
    merged = merge_graphs([left, right])
    report = {
        "self_check_nodes": graph.number_of_nodes(),
        "self_check_merged_nodes": merged.number_of_nodes(),
        "self_check_edges": graph.number_of_edges(),
        "self_check_merged_edges": merged.number_of_edges(),
        "self_check_ok": merged.number_of_nodes() == graph.number_of_nodes()
        and merged.number_of_edges() == graph.number_of_edges(),
    }
    print(json.dumps(report), flush=True)
    if not report["self_check_ok"]:
        raise SystemExit(1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--reference-graph")
    parser.add_argument("--llm", default="")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Cap for merge summaries. 0 leaves them uncapped.")
    parser.add_argument("--concurrency", type=int, default=48)
    parser.add_argument(
        "--skip-vdb",
        action="store_true",
        help="Do not concatenate shard vector matrices. Production shards store zeros.",
    )
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument(
        "--expect-docs",
        type=int,
        default=0,
        help="Require this many docs across shards. 0 keeps the MuSiQue 29750 check.",
    )
    parser.add_argument(
        "--disk-guard",
        action="store_true",
        help="Every 10 min, stop before writing if free space is under 20 GiB or 2 GiB.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    from disk_guard import DiskGuard

    guard = DiskGuard(getattr(args, "disk_guard", False))
    guard.start()
    shard_dirs = [Path(p) for p in args.shards]
    if args.self_check:
        self_check(shard_dirs[0] / "storage" if (shard_dirs[0] / "storage").is_dir() else shard_dirs[0])
        return
    storages = []
    for shard in shard_dirs:
        storage = shard / "storage" if (shard / "storage").is_dir() else shard
        storages.append(storage)
    doc_report = check_full_docs(storages)
    print("MERGE_DOCS", json.dumps(doc_report), flush=True)
    expect_docs = args.expect_docs if args.expect_docs else 29750
    if len(storages) == 1 and (
        any(doc_report["dup_keys"])
        or any(doc_report["status_dup_keys"])
        or any(row["duplicates"] or row["missing_from_storage"]
               or row["unexpected_in_storage"] for row in doc_report["input_docs"])
    ):
        print("MERGE_DOC_UNIQUENESS_FAILED", json.dumps(doc_report), flush=True)
        raise SystemExit(6)
    if doc_report["overlap"] != 0 or doc_report["union"] != expect_docs:
        raise SystemExit(6)
    status_report = check_doc_status(storages)
    print("MERGE_STATUS", json.dumps(status_report), flush=True)
    if args.expect_docs:
        real_ok = sum(status_report["real"]) == args.expect_docs
    else:
        real_ok = status_report["real"] == [14875, 14875]
    if status_report["not_processed"] != 0 or not real_ok:
        raise SystemExit(6)
    graphs = []
    for storage in storages:
        graph = load_graph(storage)
        print(
            f"MERGE_LOAD nodes={graph.number_of_nodes()} edges={graph.number_of_edges()}",
            flush=True,
        )
        graphs.append(graph)
    overlap = graph_overlap(graphs)
    print("MERGE_OVERLAP", json.dumps(overlap), flush=True)
    merged = merge_graphs(graphs)
    node_sizes = overlap["nodes"]["sizes"]
    edge_sizes = overlap["edges"]["sizes"]
    if not (max(node_sizes) <= merged.number_of_nodes() <= sum(node_sizes)):
        raise SystemExit(6)
    if not (max(edge_sizes) <= merged.number_of_edges() <= sum(edge_sizes)):
        raise SystemExit(6)
    if guard.level == "immediate":
        raise SystemExit(74)
    if guard.level == "warn":
        raise SystemExit(75)
    summaries = asyncio.run(
        summarize_long(merged, args.llm, args.max_tokens, args.concurrency, guard)
    )
    if guard.level == "immediate":
        raise SystemExit(74)
    if guard.level == "warn":
        raise SystemExit(75)
    out = Path(args.out)
    storage_out = out / "storage"
    storage_out.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(merged, storage_out / GRAPH_NAME)

    collisions = {}
    for name in KV_UNION:
        payload, bad = merge_object_maps([storage / name for storage in storages])
        if name == "kv_store_doc_status.json":
            payload = {
                key: value
                for key, value in payload.items()
                if not str(key).startswith(("dup-", "pad-"))
            }
            leftover = sum(
                1
                for value in payload.values()
                if not isinstance(value, dict) or str(value.get("status")) != "processed"
            )
            if leftover:
                raise SystemExit(f"MERGED_STATUS_NOT_PROCESSED n={leftover}")
        (storage_out / name).write_text(json.dumps(payload, ensure_ascii=False))
        collisions[name] = bad
    for name in KV_CHUNK_IDS:
        payload = merge_chunk_maps([storage / name for storage in storages])
        (storage_out / name).write_text(json.dumps(payload, ensure_ascii=False))

    if args.skip_vdb:
        vdb_counts = {
            "entities": None,
            "relationships": None,
            "chunks": None,
            "skipped": True,
        }
        print("VDB_SKIPPED reason=shard-matrices-are-zeros", flush=True)
    else:
        vdb_counts = {
            "entities": merge_vdb(
                [storage / "vdb_entities.json" for storage in storages],
                lambda item: item.get("entity_name"),
                storage_out / "vdb_entities.json",
            ),
            "relationships": merge_vdb(
                [storage / "vdb_relationships.json" for storage in storages],
                lambda item: endpoint_key(item.get("src_id"), item.get("tgt_id")),
                storage_out / "vdb_relationships.json",
            ),
            "chunks": merge_vdb(
                [storage / "vdb_chunks.json" for storage in storages],
                lambda item: item.get("__id__"),
                storage_out / "vdb_chunks.json",
            ),
            "skipped": False,
        }

    doc_sets = []
    for shard in shard_dirs:
        doc_path = shard / "doc_ids.json"
        if doc_path.exists():
            doc_sets.append(set(json.loads(doc_path.read_text())))
    doc_overlap = 0
    if len(doc_sets) == 2:
        doc_overlap = len(doc_sets[0] & doc_sets[1])

    report = {
        "shards": [str(path) for path in shard_dirs],
        "merged_nodes": merged.number_of_nodes(),
        "merged_edges": merged.number_of_edges(),
        "llm_summaries": summaries,
        "overlap": overlap,
        "docs": doc_report,
        "c1024_nodes_per_doc": 2460 / 300,
        "c1024_edges_per_doc": 1891 / 300,
        "merged_nodes_per_doc": merged.number_of_nodes() / expect_docs,
        "merged_edges_per_doc": merged.number_of_edges() / expect_docs,
        "kv_collisions": collisions,
        "vdb_rows": vdb_counts,
        "doc_id_overlap": doc_overlap,
        "doc_id_union": len(set().union(*doc_sets)) if doc_sets else None,
    }
    if args.reference_graph:
        report.update(compare(Path(args.reference_graph), merged))
    (out / "compare.json").write_text(json.dumps(report, indent=2) + "\n")
    print("MERGE", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
