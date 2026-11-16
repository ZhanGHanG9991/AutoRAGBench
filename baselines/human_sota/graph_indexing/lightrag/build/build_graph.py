#!/usr/bin/env python3
"""One LightRAG insert of a fixed MuSiQue sample, with optional sharding.

Embeddings are zeros so the clock is LLM extraction and graph merge.
LLM cache is off. max_tokens is applied only when --max-tokens is positive.
"""
import argparse
import asyncio
import hashlib
import json
import os
import re
import signal
import time
import traceback
from copy import copy
from datetime import datetime, timezone
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir("/tmp")

import networkx as nx
import numpy as np
from openai import AsyncOpenAI

from sample_corpus import load_sample, take_shard
from disk_guard import DiskGuard
from lightrag import LightRAG
from lightrag.chunker.token_size import chunking_by_token_size
from lightrag.utils import EmbeddingFunc, TiktokenTokenizer

CHUNK = 1200
OVERLAP = 100
SAFE_DOC_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def doc_status_report(storage: Path):
    path = storage / "kv_store_doc_status.json"
    if not path.exists():
        return {"counts": {}, "processed": 0, "processed_sha": ""}
    data = json.loads(path.read_text())
    counts = {}
    processed = []
    for doc_id, row in data.items():
        if not isinstance(row, dict):
            continue
        status = str(row.get("status"))
        counts[status] = counts.get(status, 0) + 1
        if status == "processed":
            processed.append(doc_id)
    processed.sort()
    digest = hashlib.sha256("\n".join(processed).encode()).hexdigest()[:16]
    return {
        "counts": counts,
        "processed": len(processed),
        "processed_sha": digest,
    }


# Statuses the pipeline will pick up on its own. FAILED is not in that set;
# reopen_real_failures turns real failures into pending before this scan.
_RESUME_STATUSES = {"pending", "processing", "parsing", "analyzing"}


def status_counts(storage: Path):
    """Count doc_status rows. None when the file does not exist yet."""
    path = storage / "kv_store_doc_status.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    counts = {}
    actionable = 0
    for doc_id, row in data.items():
        if not isinstance(row, dict) or str(doc_id).startswith(("dup-", "pad-")):
            continue
        status = str(row.get("status"))
        counts[status] = counts.get(status, 0) + 1
        if status in _RESUME_STATUSES:
            actionable += 1
    counts["actionable"] = actionable
    return counts


def reopen_real_failures(storage: Path) -> int:
    """Put non-duplicate FAILED rows back to pending so the next insert retries them.

    Duplicate-attempt rows (``dup-*``) stay failed. LightRAG does not auto-resume
    FAILED. A crash mid-write is avoided by replacing the file atomically.
    """
    path = storage / "kv_store_doc_status.json"
    if not path.exists():
        return 0
    data = json.loads(path.read_text())
    now = datetime.now(timezone.utc).isoformat()
    changed = 0
    for doc_id, row in data.items():
        if not isinstance(row, dict) or str(doc_id).startswith(("dup-", "pad-")):
            continue
        if str(row.get("status")) != "failed":
            continue
        row["status"] = "pending"
        row["error_msg"] = None
        row["updated_at"] = now
        changed += 1
    if not changed:
        return 0
    tmp = path.with_suffix(".json.reopen")
    tmp.write_text(json.dumps(data, ensure_ascii=False))
    tmp.replace(path)
    print(f"REOPEN_FAILED n={changed}", flush=True)
    return changed


def install_batched_flush(rag, every: int, on_flush=None):
    """Flush graph, vector stores, and doc_status together every N documents.

    Extraction and merging stay on the library path. doc_status upserts update
    memory only until the batch commit. The commit writes the graph and the
    vector stores first, then doc_status once, so a crash cannot leave a
    PROCESSED row whose graph rows were never flushed. Uncommitted documents
    stay at the previous on-disk status (pending, for a document that had not
    reached a commit) and the next run redoes that batch.
    """
    if every <= 1:
        async def drain_noop():
            return None

        drain_noop.status_flushes = lambda: 0
        return drain_noop

    from lightrag.base import DocStatus
    from lightrag.utils_pipeline import doc_status_transition_metadata

    original_done = rag._insert_done
    original_transition = rag._upsert_doc_status_transition
    doc_status = rag.doc_status
    original_status_flush = doc_status.index_done_callback
    state = {"held": [], "flushing": False, "status_flushes": 0}
    lock = asyncio.Lock()

    async def status_flush_noop():
        return None

    # Parsing, analyzing, and processing all upsert doc_status. Keep those
    # updates in memory so a 25MB file is not rewritten several times per doc.
    doc_status.index_done_callback = status_flush_noop

    def snapshot(kwargs):
        copied = dict(kwargs)
        status_doc = copied.get("status_doc")
        if status_doc is not None:
            cloned = copy(status_doc)
            metadata = getattr(status_doc, "metadata", None)
            if isinstance(metadata, dict):
                cloned.metadata = dict(metadata)
            copied["status_doc"] = cloned
        extra_fields = copied.get("extra_fields")
        if isinstance(extra_fields, dict):
            extra_fields = dict(extra_fields)
            chunks_list = extra_fields.get("chunks_list")
            if isinstance(chunks_list, list):
                extra_fields["chunks_list"] = list(chunks_list)
            copied["extra_fields"] = extra_fields
        if isinstance(copied.get("metadata_extra"), dict):
            copied["metadata_extra"] = dict(copied["metadata_extra"])
        return copied

    async def write_processed(kwargs):
        """Persist one PROCESSED row in memory. Disk follows in one batch write.

        This does not go through ``_upsert_doc_status_transition``: a drain
        after SIGTERM runs once the pipeline has released its busy token, and
        that helper then discards the write as a stale owner.
        """
        status_doc = kwargs.get("status_doc")
        drop = kwargs.get("metadata_drop") or ()
        payload = {
            "status": kwargs.get("status"),
            "content_summary": getattr(status_doc, "content_summary", None),
            "content_length": getattr(status_doc, "content_length", None),
            "created_at": getattr(status_doc, "created_at", None),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "file_path": kwargs.get("file_path"),
            "track_id": getattr(status_doc, "track_id", None),
            "content_hash": getattr(status_doc, "content_hash", None),
            "metadata": doc_status_transition_metadata(
                status_doc,
                extra=kwargs.get("metadata_extra"),
                drop=tuple(drop),
            ),
        }
        extra = kwargs.get("extra_fields")
        if isinstance(extra, dict):
            payload.update(extra)
        await doc_status.upsert({kwargs["doc_id"]: payload})

    async def flush_held():
        async with lock:
            if state["flushing"] or not state["held"]:
                return 0
            held = state["held"]
            state["held"] = []
            state["flushing"] = True
        try:
            # Graph and vector stores first. PROCESSED hits disk only after this.
            await original_done()
            for kwargs in held:
                await write_processed(kwargs)
            await original_status_flush()
            state["status_flushes"] += 1
            print(
                f"FLUSH_BATCH docs={len(held)} "
                f"doc_status_flushes={state['status_flushes']}",
                flush=True,
            )
            print(f"DOC_STATUS_FLUSH n={state['status_flushes']}", flush=True)
            held = []
            if on_flush is not None:
                on_flush()
            return state["status_flushes"]
        finally:
            async with lock:
                if held:
                    state["held"] = held + state["held"]
                state["flushing"] = False

    async def insert_done(*args, **kwargs):
        return None

    async def transition(*args, **kwargs):
        status = kwargs.get("status")
        if status == DocStatus.PROCESSED:
            async with lock:
                state["held"].append(snapshot(kwargs))
                ready = len(state["held"]) >= every and not state["flushing"]
            if ready:
                await flush_held()
            return None
        # Non-processed transitions still update memory. The no-op callback
        # above is what skips the per-document rewrite.
        return await original_transition(*args, **kwargs)

    async def drain():
        """Commit every held document, including rows that arrived mid-flush.

        A flush already running inside the insert task is allowed to finish
        before this starts another one. The real doc_status callback is
        restored even when a flush raises, so a later finalize can still
        persist memory.
        """
        print("DRAIN_BEGIN", flush=True)
        try:
            for _ in range(4):
                while True:
                    async with lock:
                        busy = state["flushing"]
                    if not busy:
                        break
                    await asyncio.sleep(0.05)
                await flush_held()
                async with lock:
                    more = bool(state["held"]) or state["flushing"]
                if not more:
                    break
        finally:
            doc_status.index_done_callback = original_status_flush
            print(
                f"DRAIN_END doc_status_flushes={state['status_flushes']}",
                flush=True,
            )

    rag._insert_done = insert_done
    rag._upsert_doc_status_transition = transition
    drain.status_flushes = lambda: state["status_flushes"]
    return drain


def filter_id_file(sample, id_file: str):
    """Keep the requested ids, sorted. Missing ids abort. Ids are not printed."""
    wanted = []
    seen = set()
    for line in Path(id_file).read_text().splitlines():
        doc_id = line.strip()
        if not doc_id or doc_id.startswith("#"):
            continue
        if doc_id in seen:
            continue
        seen.add(doc_id)
        wanted.append(doc_id)
    by_id = {}
    for doc_id, content in sample:
        by_id.setdefault(doc_id, content)
    missing = sum(1 for doc_id in wanted if doc_id not in by_id)
    if missing or not wanted:
        print(f"ID_FILE_MISSING n={missing} wanted={len(wanted)}", flush=True)
        raise SystemExit(6)
    wanted.sort()
    return [(doc_id, by_id[doc_id]) for doc_id in wanted]


async def _isolate_paper_storage() -> None:
    """Load the next paper from its own files.

    LightRAG keeps storage in a process-wide namespace and reads each file
    only on the first init. Later papers would otherwise resume against the
    previous paper's memory and leave their own pending rows untouched.
    """
    from lightrag.kg import shared_storage as shared

    if not shared._initialized or shared._shared_dicts is None or shared._init_flags is None:
        return
    async with shared.get_internal_lock():
        for key in list(shared._shared_dicts.keys()):
            bucket = shared._shared_dicts.get(key)
            if hasattr(bucket, "clear"):
                bucket.clear()
        shared._init_flags.clear()
        if shared._namespace_data_cache is not None:
            shared._namespace_data_cache.clear()
        if shared._update_flags is not None:
            shared._update_flags.clear()


def paper_kind(storage: Path) -> str:
    """fresh, resume, or done. Duplicate marker rows are ignored."""
    counts = status_counts(storage)
    if not counts:
        return "fresh"
    real = sum(value for key, value in counts.items() if key != "actionable")
    failed = int(counts.get("failed", 0))
    processed = int(counts.get("processed", 0))
    actionable = int(counts.get("actionable", 0))
    if real >= 1 and failed == 0 and actionable == 0 and processed == real:
        return "done"
    if real:
        return "resume"
    return "fresh"


async def run_graph_per_doc(args, out: Path, sample, stop, api_key: str, n_chunks: int):
    """One LightRAG working directory per document. Does not merge graphs.

    A document that is already processed is skipped, so a later run does not
    ainsert it again. Real failures are reopened and resumed.
    """
    bad = sum(1 for doc_id, _content in sample if not SAFE_DOC_ID.fullmatch(str(doc_id)))
    if bad:
        print(f"BAD_DOC_ID n={bad}", flush=True)
        raise SystemExit(6)
    stats = {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "finish": {},
        "n_at_cap": 0,
        "max_completion": 0,
        "empty": 0,
    }
    lock = asyncio.Lock()
    client = AsyncOpenAI(
        base_url=args.llm,
        api_key=api_key,
        timeout=600,
        max_retries=0,
    )

    async def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for message in history_messages or []:
            messages.append(message)
        messages.append({"role": "user", "content": prompt})
        request = {
            "model": "generator",
            "messages": messages,
            "temperature": 0,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
        if args.max_tokens > 0:
            request["max_tokens"] = args.max_tokens
        response_format = kwargs.get("response_format")
        if response_format:
            request["response_format"] = response_format
        resp = await client.chat.completions.create(**request)
        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = resp.usage
        reason = choice.finish_reason or ""
        completion = int(usage.completion_tokens or 0) if usage else 0
        prompt_tokens = int(usage.prompt_tokens or 0) if usage else 0
        async with lock:
            stats["calls"] += 1
            stats["prompt_tokens"] += prompt_tokens
            stats["completion_tokens"] += completion
            stats["finish"][reason] = stats["finish"].get(reason, 0) + 1
            stats["max_completion"] = max(stats["max_completion"], completion)
            if args.max_tokens > 0 and completion >= args.max_tokens - 1:
                stats["n_at_cap"] += 1
            if not text:
                stats["empty"] += 1
            done = stats["calls"]
            if done == 1 or done % 20 == 0:
                print(
                    f"CALLS {done} prompt={stats['prompt_tokens']} "
                    f"completion={stats['completion_tokens']} "
                    f"finish={reason} completion_this={completion} "
                    f"at_cap={stats['n_at_cap']}",
                    flush=True,
                )
        return text

    async def embed(texts, **kwargs):
        return np.zeros((len(texts), 1024), dtype=np.float32)

    embedding_func = EmbeddingFunc(
        embedding_dim=1024,
        max_token_size=8192,
        func=embed,
        model_name="zero-for-probe",
    )
    papers_root = out / "papers"
    papers_root.mkdir(parents=True, exist_ok=True)
    skipped = 0
    ran = 0
    failed_left = 0
    status = "done"
    work_error: BaseException | None = None
    t0 = time.time()

    async def one_paper(doc_id, content):
        paper_out = papers_root / doc_id
        paper_out.mkdir(parents=True, exist_ok=True)
        if paper_kind(paper_out / "storage") == "done":
            return "skip"
        reopen_real_failures(paper_out / "storage")
        counts = status_counts(paper_out / "storage")
        real_rows = 0
        if counts is not None:
            real_rows = sum(value for key, value in counts.items() if key != "actionable")
        mode = "resume" if real_rows else "fresh"
        await _isolate_paper_storage()
        rag = LightRAG(
            working_dir=str(paper_out / "storage"),
            llm_model_func=llm_model_func,
            embedding_func=embedding_func,
            llm_model_max_async=args.async_llm,
            max_parallel_insert=args.parallel_insert,
            embedding_func_max_async=args.embedding_async,
            chunk_token_size=CHUNK,
            chunk_overlap_token_size=OVERLAP,
            entity_extract_max_gleaning=1,
            entity_extraction_use_json=False,
            enable_llm_cache=args.llm_cache,
            enable_llm_cache_for_entity_extract=args.llm_cache,
            tiktoken_model_name="gpt-4o-mini",
        )
        await rag.initialize_storages()
        drain = install_batched_flush(rag, args.flush_every)
        try:
            if mode == "fresh":
                await rag.ainsert([content], ids=[doc_id], file_paths=[doc_id])
            else:
                await rag.apipeline_process_enqueue_documents()
        finally:
            try:
                await drain()
            except Exception as exc:
                print(f"DRAIN_WARN {type(exc).__name__}", flush=True)
            try:
                await rag.finalize_storages()
            except Exception as exc:
                print(f"FINALIZE_WARN {type(exc).__name__}", flush=True)
        if paper_kind(paper_out / "storage") != "done":
            return "failed"
        return "ok"

    async def run_all():
        nonlocal skipped, ran, failed_left
        for doc_id, content in sample:
            if stop.is_set():
                raise asyncio.CancelledError()
            result = await one_paper(doc_id, content)
            if result == "skip":
                skipped += 1
            else:
                ran += 1
                if result == "failed":
                    failed_left += 1
            finished = skipped + ran
            if finished == 1 or finished % 25 == 0:
                print(
                    f"PAPERS ok={finished - failed_left} skipped={skipped} "
                    f"ran={ran} failed_left={failed_left} total={len(sample)}",
                    flush=True,
                )
        print(
            f"PAPERS_END skipped={skipped} ran={ran} "
            f"failed_left={failed_left} total={len(sample)}",
            flush=True,
        )

    work = None
    stop_task = None
    timer = None
    try:
        if stop.is_set():
            status = "signal"
            print(f"SIGTERM group={args.group}", flush=True)
        else:
            work = asyncio.create_task(run_all())
            stop_task = asyncio.create_task(stop.wait())
            timer = asyncio.create_task(asyncio.sleep(args.max_wall_sec))
            done, _pending = await asyncio.wait(
                {work, stop_task, timer},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if work not in done:
                status = "signal" if stop.is_set() else "timeout"
                print(
                    f"{status.upper()} group={args.group} wall={args.max_wall_sec}",
                    flush=True,
                )
                work.cancel()
                try:
                    await work
                except asyncio.CancelledError:
                    pass
            elif work.cancelled():
                status = "signal" if stop.is_set() else "error"
            else:
                work_error = work.exception()
                if work_error is not None:
                    status = "error"
                    print(f"ERROR group={args.group} {type(work_error).__name__}", flush=True)
    finally:
        pending_tasks = [
            task
            for task in (stop_task, timer)
            if task is not None and not task.done()
        ]
        for task in pending_tasks:
            task.cancel()
        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)
        elapsed = time.time() - t0
        try:
            await client.close()
        except Exception as exc:
            print(f"CLIENT_CLOSE_WARN {type(exc).__name__}", flush=True)
        try:
            loop = asyncio.get_running_loop()
            loop.remove_signal_handler(signal.SIGTERM)
        except Exception:
            pass
        calls = stats["calls"]
        summary = {
            "status": status,
            "group": args.group,
            "route": args.route,
            "submit_mode": "graph_per_doc",
            "graph_per_doc": True,
            "docs": len(sample),
            "chunks": n_chunks,
            "papers_skipped": skipped,
            "papers_ran": ran,
            "papers_failed_left": failed_left,
            "wrapper_calls": calls,
            "insert_sec": elapsed,
            "chunks_per_min": (n_chunks / elapsed * 60) if elapsed else None,
            "calls_per_chunk": (calls / n_chunks) if n_chunks else None,
            "usage": {
                "prompt_tokens": stats["prompt_tokens"],
                "completion_tokens": stats["completion_tokens"],
                "total_tokens": stats["prompt_tokens"] + stats["completion_tokens"],
                "call_count": calls,
            },
            "finish_reason": stats["finish"],
            "n_at_cap": stats["n_at_cap"],
            "frac_at_cap": (stats["n_at_cap"] / calls) if calls else None,
            "max_completion_tokens": stats["max_completion"],
            "empty_responses": stats["empty"],
            "max_tokens": args.max_tokens,
            "llm_model_max_async": args.async_llm,
            "max_parallel_insert": args.parallel_insert,
            "embedding_func_max_async": args.embedding_async,
            "entity_extract_max_gleaning": 1,
            "enable_llm_cache": args.llm_cache,
            "flush_every": args.flush_every,
            "seed": args.seed,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "gpu": os.environ.get("PROBE_GPU", ""),
            "max_wall_sec": args.max_wall_sec,
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print("SUMMARY", json.dumps(summary), flush=True)
    if work_error is not None or status != "done":
        raise SystemExit(1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--llm", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--route", choices=["direct", "gateway"], required=True)
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--async-llm", type=int, default=32)
    parser.add_argument("--parallel-insert", type=int, default=16)
    parser.add_argument("--embedding-async", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=2048, help="0 leaves generation uncapped. Default 2048.")
    parser.add_argument("--shard-index", type=int, default=-1)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-wall-sec", type=int, default=720)
    parser.add_argument(
        "--flush-every",
        type=int,
        default=1,
        help="Persist the graph and vector stores every N finished docs. 1 is the library default (flush after each doc).",
    )
    parser.add_argument(
        "--llm-cache",
        action="store_true",
        help="Enable the LLM response cache. Default off, matching the index runs.",
    )
    parser.add_argument(
        "--id-file",
        default="",
        help="Keep only these doc ids, sorted. Empty keeps the sampled shard.",
    )
    parser.add_argument(
        "--graph-per-doc",
        action="store_true",
        help="Write one LightRAG working directory per document. Do not merge them.",
    )
    parser.add_argument(
        "--disk-guard",
        action="store_true",
        help="Every 10 min, exit after the current batch under 20 GiB free, or immediately under 2 GiB.",
    )
    return parser.parse_args()


async def main():
    args = parse_args()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, stop.set)
    guard = DiskGuard(args.disk_guard)
    guard.bind(loop, stop)
    guard.start()
    api_key = os.environ["LIGHTRAG_LLM_API_KEY"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reopen_real_failures(out / "storage")
    # Decide from the file on disk, before storages load it. An id that is
    # already present must not be passed to ainsert: enqueue treats every
    # existing row as a duplicate and will not reprocess it.
    counts = status_counts(out / "storage")
    real_rows = 0
    if counts is not None:
        real_rows = sum(value for key, value in counts.items() if key != "actionable")
    mode = "resume" if real_rows else "fresh"
    before_processed = doc_status_report(out / "storage")["processed"]
    sample = take_shard(
        load_sample(Path(args.corpus), args.n, args.seed),
        args.shard_index,
        args.num_shards,
    )
    if args.id_file:
        sample = filter_id_file(sample, args.id_file)
    tokenizer = TiktokenTokenizer("gpt-4o-mini")
    n_chunks = 0
    for _, content in sample:
        n_chunks += len(
            chunking_by_token_size(
                tokenizer,
                content,
                chunk_overlap_token_size=OVERLAP,
                chunk_token_size=CHUNK,
            )
        )
    (out / "doc_ids.json").write_text(
        json.dumps([doc_id for doc_id, _ in sample]) + "\n"
    )
    print(
        f"SAMPLE group={args.group} route={args.route} docs={len(sample)} "
        f"chunks={n_chunks} max_tokens={args.max_tokens} "
        f"async={args.async_llm} parallel={args.parallel_insert} "
        f"embed_async={args.embedding_async} flush_every={args.flush_every}",
        flush=True,
    )
    if args.graph_per_doc:
        await run_graph_per_doc(args, out, sample, stop, api_key, n_chunks)
        return

    stats = {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "finish": {},
        "n_at_cap": 0,
        "max_completion": 0,
        "empty": 0,
    }
    lock = asyncio.Lock()
    client = AsyncOpenAI(
        base_url=args.llm,
        api_key=api_key,
        timeout=600,
        max_retries=0,
    )

    async def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for message in history_messages or []:
            messages.append(message)
        messages.append({"role": "user", "content": prompt})
        request = {
            "model": "generator",
            "messages": messages,
            "temperature": 0,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
        if args.max_tokens > 0:
            request["max_tokens"] = args.max_tokens
        response_format = kwargs.get("response_format")
        if response_format:
            request["response_format"] = response_format
        resp = await client.chat.completions.create(**request)
        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = resp.usage
        reason = choice.finish_reason or ""
        completion = int(usage.completion_tokens or 0) if usage else 0
        prompt_tokens = int(usage.prompt_tokens or 0) if usage else 0
        async with lock:
            stats["calls"] += 1
            stats["prompt_tokens"] += prompt_tokens
            stats["completion_tokens"] += completion
            stats["finish"][reason] = stats["finish"].get(reason, 0) + 1
            stats["max_completion"] = max(stats["max_completion"], completion)
            if args.max_tokens > 0 and completion >= args.max_tokens - 1:
                stats["n_at_cap"] += 1
            if not text:
                stats["empty"] += 1
            done = stats["calls"]
            if done == 1 or done % 20 == 0:
                print(
                    f"CALLS {done} prompt={stats['prompt_tokens']} "
                    f"completion={stats['completion_tokens']} "
                    f"finish={reason} completion_this={completion} "
                    f"at_cap={stats['n_at_cap']}",
                    flush=True,
                )
        return text

    async def embed(texts, **kwargs):
        return np.zeros((len(texts), 1024), dtype=np.float32)

    embedding_func = EmbeddingFunc(
        embedding_dim=1024,
        max_token_size=8192,
        func=embed,
        model_name="zero-for-probe",
    )
    rag = LightRAG(
        working_dir=str(out / "storage"),
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        llm_model_max_async=args.async_llm,
        max_parallel_insert=args.parallel_insert,
        embedding_func_max_async=args.embedding_async,
        chunk_token_size=CHUNK,
        chunk_overlap_token_size=OVERLAP,
        entity_extract_max_gleaning=1,
        entity_extraction_use_json=False,
        enable_llm_cache=args.llm_cache,
        enable_llm_cache_for_entity_extract=args.llm_cache,
        tiktoken_model_name="gpt-4o-mini",
    )
    await rag.initialize_storages()
    # After initialize, so startup can still load and flush with the real callback.
    def on_flush():
        if guard.note_batch_end():
            stop.set()

    drain_batched = install_batched_flush(
        rag,
        args.flush_every,
        on_flush if args.disk_guard else None,
    )
    status = "done"
    work_error: BaseException | None = None
    t0 = time.time()

    async def run_work():
        if mode == "fresh":
            print(f"SUBMIT_MODE fresh docs={len(sample)}", flush=True)
            await rag.ainsert(
                [content for _, content in sample],
                ids=[doc_id for doc_id, _ in sample],
                file_paths=[doc_id for doc_id, _ in sample],
            )
            return
        print(
            "SUBMIT_MODE resume "
            f"actionable={counts.get('actionable')} "
            f"counts={json.dumps(counts, sort_keys=True)}",
            flush=True,
        )
        # A finished run can still be missing a doc that never reached
        # doc_status. Insert only those ids. Existing rows stay on enqueue.
        status_path = out / "storage" / "kv_store_doc_status.json"
        present = set()
        if status_path.exists():
            present = {
                str(key)
                for key in json.loads(status_path.read_text())
                if not str(key).startswith(("dup-", "pad-"))
            }
        missing = [
            (doc_id, content)
            for doc_id, content in sample
            if doc_id not in present
        ]
        if missing:
            print(f"SUBMIT_MISSING n={len(missing)}", flush=True)
            await rag.ainsert(
                [content for _, content in missing],
                ids=[doc_id for doc_id, _ in missing],
                file_paths=[doc_id for doc_id, _ in missing],
            )
        await rag.apipeline_process_enqueue_documents()

    work = None
    stop_task = None
    timer = None
    try:
        if stop.is_set():
            status = "signal"
            print(f"SIGTERM group={args.group}", flush=True)
        else:
            work = asyncio.create_task(run_work())
            stop_task = asyncio.create_task(stop.wait())
            timer = asyncio.create_task(asyncio.sleep(args.max_wall_sec))
            done, _pending = await asyncio.wait(
                {work, stop_task, timer},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if work not in done:
                status = "signal" if stop.is_set() else "timeout"
                print(
                    f"{status.upper()} group={args.group} wall={args.max_wall_sec}",
                    flush=True,
                )
                work.cancel()
                try:
                    await work
                except asyncio.CancelledError:
                    pass
            elif work.cancelled():
                status = "signal" if stop.is_set() else "error"
            else:
                work_error = work.exception()
                if work_error is not None:
                    status = "error"
                    print(
                        f"ERROR group={args.group} {type(work_error).__name__}",
                        flush=True,
                    )
    finally:
        pending_tasks = [
            task
            for task in (stop_task, timer)
            if task is not None and not task.done()
        ]
        for task in pending_tasks:
            task.cancel()
        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)
        elapsed = time.time() - t0
        try:
            await drain_batched()
        except Exception as exc:
            print(f"DRAIN_WARN {type(exc).__name__}", flush=True)
        try:
            await client.close()
        except Exception as exc:
            print(f"CLIENT_CLOSE_WARN {type(exc).__name__}", flush=True)
        try:
            await rag.finalize_storages()
        except Exception as exc:
            print(f"FINALIZE_WARN {type(exc).__name__}", flush=True)
        try:
            loop.remove_signal_handler(signal.SIGTERM)
        except Exception:
            pass
        graph_path = out / "storage" / "graph_chunk_entity_relation.graphml"
        nodes = edges = None
        if graph_path.exists():
            graph = nx.read_graphml(graph_path)
            nodes = graph.number_of_nodes()
            edges = graph.number_of_edges()
        calls = stats["calls"]
        after_processed = doc_status_report(out / "storage")["processed"]
        new_processed = max(0, after_processed - before_processed)
        flushes = 0
        status_flushes = getattr(drain_batched, "status_flushes", None)
        if callable(status_flushes):
            flushes = int(status_flushes())
        summary = {
            "status": status,
            "group": args.group,
            "route": args.route,
            "submit_mode": mode,
            "docs": len(sample),
            "chunks": n_chunks,
            "before_processed": before_processed,
            "new_processed": new_processed,
            "docs_per_min": (new_processed / elapsed * 60) if elapsed else None,
            "doc_status_flushes": flushes,
            "wrapper_calls": calls,
            "insert_sec": elapsed,
            "chunks_per_min": (n_chunks / elapsed * 60) if elapsed else None,
            "calls_per_chunk": (calls / n_chunks) if n_chunks else None,
            "usage": {
                "prompt_tokens": stats["prompt_tokens"],
                "completion_tokens": stats["completion_tokens"],
                "total_tokens": stats["prompt_tokens"] + stats["completion_tokens"],
                "call_count": calls,
            },
            "finish_reason": stats["finish"],
            "n_at_cap": stats["n_at_cap"],
            "frac_at_cap": (stats["n_at_cap"] / calls) if calls else None,
            "max_completion_tokens": stats["max_completion"],
            "empty_responses": stats["empty"],
            "max_tokens": args.max_tokens,
            "llm_model_max_async": args.async_llm,
            "max_parallel_insert": args.parallel_insert,
            "embedding_func_max_async": args.embedding_async,
            "entity_extract_max_gleaning": 1,
            "enable_llm_cache": args.llm_cache,
            "flush_every": args.flush_every,
            "doc_status": doc_status_report(out / "storage"),
            "seed": args.seed,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "nodes": nodes,
            "edges": edges,
            "gpu": os.environ.get("PROBE_GPU", ""),
            "max_wall_sec": args.max_wall_sec,
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print("SUMMARY", json.dumps(summary), flush=True)
    guard.close()
    disk_code = 74 if guard.level == "immediate" else 75 if guard.level == "warn" else None
    if work_error is not None:
        traceback.print_tb(work_error.__traceback__)
        raise SystemExit(1)
    if status == "done":
        return
    if disk_code is not None:
        raise SystemExit(disk_code)
    raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
