#!/usr/bin/env python3
"""Query adapter.

hybrid retrieval, one keyword call plus one answer call, thinking off,
query cache off, no reranker. QueryParam token caps stay at the library
defaults. A send-time guard drops Document Chunks until the Qwen chat
template count of both calls, plus two 512 output caps, is within 32k.
"""

import argparse
import asyncio
import atexit
import json
import os
import re
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(os.environ.get("SCRATCH_DIR", "/tmp"))

import numpy as np
from openai import AsyncOpenAI

from bge_client import embed_texts
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc, Tokenizer
from qwen_tok import QwenPieceTokenizer, guard_messages

ANSWER_MAX_TOKENS = 512
SAFE_DOC_ID = re.compile(r"^[A-Za-z0-9_-]+$")
GUARD = {"questions": 0, "triggered": 0, "dropped": 0, "printed": False}
QUERY_LOCK = threading.Lock()


# The library answer template still demands a long write-up and a
# References section. user_prompt is only its last slot, so the same
# ragcore lines have to be the whole system prompt. Placeholders match
# kg_query's format call. Retrieval is unchanged.
ANSWER_SYSTEM_PROMPT = "{user_prompt}\n\n---Context---\n\n{context_data}\n"


def short_answer_instruction(info_path: str) -> str:
    """Same instruction lines as baselines/ragcore.py answer_prompt.

    Passages stay in LightRAG's context slot. This text is
    QueryParam.user_prompt. The answer system prompt is only that text
    plus the retrieved context.
    """
    info = json.loads(Path(info_path).read_text(encoding="utf-8"))
    answer_format = str(info.get("answer_format") or "").strip()
    if not answer_format or "\n" in answer_format or len(answer_format) > 200:
        raise SystemExit("BAD_ANSWER_FORMAT")
    return (
        "Use the passages to answer the question.\n"
        f"Answer format: {answer_format}.\n"
        "Reply with the answer only."
    )


def emit_guard_sum():
    if GUARD["printed"]:
        return
    GUARD["printed"] = True
    print(
        f"CTX_GUARD_SUM questions={GUARD['questions']} "
        f"triggered={GUARD['triggered']} dropped_chunks={GUARD['dropped']}",
        flush=True,
    )


def build_rag(storage: str, llm_url: str, bge_sock: str, qwen: QwenPieceTokenizer):
    stats = {
        "calls": 0,
        "prompt": 0,
        "completion": 0,
        "thinking": False,
        "qid": "",
        "kw_tokens": 0,
    }
    client = AsyncOpenAI(base_url=llm_url, api_key="-", timeout=90, max_retries=2)

    async def llm_model_func(
        prompt, system_prompt=None, history_messages=None, **kwargs
    ):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for message in history_messages or []:
            messages.append(message)
        messages.append({"role": "user", "content": prompt})
        is_answer = any(
            "Document Chunks (" in (item.get("content") or "") for item in messages
        )
        if not is_answer:
            stats["kw_tokens"] = qwen.count_chat(messages)
        else:
            before, after, dropped, total = guard_messages(
                qwen, messages, int(stats.get("kw_tokens") or 0)
            )
            GUARD["questions"] += 1
            GUARD["dropped"] += dropped
            if dropped:
                GUARD["triggered"] += 1
            print(
                f"CTX_GUARD qid={stats.get('qid') or '-'} "
                f"chunks_before={before} chunks_after={after} "
                f"dropped={dropped} total_est={total}",
                flush=True,
            )
        if is_answer and stats.get("retrieve_only"):
            from qwen_tok import chunk_lines

            context = next(
                (
                    m["content"]
                    for m in messages
                    if "Document Chunks (" in m.get("content", "")
                ),
                "",
            )
            stats["guarded_chunks"] = [
                json.loads(line) for line in (chunk_lines(context) or [])
            ]
            return ""
        response = await client.chat.completions.create(
            model="generator",
            messages=messages,
            temperature=0,
            max_tokens=ANSWER_MAX_TOKENS,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        choice = response.choices[0]
        message = choice.message
        text = message.content or ""
        reasoning = getattr(message, "reasoning_content", None) or ""
        if reasoning or "<think>" in text or "</think>" in text:
            stats["thinking"] = True
        usage = response.usage
        stats["calls"] += 1
        if usage is not None:
            stats["prompt"] += int(usage.prompt_tokens or 0)
            stats["completion"] += int(usage.completion_tokens or 0)
        return text

    async def embed(texts, **kwargs):
        if not texts:
            return np.zeros((0, 1024), dtype=np.float32)
        return embed_texts(bge_sock, list(texts))

    rag = LightRAG(
        working_dir=storage,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=embed,
            model_name="bge-m3",
        ),
        llm_model_max_async=2,
        embedding_func_max_async=1,
        enable_llm_cache=False,
        enable_llm_cache_for_entity_extract=False,
        entity_extract_max_gleaning=1,
        tokenizer=Tokenizer("Qwen3.5-4B", qwen),
    )
    return rag, stats, client


def citations_from(result: dict) -> list[str]:
    data = result.get("data") or {}
    found = []
    for bucket in ("chunks", "entities", "relationships"):
        for item in data.get(bucket) or []:
            if not isinstance(item, dict):
                continue
            file_path = str(item.get("file_path") or "").strip()
            if file_path and file_path not in found and file_path != "unknown_source":
                found.append(file_path)
            if len(found) >= 20:
                return found
    return found


class PaperIndex:
    """Load one paper graph on demand. Paths stay inside the given roots."""

    def __init__(self, roots, llm_url: str, bge_sock: str, loop, qwen, id_map=None):
        self.roots = [Path(root) for root in roots]
        self.llm_url = llm_url
        self.bge_sock = bge_sock
        self.loop = loop
        self.qwen = qwen
        self.id_map = id_map or {}
        self.cache = {}
        self.lock = threading.Lock()

    def resolve(self, meta):
        doc_id = ""
        if isinstance(meta, dict):
            value = meta.get("doc_id")
            if isinstance(value, str):
                doc_id = value
        if not SAFE_DOC_ID.fullmatch(doc_id):
            return None
        mapped = self.id_map.get(doc_id)
        if isinstance(mapped, str) and mapped != doc_id:
            if not SAFE_DOC_ID.fullmatch(mapped):
                return None
            doc_id = mapped
        storage = None
        for root in self.roots:
            candidate = root / doc_id / "storage"
            try:
                candidate.resolve().relative_to(root.resolve())
            except ValueError:
                return None
            if candidate.is_dir():
                storage = candidate
                break
        if storage is None:
            return None
        key = str(storage.resolve())
        with self.lock:
            hit = self.cache.get(key)
        if hit is not None:
            return hit
        rag, stats, _client = build_rag(key, self.llm_url, self.bge_sock, self.qwen)
        future = asyncio.run_coroutine_threadsafe(rag.initialize_storages(), self.loop)
        future.result(timeout=180)
        with self.lock:
            self.cache.setdefault(key, (rag, stats))
            return self.cache[key]


async def retrieve_chunks(
    rag: LightRAG,
    question: str,
    param: QueryParam,
    stats: dict,
    source_id_map: dict[str, str] | None = None,
) -> dict:
    from fe_corpus import corpus_evidence, check_limits

    stats["retrieve_only"] = True
    stats["guarded_chunks"] = []
    try:
        result = await rag.aquery_llm(
            question, param=param, system_prompt=ANSWER_SYSTEM_PROMPT
        )
    finally:
        stats["retrieve_only"] = False
    if result.get("status") == "failure":
        if (result.get("metadata") or {}).get("failure_reason") == "no_results":
            return {"passages": []}
        raise ValueError("LightRAG retrieval failed")
    selected = stats["guarded_chunks"]
    evidence = corpus_evidence()
    passages = []
    remaining = list((result.get("data") or {}).get("chunks", []))
    for entry in selected:
        # Follow the exact guarded context order; every selected entry must
        # map back to one stored chunk, never silently drop a mismatch.
        match = next(
            (
                i
                for i, c in enumerate(remaining)
                if str(entry.get("reference_id")) == str(c.get("reference_id"))
                and entry.get("content") == c.get("content")
            ),
            None,
        )
        if match is None:
            raise ValueError("guarded chunk missing from retrieval result")
        chunk = remaining.pop(match)
        stored = await rag.text_chunks.get_by_id(chunk["chunk_id"])
        if not stored:
            raise ValueError("selected chunk source missing")
        did = str(stored.get("file_path") or chunk.get("file_path") or "")
        if source_id_map is not None:
            if did not in source_id_map:
                raise ValueError("qasper chunk ID missing from formal map")
            did = source_id_map[did]
        text = str(stored.get("content") or "")
        passages.append(evidence.chunk_passage(text, doc_id=did))
    return {"passages": check_limits(passages)}


def make_handler(rag, stats, loop, user_prompt, paper_index=None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def _send(self, code: int, payload: dict):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.split("?", 1)[0] != "/health":
                self._send(404, {"error": "not found"})
                return
            self._send(200, {"status": "ok"})

        def do_POST(self):
            if self.path.split("?", 1)[0] not in ("/answer", "/retrieve"):
                self._send(404, {"error": "not found"})
                return
            retrieval_only = self.path.split("?", 1)[0] == "/retrieve"
            length = int(self.headers.get("Content-Length") or "0")
            raw = self.rfile.read(length) if length else b"{}"
            try:
                request = json.loads(raw.decode("utf-8"))
                question = str(request.get("question") or "").strip()
            except Exception:
                self._send(400, {"error": "invalid json"})
                return
            if not question:
                self._send(400, {"error": "empty question"})
                return
            use_rag = rag
            use_stats = stats
            if paper_index is not None:
                meta = request.get("meta") if isinstance(request, dict) else None
                try:
                    resolved = paper_index.resolve(meta)
                except Exception as exc:
                    self._send(500, {"error": type(exc).__name__})
                    print(f"PAPER_INIT_FAIL err={type(exc).__name__}", flush=True)
                    return
                if resolved is None:
                    self._send(404, {"error": "paper_not_found"})
                    return
                use_rag, use_stats = resolved
            use_stats["calls"] = 0
            use_stats["prompt"] = 0
            use_stats["completion"] = 0
            use_stats["thinking"] = False
            use_stats["kw_tokens"] = 0
            qid = request.get("qid")
            use_stats["qid"] = str(qid) if isinstance(qid, (str, int)) else ""
            started = time.time()
            param = QueryParam(
                mode="hybrid",
                enable_rerank=False,
                response_type="A short phrase",
                user_prompt=user_prompt,
                stream=False,
            )
            try:
                with QUERY_LOCK:
                    future = asyncio.run_coroutine_threadsafe(
                        (
                            retrieve_chunks(
                                use_rag,
                                question,
                                param,
                                use_stats,
                                (
                                    {v: k for k, v in paper_index.id_map.items()}
                                    if paper_index is not None
                                    else None
                                ),
                            )
                            if retrieval_only
                            else use_rag.aquery_llm(
                                question,
                                param=param,
                                system_prompt=ANSWER_SYSTEM_PROMPT,
                            )
                        ),
                        loop,
                    )
                    result = future.result(timeout=120)
            except Exception as exc:
                self._send(500, {"error": f"{type(exc).__name__}"})
                print(
                    f"ANSWER_FAIL calls={use_stats['calls']} sec={time.time() - started:.2f} "
                    f"err={type(exc).__name__}",
                    flush=True,
                )
                return
            if not isinstance(result, dict) or result.get("status") == "failure":
                self._send(500, {"error": "query_failed"})
                print(
                    f"ANSWER_FAIL calls={use_stats['calls']} sec={time.time() - started:.2f} "
                    f"err=query_failed",
                    flush=True,
                )
                return
            if retrieval_only:
                self._send(200, {"passages": result["passages"]})
                return
            content = str((result.get("llm_response") or {}).get("content") or "")
            elapsed = time.time() - started
            print(
                f"ANSWER calls={use_stats['calls']} prompt={use_stats['prompt']} "
                f"completion={use_stats['completion']} "
                f"total={use_stats['prompt'] + use_stats['completion']} "
                f"thinking={int(use_stats['thinking'])} sec={elapsed:.2f}",
                flush=True,
            )
            self._send(
                200,
                {
                    "answer": content.strip(),
                    "citations": citations_from(result),
                    "thinking": bool(use_stats["thinking"]),
                },
            )

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage", default="")
    parser.add_argument("--paper-root", action="append", default=[])
    parser.add_argument("--bge-sock", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--llm", required=True)
    parser.add_argument("--task-info", required=True)
    parser.add_argument("--doc-id-map", default="")
    args = parser.parse_args()
    if bool(args.storage) == bool(args.paper_root):
        raise SystemExit("QUERY_STORAGE_MODE")
    from fe_corpus import corpus_evidence

    corpus_evidence()  # Build provenance lookup during service startup, outside question budget.
    user_prompt = short_answer_instruction(args.task_info)
    print(f"USER_PROMPT_CHARS={len(user_prompt)}", flush=True)
    qwen = QwenPieceTokenizer()
    print("QWEN_TOKENIZER_READY", flush=True)
    atexit.register(emit_guard_sum)

    def _on_term(signum, frame):
        emit_guard_sum()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _on_term)
    loop = asyncio.new_event_loop()
    ready = threading.Event()
    if args.storage:
        rag, stats, _client = build_rag(args.storage, args.llm, args.bge_sock, qwen)

        def _run_loop():
            asyncio.set_event_loop(loop)
            loop.run_until_complete(rag.initialize_storages())
            ready.set()
            loop.run_forever()

        handler = make_handler(rag, stats, loop, user_prompt)
    else:
        rag = None

        def _run_loop():
            asyncio.set_event_loop(loop)
            ready.set()
            loop.run_forever()

        id_map = {}
        if args.doc_id_map:
            loaded = json.loads(Path(args.doc_id_map).read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise SystemExit("BAD_DOC_MAP")
            id_map = {str(key): str(value) for key, value in loaded.items()}
            print(f"DOC_ID_MAP n={len(id_map)}", flush=True)
        paper_index = PaperIndex(
            args.paper_root, args.llm, args.bge_sock, loop, qwen, id_map
        )
        handler = make_handler(
            None,
            {"calls": 0, "prompt": 0, "completion": 0, "thinking": False},
            loop,
            user_prompt,
            paper_index,
        )

    thread = threading.Thread(target=_run_loop, daemon=True)
    thread.start()
    if not ready.wait(timeout=900):
        raise SystemExit("QUERY_INIT_TIMEOUT")
    print("QUERY_READY", flush=True)
    print("QUERY_LISTEN", flush=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
