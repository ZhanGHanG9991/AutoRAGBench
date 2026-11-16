"""Build a restartable formal-v1 Contriever index without reading hidden data."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from encoder import Encoder

ROOT = Path(os.environ["AUTORAGBENCH_ROOT"])
BASE = ROOT / "humansota/HyDE"
FORMAL = BASE / "rageng/formal-v1"
MODEL = BASE / "rageng/models/contriever"
DIM = 768
CHUNK_TOKENS = 400
CHUNKED_TASKS = {"qasper", "webqsp", "cwq", "multihop_rag"}
TASKS = {"musique", "hotpotqa", "nq_open", "pubmedqa", "fever", "qasper", "webqsp", "cwq", "multihop_rag"}
WINDOW = 4096
BATCH = 256


def atomic_json(path: Path, value: object) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2))
    os.replace(temp, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(task: str, corpus: Path, out: Path, tokenizer) -> dict:
    plan_path = out / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if plan.get("prepared") and Path(plan["corpus"]).resolve() == corpus.resolve():
            return plan
        raise RuntimeError(f"incomplete or mismatched plan: {plan_path}")

    digest = hashlib.sha256()
    line_offsets = [0]
    meta: list[tuple[int, int, int]] = []
    n_documents = 0
    position = 0
    chunked = task in CHUNKED_TASKS
    with corpus.open("rb") as handle:
        for line in handle:
            position += len(line)
            if not line.strip():
                line_offsets[-1] = position
                continue
            digest.update(line)
            doc = json.loads(line)
            if "doc_id" not in doc or "text" not in doc:
                raise ValueError("corpus row lacks doc_id/text")
            if chunked:
                offsets = tokenizer(
                    str(doc["text"]), add_special_tokens=False, return_offsets_mapping=True
                )["offset_mapping"]
                for start in range(0, max(1, len(offsets)), CHUNK_TOKENS):
                    span = offsets[start : start + CHUNK_TOKENS]
                    begin, end = (span[0][0], span[-1][1]) if span else (0, 0)
                    meta.append((n_documents, begin, end))
            n_documents += 1
            line_offsets.append(position)
            if n_documents % 500000 == 0:
                print(f"PREPARE {task} documents={n_documents} chunks={len(meta)}", flush=True)

    np.save(out / "line_offsets.npy", np.asarray(line_offsets, dtype=np.uint64))
    if chunked:
        dtype = np.dtype([("doc_row", "<u4"), ("start", "<u4"), ("end", "<u4")])
        np.save(out / "chunk_meta.npy", np.asarray(meta, dtype=dtype))
        n_vectors = len(meta)
    else:
        n_vectors = n_documents
    plan = {
        "prepared": True,
        "task": task,
        "data_version": "formal-v1",
        "corpus": str(corpus),
        "source_sha256": digest.hexdigest(),
        "n_documents": n_documents,
        "n_vectors": n_vectors,
        "chunked": chunked,
        "chunk_tokens": CHUNK_TOKENS if chunked else None,
        "dimension": DIM,
        "dtype": "float16",
        "pooling": "attention-mask mean",
        "max_length": 512,
        "normalized": False,
        "similarity": "inner product",
        "model": "facebook/contriever",
        "model_revision": "2bd46a25019aeea091fd42d1f0fd4801675cf699",
    }
    atomic_json(plan_path, plan)
    return plan


def read_document(handle, offsets: np.ndarray, row: int) -> dict:
    handle.seek(int(offsets[row]))
    line = handle.readline()
    if not line:
        raise EOFError(f"missing corpus row {row}")
    return json.loads(line)


def texts_for_range(corpus: Path, offsets: np.ndarray, meta, start: int, end: int) -> list[str]:
    texts: list[str] = []
    with corpus.open("rb") as handle:
        for vector_row in range(start, end):
            if meta is None:
                doc_row, begin, finish = vector_row, 0, None
            else:
                item = meta[vector_row]
                doc_row, begin, finish = int(item["doc_row"]), int(item["start"]), int(item["end"])
            doc = read_document(handle, offsets, doc_row)
            body = str(doc.get("text", ""))
            if finish is not None:
                body = body[begin:finish]
            title = str(doc.get("title") or "").strip()
            texts.append((title + "\n" + body).strip() if title else body)
    return texts


def encode(task: str, corpus: Path, out: Path, plan: dict, encoder: Encoder) -> None:
    final = out / "vectors.npy"
    if final.exists() and (out / "COMPLETE").exists():
        print(f"INDEX_REUSED {task}", flush=True)
        return
    partial = out / "vectors.partial.npy"
    progress_path = out / "progress.json"
    n = int(plan["n_vectors"])
    if partial.exists():
        progress = json.loads(progress_path.read_text())
        done = int(progress["encoded"])
        vectors = np.lib.format.open_memmap(partial, mode="r+")
        if vectors.shape != (n, DIM) or vectors.dtype != np.float16:
            raise RuntimeError("partial vector shape/dtype mismatch")
    else:
        done = 0
        vectors = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float16, shape=(n, DIM))
        atomic_json(progress_path, {"encoded": 0, "n_vectors": n})
    offsets = np.load(out / "line_offsets.npy", mmap_mode="r")
    meta = np.load(out / "chunk_meta.npy", mmap_mode="r") if plan["chunked"] else None
    started = time.monotonic()
    for window_start in range(done, n, WINDOW):
        window_end = min(n, window_start + WINDOW)
        texts = texts_for_range(corpus, offsets, meta, window_start, window_end)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        stored = np.empty((len(texts), DIM), dtype=np.float16)
        for batch_start in range(0, len(order), BATCH):
            indices = order[batch_start : batch_start + BATCH]
            encoded = encoder.batch([texts[i] for i in indices]).astype(np.float16)
            stored[indices] = encoded
        vectors[window_start:window_end] = stored
        vectors.flush()
        atomic_json(progress_path, {"encoded": window_end, "n_vectors": n})
        elapsed = max(time.monotonic() - started, 1e-6)
        print(
            f"ENCODE {task} {window_end}/{n} session_rate={(window_end-done)/elapsed:.1f}/s",
            flush=True,
        )
    del vectors
    os.replace(partial, final)


def verify(task: str, out: Path, plan: dict, elapsed: float) -> None:
    vectors = np.load(out / "vectors.npy", mmap_mode="r")
    if vectors.shape != (int(plan["n_vectors"]), DIM) or vectors.dtype != np.float16:
        raise RuntimeError("final vector shape/dtype mismatch")
    nonfinite = zero = 0
    norm_min = float("inf")
    norm_max = 0.0
    for start in range(0, len(vectors), 100000):
        block = np.asarray(vectors[start : start + 100000], dtype=np.float32)
        nonfinite += int((~np.isfinite(block)).any(axis=1).sum())
        norms = np.linalg.norm(block, axis=1)
        zero += int((norms == 0).sum())
        if len(norms):
            norm_min = min(norm_min, float(norms.min()))
            norm_max = max(norm_max, float(norms.max()))
    if nonfinite or zero:
        raise RuntimeError(f"invalid vectors nonfinite={nonfinite} zero={zero}")
    manifest = {
        **plan,
        "build_precision": "bfloat16 autocast; float32 mean pooling; stored float16",
        "retrieval_precision": "float16 corpus vectors and query vector; exact exhaustive inner product",
        "n_nonfinite": nonfinite,
        "n_zero_rows": zero,
        "norm_min": norm_min,
        "norm_max": norm_max,
        "vectors_bytes": (out / "vectors.npy").stat().st_size,
        "elapsed_seconds_current_invocation": elapsed,
        "completed": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    atomic_json(out / "manifest.json", manifest)
    (out / "COMPLETE").write_text(json.dumps({"task": task, "n_vectors": len(vectors)}) + "\n")
    print(f"INDEX_COMPLETE {task} vectors={len(vectors)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=sorted(TASKS))
    args = parser.parse_args()
    corpus = ROOT / "data_formal-v1" / args.task / "public/corpus/corpus.jsonl"
    if not corpus.is_file() or "/data_formal-v1/" not in str(corpus):
        raise SystemExit(f"REFUSE corpus {corpus}")
    out = FORMAL / "index" / args.task
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    encoder = Encoder(str(MODEL), build_mode=True)
    plan = prepare(args.task, corpus, out, encoder.tokenizer)
    encode(args.task, corpus, out, plan, encoder)
    verify(args.task, out, plan, time.monotonic() - started)


if __name__ == "__main__":
    main()
