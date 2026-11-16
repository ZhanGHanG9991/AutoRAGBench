#!/usr/bin/env python3
"""Build formal-v1 Contriever indexes under CorrectiveRAG/rageng/formal-v1 only.

fp16 embeddings, mean pooling, unnormalised dot product. Chunking matches the
pilot final configs: multihop_rag and qasper are 200-word windows, the other
five tasks are whole documents. Does not read hidden/ or test, and does not
write the pilot index directory.
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
import hashlib
import json
import os
import sys
import time

import numpy as np

ROOT = _release_path("@AUTORAGBENCH_ROOT@")
C = os.path.join(ROOT, "humansota/CorrectiveRAG")
FORMAL = os.path.join(C, "rageng/formal-v1")
MODEL = os.path.join(C, "models/contriever-msmarco")
PILOT_INDEX = os.path.join(C, "rageng/index")
SHARD_SIZE = 1_500_000
DIM = 768

TASKS = {
    "musique": None,
    "hotpotqa": None,
    "nq_open": None,
    "fever": None,
    "pubmedqa": None,
    "multihop_rag": 200,
    "qasper": 200,
}


def refuse(path):
    text = os.path.abspath(path).replace("\\", "/")
    parts = text.split("/")
    if "hidden" in parts or "test" in parts or text.endswith("test.jsonl"):
        raise SystemExit(f"REFUSE path {path}")
    real = os.path.realpath(path) if os.path.exists(path) else os.path.realpath(os.path.dirname(path))
    pilot = os.path.realpath(PILOT_INDEX)
    if real == pilot or real.startswith(pilot + os.sep):
        raise SystemExit(f"REFUSE pilot index {path}")
    if "/rageng/index/" in text:
        raise SystemExit(f"REFUSE pilot index path {path}")


def disk_ok():
    line = os.popen(_release_path("df -BG --output=avail @BUILD_TMP@ | tail -1")).read()
    digits = "".join(ch for ch in line if ch.isdigit())
    if not digits:
        raise SystemExit("REFUSE disk check failed")
    avail = int(digits)
    print(f"DISK_AVAIL_GB {avail}", flush=True)
    if avail < 200:
        raise SystemExit(f"DISK_LOW {avail}")
    return avail


def corpus_path(task):
    path = os.path.join(ROOT, "data_formal-v1", task, "public", "corpus", "corpus.jsonl")
    refuse(path)
    suffix = f"/data_formal-v1/{task}/public/corpus/corpus.jsonl"
    if not os.path.realpath(path).endswith(suffix) and not path.endswith(suffix):
        raise SystemExit(f"REFUSE corpus {path}")
    if not os.path.isfile(path):
        raise SystemExit(f"missing corpus {path}")
    return path


def index_dir(task):
    name = f"{task}-contriever" if TASKS[task] is None else f"{task}-contriever-w200"
    path = os.path.join(FORMAL, "index", name)
    refuse(path)
    if "/formal-v1/index/" not in path:
        raise SystemExit(f"REFUSE index dir {path}")
    return path


def n_shards_for(n, chunk_words):
    if chunk_words is not None:
        return 1
    return max(1, (int(n) + SHARD_SIZE - 1) // SHARD_SIZE)


def ranges_for(n, n_shards):
    out = []
    for shard in range(n_shards):
        start = shard * n // n_shards
        end = (shard + 1) * n // n_shards
        out.append([start, end])
    return out


def passages_of(doc, chunk_words):
    """Match the pilot loaders. Unchunked matches bm25_index.load_corpus.

    Chunked matches contriever_index.load_chunked_corpus, which is the same
    200-word window as baselines/ragcore.py iter_passages. The stored passage
    is title, a newline, and the window.
    """
    if "doc_id" not in doc or "text" not in doc:
        raise SystemExit("corpus row missing doc_id or text")
    if chunk_words is None:
        title = (doc.get("title") or "").strip()
        body = (doc.get("text") or "").strip()
        passage = f"{title}\n{body}".strip() if title else body
        return [{"doc_id": doc["doc_id"], "title": title, "passage": passage}]
    title = doc["title"].replace("\t", " ").replace("\n", " ")
    toks = doc["text"].split()
    rows = []
    for start in range(0, max(len(toks), 1), chunk_words):
        window = " ".join(toks[start : start + chunk_words])
        passage = f"{title}\n{window}".strip() if title else window
        rows.append({"doc_id": doc["doc_id"], "title": title, "passage": passage})
    return rows


def count_passages(path, chunk_words):
    n = 0
    with open(path, "rb") as handle:
        for line in handle:
            if not line.strip():
                continue
            doc = json.loads(line)
            n += len(passages_of(doc, chunk_words))
    return n


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_offsets(path, dest):
    """Byte offsets of non-empty lines. A blank line is not a passage."""
    offs = []
    blanks = 0
    pos = 0
    with open(path, "rb") as handle:
        for line in handle:
            if not line.strip():
                blanks += 1
                pos += len(line)
                continue
            offs.append(pos)
            pos += len(line)
    offs.append(pos)
    arr = np.asarray(offs, dtype=np.int64)
    np.save(dest, arr)
    return int(arr.shape[0] - 1), blanks


def load_plan(task):
    path = os.path.join(index_dir(task), "plan.json")
    refuse(path)
    plan = json.load(open(path, encoding="utf-8"))
    if plan.get("task") != task or plan.get("chunk_words") != TASKS[task]:
        raise SystemExit(f"REFUSE plan mismatch {path}")
    return plan


def prepare_task(task, expect_n=None):
    disk_ok()
    chunk = TASKS[task]
    src = corpus_path(task)
    dest = index_dir(task)
    os.makedirs(dest, exist_ok=True)
    n_blank = 0
    if chunk is None:
        n, n_blank = write_offsets(src, os.path.join(dest, "line_offsets.npy"))
        if expect_n is not None and int(expect_n) != n:
            raise SystemExit(f"REFUSE {task} line count {n} != {expect_n}")
        print(f"OFFSETS {task} n={n} blank={n_blank}", flush=True)
    else:
        n = count_passages(src, chunk)
        if expect_n is not None and int(expect_n) != n:
            raise SystemExit(f"REFUSE {task} passage count {n} != {expect_n}")
    n_shards = n_shards_for(n, chunk)
    plan = {
        "task": task,
        "chunk_words": chunk,
        "n_passages": n,
        "n_blank": n_blank,
        "n_shards": n_shards,
        "shard_size_cap": SHARD_SIZE,
        "ranges": ranges_for(n, n_shards),
        "corpus": src,
        "dtype": "float16",
        "normalized": False,
        "similarity": "dot",
        "pooling": "mean",
        "max_length": 512,
        "dim": DIM,
    }
    out = os.path.join(dest, "plan.json")
    json.dump(plan, open(out, "w"), indent=1)
    print(f"PREPARE_DONE {task} n={n} shards={n_shards} chunk={chunk}", flush=True)
    return plan


def load_encoder():
    import torch
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        raise SystemExit("REFUSE encoder without CUDA")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModel.from_pretrained(MODEL, local_files_only=True)
    model.to("cuda").eval()
    return tokenizer, model


ENCODE_BATCH = 256
ENCODE_WINDOW = 4096


def encode_window(tokenizer, model, texts):
    """Same Contriever definition: max_length 512, mean pool, no L2 norm, stored fp16.

    Batches are sorted by string length so padding is not the longest row in a
    random batch. The forward uses bf16 autocast. Profile job 697622 on 1000
    musique passages: cosine against the fp32 path, min 0.999976, mean 0.999991.
    """
    import torch

    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    stored = np.empty((len(texts), DIM), dtype=np.float16)
    for start in range(0, len(order), ENCODE_BATCH):
        idx = order[start : start + ENCODE_BATCH]
        chunk = [texts[i] for i in idx]
        enc = tokenizer(chunk, padding=True, truncation=True, max_length=512, return_tensors="pt")
        enc = {key: value.to("cuda", non_blocking=True) for key, value in enc.items()}
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            hidden = model(**enc).last_hidden_state
        weights = enc["attention_mask"].unsqueeze(-1).type_as(hidden)
        pooled = (hidden * weights).sum(1) / weights.sum(1).clamp(min=1e-9)
        vecs = pooled.float().cpu().numpy().astype(np.float16)
        for row, vec in zip(idx, vecs):
            stored[row] = vec
    return stored


def iter_shard_docs(task, plan, shard, resume_rows=0):
    start, end = [int(x) for x in plan["ranges"][shard]]
    start += int(resume_rows)
    if start > end:
        raise SystemExit(f"REFUSE resume {resume_rows} past shard end")
    src = plan["corpus"]
    chunk = plan["chunk_words"]
    if chunk is None:
        offsets = np.load(os.path.join(index_dir(task), "line_offsets.npy"), mmap_mode="r")
        with open(src, "rb") as handle:
            if plan.get("n_blank"):
                for i in range(start, end):
                    handle.seek(int(offsets[i]))
                    line = handle.readline()
                    if not line.strip():
                        raise SystemExit("REFUSE blank line in the offset table")
                    yield from passages_of(json.loads(line), None)
                return
            handle.seek(int(offsets[start]))
            for _ in range(end - start):
                line = handle.readline()
                if not line or not line.strip():
                    raise SystemExit("REFUSE short corpus read")
                yield from passages_of(json.loads(line), None)
        return
    seen = 0
    with open(src, "rb") as handle:
        for line in handle:
            if not line.strip():
                continue
            for row in passages_of(json.loads(line), chunk):
                if seen >= end:
                    return
                if seen >= start:
                    yield row
                seen += 1
    if seen < end:
        raise SystemExit(f"REFUSE chunked corpus ended at {seen} before {end}")


def _write_progress(path, payload):
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def encode_shard(task, shard):
    disk_ok()
    t0 = time.time()
    plan = load_plan(task)
    if shard < 0 or shard >= plan["n_shards"]:
        raise SystemExit(f"REFUSE shard {shard} of {plan['n_shards']}")
    dest = index_dir(task)
    done = os.path.join(dest, "done.json")
    if os.path.isfile(done):
        print(f"SHARD_SKIP {task} {shard} index already done", flush=True)
        return
    start, end = plan["ranges"][shard]
    count = end - start
    side = os.path.join(dest, "shards", f"shard_{shard:03d}.json")
    npy = os.path.join(dest, "shards", f"shard_{shard:03d}.npy")
    partial = npy + ".partial"
    progress_path = npy + ".progress.json"
    os.makedirs(os.path.dirname(npy), exist_ok=True)
    if os.path.isfile(side) and os.path.isfile(npy):
        meta = json.load(open(side, encoding="utf-8"))
        if meta.get("start") == start and meta.get("end") == end and meta.get("rows") == count:
            arr = np.load(npy, mmap_mode="r")
            if arr.shape == (count, DIM) and arr.dtype == np.float16:
                print(f"SHARD_DONE {task} {shard} already", flush=True)
                return
    filled = 0
    if os.path.isfile(partial) and os.path.isfile(progress_path):
        try:
            meta = json.load(open(progress_path, encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
        if meta.get("task") == task and meta.get("start") == start and meta.get("end") == end:
            filled = int(meta.get("filled") or 0)
        else:
            filled = 0
        if filled == count:
            os.replace(partial, npy)
            if os.path.isfile(progress_path):
                os.remove(progress_path)
            json.dump(
                {"task": task, "shard": shard, "start": start, "end": end, "rows": count, "dtype": "float16"},
                open(side, "w"),
            )
            print(f"SHARD_DONE {task} {shard} recovered", flush=True)
            return
        if not 0 < filled < count:
            filled = 0
    if filled == 0 and os.path.isfile(partial):
        os.remove(partial)
    tokenizer, model = load_encoder()
    mode = "r+" if filled else "w+"
    out = np.lib.format.open_memmap(partial, mode=mode, dtype=np.float16, shape=(count, DIM))
    print(
        f"ENCODE_START {task} shard {shard} filled {filled}/{count} "
        f"mode bf16_sorted batch {ENCODE_BATCH}",
        flush=True,
    )
    batch = []
    base = filled
    t_encode = time.time()

    def flush_window(rows):
        nonlocal filled
        if not rows:
            return
        out[filled : filled + len(rows)] = encode_window(tokenizer, model, rows)
        filled += len(rows)
        newly = filled - base
        if newly > 0 and newly % ENCODE_WINDOW == 0:
            out.flush()
            _write_progress(progress_path, {
                "task": task, "shard": shard, "start": start, "end": end, "filled": filled, "dtype": "float16",
            })
            elapsed = time.time() - t_encode
            rate = newly / elapsed
            eta = (count - filled) / rate
            print(
                f"ENCODE {task} shard {shard} {filled}/{count} rate {rate:.1f} eta_sec {eta:.0f}",
                flush=True,
            )
            if newly >= ENCODE_WINDOW * 2 and eta > 3 * 3600:
                out.flush()
                print(
                    f"ESTIMATE_OVER_3H {task} shard {shard} eta_sec {eta:.0f} rate {rate:.1f} filled {filled}",
                    flush=True,
                )
                raise SystemExit(2)

    for row in iter_shard_docs(task, plan, shard, filled):
        batch.append(row["passage"])
        if len(batch) >= ENCODE_WINDOW:
            flush_window(batch)
            batch = []
    flush_window(batch)
    if filled != count:
        del out
        raise SystemExit(f"REFUSE encoded {filled} rows, expected {count}")
    out.flush()
    del out
    os.replace(partial, npy)
    if os.path.isfile(progress_path):
        os.remove(progress_path)
    json.dump(
        {"task": task, "shard": shard, "start": start, "end": end, "rows": count, "dtype": "float16"},
        open(side, "w"),
    )
    print(f"SHARD_DONE {task} {shard} rows {count} sec {time.time() - t0:.1f}", flush=True)


def write_docs(task, plan, dest):
    docs_path = os.path.join(dest, "docs.jsonl")
    off_path = os.path.join(dest, "offsets.npy")
    offs = []
    pos = 0
    n = 0
    with open(docs_path, "wb") as out:
        for row in iter_all_docs(task, plan):
            offs.append(pos)
            line = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
            out.write(line)
            pos += len(line)
            n += 1
    offs.append(pos)
    if n != plan["n_passages"]:
        raise SystemExit(f"REFUSE docs {n} != plan {plan['n_passages']}")
    np.save(off_path, np.asarray(offs, dtype=np.int64))
    return docs_path


def iter_all_docs(task, plan):
    for shard in range(plan["n_shards"]):
        yield from iter_shard_docs(task, plan, shard)


def vector_check(path, n):
    mm = np.load(path, mmap_mode="r")
    if mm.shape != (n, DIM) or mm.dtype != np.float16:
        raise SystemExit(f"REFUSE embeddings shape {mm.shape} dtype {mm.dtype}")
    norms = np.empty(n, dtype=np.float32)
    n_bad = 0
    n_zero = 0
    for start in range(0, n, 200000):
        block = np.asarray(mm[start : start + 200000], dtype=np.float32)
        n_bad += int((~np.isfinite(block).all(axis=1)).sum())
        n_zero += int(np.all(block == 0, axis=1).sum())
        norms[start : start + len(block)] = np.linalg.norm(block, axis=1)
    if n_bad or n_zero:
        raise SystemExit(f"REFUSE vectors nonfinite {n_bad} zero {n_zero}")
    qs = [0, 1, 50, 99, 99.9, 100]
    pct = np.percentile(norms, qs)
    stats = {
        "min": float(pct[0]),
        "p01": float(pct[1]),
        "p50": float(pct[2]),
        "p99": float(pct[3]),
        "p999": float(pct[4]),
        "max": float(pct[5]),
        "mean": float(norms.mean()),
        "n_nonfinite": n_bad,
        "n_zero_rows": n_zero,
        "normalized": False,
    }
    print(
        "NORMS min {min:.6f} p01 {p01:.6f} p50 {p50:.6f} p99 {p99:.6f} p999 {p999:.6f} max {max:.6f} mean {mean:.6f}".format(**stats),
        flush=True,
    )
    return stats


def drop_shard_matrices(dest, n_shards):
    """Drop shard matrices only. The shard json stays so the ranges remain recorded."""
    for shard in range(n_shards):
        npy = os.path.join(dest, "shards", f"shard_{shard:03d}.npy")
        for path in (npy, npy + ".partial", npy + ".progress.json", npy + ".progress.json.tmp"):
            if os.path.isfile(path):
                os.remove(path)


def merge_task(task):
    disk_ok()
    plan = load_plan(task)
    dest = index_dir(task)
    done_path = os.path.join(dest, "done.json")
    if os.path.isfile(done_path):
        done = json.load(open(done_path, encoding="utf-8"))
        if done.get("n_passages") == plan["n_passages"] and done.get("check") == "pass":
            drop_shard_matrices(dest, plan["n_shards"])
            print(f"MERGE_DONE {task} already", flush=True)
            return
    n = plan["n_passages"]
    emb_path = os.path.join(dest, "embeddings.npy")
    if os.path.isfile(emb_path):
        os.remove(emb_path)
    out = np.lib.format.open_memmap(emb_path + ".partial", mode="w+", dtype=np.float16, shape=(n, DIM))
    cursor = 0
    for shard, (start, end) in enumerate(plan["ranges"]):
        side = json.load(open(os.path.join(dest, "shards", f"shard_{shard:03d}.json"), encoding="utf-8"))
        if side["start"] != start or side["end"] != end:
            raise SystemExit(f"REFUSE shard {shard} range")
        part = np.load(os.path.join(dest, "shards", f"shard_{shard:03d}.npy"), mmap_mode="r")
        if part.shape != (end - start, DIM) or part.dtype != np.float16:
            raise SystemExit(f"REFUSE shard {shard} array {part.shape} {part.dtype}")
        out[cursor : cursor + len(part)] = part
        cursor += len(part)
    if cursor != n:
        raise SystemExit(f"REFUSE merged {cursor} != {n}")
    out.flush()
    del out
    os.replace(emb_path + ".partial", emb_path)
    docs_path = write_docs(task, plan, dest)
    stats = vector_check(emb_path, n)
    emb_sha = file_sha256(emb_path)
    corpus_sha = file_sha256(plan["corpus"])
    stamp = {
        "model": "facebook/contriever-msmarco",
        "pooling": "mean",
        "similarity": "dot",
        "normalized": False,
        "dtype": "float16",
        "max_length": 512,
        "n_docs": n,
        "dim": DIM,
        "chunk_words": plan["chunk_words"],
    }
    json.dump(stamp, open(os.path.join(dest, "contriever.json"), "w"), indent=1)
    done = {
        "task": task,
        "check": "pass",
        "n_passages": n,
        "n_expected": n,
        "dim": DIM,
        "dtype": "float16",
        "normalized": False,
        "similarity": "dot",
        "pooling": "mean",
        "max_length": 512,
        "chunk_words": plan["chunk_words"],
        "embeddings_sha256": emb_sha,
        "corpus_sha256": corpus_sha,
        "corpus": plan["corpus"],
        "embeddings_bytes": os.path.getsize(emb_path),
        "docs_bytes": os.path.getsize(docs_path),
        "norms": stats,
    }
    json.dump(done, open(done_path, "w"), indent=1)
    # Re-read before dropping shard matrices. Shard json stays for the record.
    again = json.load(open(done_path, encoding="utf-8"))
    if again.get("check") != "pass" or again.get("n_passages") != n:
        raise SystemExit("REFUSE done.json failed to stick")
    drop_shard_matrices(dest, plan["n_shards"])
    print(
        f"MERGE_DONE {task} n {n} emb_bytes {done['embeddings_bytes']} docs_bytes {done['docs_bytes']} sha {emb_sha}",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("n-shards")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--chunk", default="none")
    p = sub.add_parser("prepare")
    p.add_argument("--task", required=True)
    p.add_argument("--expect-n", type=int)
    p = sub.add_parser("shard")
    p.add_argument("--task", required=True)
    p.add_argument("--shard", type=int, required=True)
    p = sub.add_parser("merge")
    p.add_argument("--task", required=True)
    args = ap.parse_args()
    if args.cmd == "n-shards":
        chunk = None if args.chunk == "none" else int(args.chunk)
        print(n_shards_for(args.n, chunk))
        return
    if args.task not in TASKS:
        raise SystemExit(f"REFUSE task {args.task}")
    if args.cmd == "prepare":
        prepare_task(args.task, args.expect_n)
    elif args.cmd == "shard":
        encode_shard(args.task, args.shard)
    elif args.cmd == "merge":
        merge_task(args.task)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
