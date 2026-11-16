"""Build a formal-v2 CARROT index while retaining verbatim evidence text."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import faiss
import numpy as np
import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.upstream))
    from data.build_faiss_index import FAISSIndexBuilder

    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise RuntimeError("CARROT formal index output must be a fresh directory")
    builder = FAISSIndexBuilder(model_name=str(args.models / "bge-m3"), chunk_size=256,
                                batch_size=32, use_fp16=True)
    chunks = []
    corpus_path = args.corpus / "corpus.jsonl"
    document_count = 0
    for line in corpus_path.open():
        if not line.strip():
            continue
        document_count += 1
        doc = json.loads(line)
        title = doc.get("title", "") or ""
        for chunk in builder.chunk_text(doc["text"], doc["doc_id"], "document"):
            raw = chunk["text"]
            tokens = ENC.encode(raw, disallowed_special=())
            parts = [ENC.decode(tokens[start:start + 480]) for start in range(0, max(len(tokens), 1), 480)]
            for part in parts:
                if not part:
                    continue
                entry = dict(chunk)
                entry.update({"doc_id": str(doc["doc_id"]), "title": title, "evidence_text": part,
                              "text": (title + "\n" + part).strip()})
                chunks.append(entry)
    embeddings = builder.encode_chunks_dense([chunk["text"] for chunk in chunks])
    faiss.normalize_L2(embeddings)
    (args.output / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False))
    np.save(args.output / "embeddings.npy", embeddings)
    stats = {"documents": document_count, "chunks": len(chunks), "corpus_sha256": sha256(corpus_path),
             "embedding_model": "BAAI/bge-m3", "chunk_words": 256, "overlap_words": 64,
             "long_chunk_split_cl100k": 480, "evidence_field": "evidence_text",
             "model_text_field": "text"}
    (args.output / "index_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
