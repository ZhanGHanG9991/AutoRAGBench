#!/usr/bin/env python3
"""Split a formal-v1 public corpus into round-robin id files.

Pilot documents are reused only when every pilot id is in the formal corpus
and the inserted text hash matches. Ids and text are not printed.
"""
import argparse
import hashlib
import json
from pathlib import Path


def corpus_content(row: dict) -> str:
    title = (row.get("title") or "").strip()
    text = row.get("text") or ""
    return f"{title}\n\n{text}" if title else text


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--pilot-storage", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--nshards", type=int, required=True)
    parser.add_argument("--expect-formal", type=int, required=True)
    parser.add_argument("--expect-pilot", type=int, required=True)
    args = parser.parse_args()
    if args.nshards < 1:
        raise SystemExit("BAD_NSHARDS")
    pilot_path = Path(args.pilot_storage) / "kv_store_full_docs.json"
    pilot_rows = json.loads(pilot_path.read_text())
    pilot_hash = {}
    for doc_id, row in pilot_rows.items():
        if not isinstance(row, dict):
            continue
        pilot_hash[str(doc_id)] = digest(str(row.get("content") or ""))
    found = 0
    mismatch = 0
    formal_ids = []
    seen = set()
    with Path(args.corpus).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            content = corpus_content(row)
            if not content.strip():
                continue
            doc_id = str(row["doc_id"])
            if doc_id in seen:
                raise SystemExit("DUP_DOC")
            seen.add(doc_id)
            formal_ids.append(doc_id)
            if doc_id in pilot_hash:
                found += 1
                if digest(content) != pilot_hash[doc_id]:
                    mismatch += 1
    missing = len(pilot_hash) - found
    reuse = missing == 0 and mismatch == 0 and found == len(pilot_hash)
    if formal_ids and args.expect_formal and len(formal_ids) != args.expect_formal:
        print(
            f"FORMAL_COUNT got={len(formal_ids)} expect={args.expect_formal}",
            flush=True,
        )
        raise SystemExit(6)
    if len(pilot_hash) != args.expect_pilot:
        print(
            f"PILOT_COUNT got={len(pilot_hash)} expect={args.expect_pilot}",
            flush=True,
        )
        raise SystemExit(6)
    pilot_set = set(pilot_hash)
    chosen = [doc_id for doc_id in formal_ids if not reuse or doc_id not in pilot_set]
    chosen.sort()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    counts = []
    for shard in range(args.nshards):
        ids = chosen[shard :: args.nshards]
        shard_dir = out / f"shard{shard}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        (shard_dir / "ids.txt").write_text("".join(f"{doc_id}\n" for doc_id in ids))
        counts.append(len(ids))
    manifest = {
        "reuse": reuse,
        "pilot_n": len(pilot_hash),
        "formal_n": len(formal_ids),
        "found_in_formal": found,
        "missing_from_formal": missing,
        "text_mismatch": mismatch,
        "index_n": len(chosen),
        "nshards": args.nshards,
        "shard_counts": counts,
    }
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("SPLIT", json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
