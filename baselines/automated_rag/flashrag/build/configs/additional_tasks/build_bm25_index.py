"""Build a bm25s index for one benchmark. Run this inside a job, not on the login node.

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.bench not in {"musique", "multihop_rag", "qasper", "pubmedqa", "hotpotqa", "fever", "cwq", "webqsp"}:  # exp1109-night: webqsp added (KG entity docs as text, like cwq)
        raise SystemExit("bench is not one of the six")
    import os
    if not os.environ.get("SLURM_JOB_ID"): raise SystemExit("Slurm allocation required")
    corpus_path = Path(os.environ["EXP3_CORPUS"])
    if any(part in {"hidden", "test", "test_questions", "test_gold"} for part in corpus_path.parts):
        raise SystemExit("refusing a hidden or test path")
    import bm25s

    ids = []
    texts = []
    with corpus_path.open(encoding="utf-8") as handle:
        for line in handle:
            if args.limit and len(ids) >= args.limit:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            ids.append(str(row["doc_id"]))
            texts.append(row["text"])
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize(texts))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    retriever.save(str(out), corpus=[{"doc_id": i, "text": t} for i, t in zip(ids, texts)])
    (out / "doc_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    print(f"indexed {len(ids)} documents")


if __name__ == "__main__":
    main()
