#!/usr/bin/env python3
"""Split each formal-v1 train_fit into evaluator-train and threshold-tune qids.

Seed 20261019, reseeded per task. random.Random.shuffle, then the first tune
slice is the holdout and the rest trains the evaluator. pubmedqa is 300/100.
The other six tasks are 1600/200. Does not read dev or hidden test.
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

import hashlib
import json
import os
import random
import sys

ROOT = _release_path("@AUTORAGBENCH_ROOT@")
SPLIT_SRC = os.path.join(ROOT, "humansota/formal-v1-split")
OUT = os.path.join(ROOT, "humansota/CorrectiveRAG/rageng/formal-v1/split")
SEED = 20261019
SPEC = {
    "musique": (1600, 200),
    "hotpotqa": (1600, 200),
    "multihop_rag": (1600, 200),
    "nq_open": (1600, 200),
    "fever": (1600, 200),
    "pubmedqa": (300, 100),
    "qasper": (1600, 200),
}


def refuse(path):
    parts = os.path.abspath(path).replace("\\", "/").split("/")
    if "hidden" in parts or "test" in parts:
        raise SystemExit(f"REFUSE {path}")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def write_qids(path, qids):
    body = ("".join(qid + "\n" for qid in qids)).encode("utf-8")
    with open(path, "wb") as handle:
        handle.write(body)
    return sha256_bytes(body)


def disk_ok():
    line = os.popen(_release_path("df -BG --output=avail @BUILD_TMP@ | tail -1")).read()
    digits = "".join(ch for ch in line if ch.isdigit())
    if not digits:
        raise SystemExit("REFUSE disk check failed")
    avail = int(digits)
    print(f"DISK_AVAIL_GB {avail}", flush=True)
    if avail < 200:
        raise SystemExit(f"DISK_LOW {avail}")


# Flat per-class minimum for threshold tuning. configuration.
MIN_COUNT = 10


def main():
    disk_ok()
    os.makedirs(OUT, exist_ok=True)
    manifest = {
        "seed": SEED,
        "python": sys.version,
        "rule": "Random(20261019).shuffle per task; first tune_n are tune; the rest train the evaluator",
        "min_count_rule": "flat 10 for every task (configuration); was 10 percent of tune",
        "tasks": {},
    }
    for task, (n_train, n_tune) in SPEC.items():
        src = os.path.join(SPLIT_SRC, task, "train_fit.jsonl")
        refuse(src)
        if not src.endswith(f"/formal-v1-split/{task}/train_fit.jsonl"):
            raise SystemExit(f"REFUSE {src}")
        rows = [json.loads(line) for line in open(src, encoding="utf-8") if line.strip()]
        if len(rows) != n_train + n_tune:
            raise SystemExit(f"{task} train_fit has {len(rows)} rows, expected {n_train + n_tune}")
        qids = [row["qid"] for row in rows]
        if len(set(qids)) != len(qids):
            raise SystemExit(f"{task} duplicate qid")
        task_dir = os.path.join(OUT, task)
        os.makedirs(task_dir, exist_ok=True)
        train_path = os.path.join(task_dir, "evaluator_train_qids.txt")
        tune_path = os.path.join(task_dir, "tune_qids.txt")
        if os.path.isfile(train_path) and os.path.isfile(tune_path):
            train_q = [line.strip() for line in open(train_path, encoding="utf-8") if line.strip()]
            tune_q = [line.strip() for line in open(tune_path, encoding="utf-8") if line.strip()]
            if len(train_q) != n_train or len(tune_q) != n_tune:
                raise SystemExit(f"REFUSE existing split counts for {task}")
            if set(train_q) & set(tune_q) or set(train_q) | set(tune_q) != set(qids):
                raise SystemExit(f"REFUSE existing split qids for {task}")
        else:
            order = list(rows)
            random.Random(SEED).shuffle(order)
            tune_q = [row["qid"] for row in order[:n_tune]]
            train_q = [row["qid"] for row in order[n_tune:]]
            if set(train_q) & set(tune_q) or set(train_q) | set(tune_q) != set(qids):
                raise SystemExit(f"REFUSE split identity {task}")
            write_qids(train_path, train_q)
            write_qids(tune_path, tune_q)
        manifest["tasks"][task] = {
            "evaluator_train": n_train,
            "tune": n_tune,
            "min_count": MIN_COUNT,
            "evaluator_train_sha256": sha256_bytes(open(train_path, "rb").read()),
            "tune_sha256": sha256_bytes(open(tune_path, "rb").read()),
            "train_fit": src,
        }
        print(f"SPLIT {task} train {n_train} tune {n_tune} min_count {MIN_COUNT}", flush=True)
    out = os.path.join(OUT, "manifest.json")
    json.dump(manifest, open(out, "w"), indent=1)
    print("SPLIT_DONE", out, flush=True)


if __name__ == "__main__":
    main()
