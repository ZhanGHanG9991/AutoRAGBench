"""Task paths for the train-only T5-large evaluator. No retrieval and no training."""

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

import os

ROOT = _release_path("@AUTORAGBENCH_ROOT@")
C = os.path.join(ROOT, "humansota/CorrectiveRAG")
SPLIT = os.path.join(C, "rageng/results/evaluator_split_20261018.json")
PAIRS = os.path.join(C, "rageng/results/evaluator_rageng_pairs.jsonl")
PAIR_STATS = os.path.join(C, "rageng/results/evaluator_rageng_pair_stats.json")
STATUS = os.path.join(C, "rageng/results/evaluator_rageng_status.json")
MODEL_ROOT = os.path.join(C, "models/evaluator-rageng")
AUTHOR_EVALUATOR = os.path.join(C, "models/evaluator")
T5_BASE = os.path.join(ROOT, "humansota/AdaptiveRAG/models/t5-large")
CONTRIEVER = os.path.join(C, "models/contriever-msmarco")
SEED = 20261018
FILTER_VERSION = "gold_window_v1"
# Upper median of per-article answer-containment counts, and the nearest-rank
# 25th percentile of overlap on those windows. Both come from the 400 only.
EXPECTED_M = 2
EXPECTED_TAU = 0.08108108108108109
OLD_POSITIVES = {
    "musique": 568,
    "hotpotqa": 663,
    "nq_open": 1030,
    "fever": 257,
    "pubmedqa": 391,
    "multihop_rag": 1646,
}
CHECKPOINT = os.path.join(MODEL_ROOT, "resume.pt")
BOUNDARY = os.path.join(C, "rageng/results/evaluator_rageng_resume_boundary.json")

# qasper is scored at dev time but its questions are not training pairs.
TRAIN_TASKS = ["musique", "hotpotqa", "nq_open", "fever", "pubmedqa", "multihop_rag"]
DEV_TASKS = TRAIN_TASKS + ["qasper"]

INDEX_NAME = {
    "musique": "musique-contriever",
    "hotpotqa": "hotpotqa-contriever",
    "nq_open": "nq_open-contriever",
    "fever": "fever-contriever",
    "pubmedqa": "pubmedqa-contriever",
    "multihop_rag": "multihop_rag-contriever-w200",
    "qasper": "qasper-contriever-w200",
}
CHUNK_WORDS = {
    "musique": None,
    "hotpotqa": None,
    "nq_open": None,
    "fever": None,
    "pubmedqa": None,
    "multihop_rag": 200,
    "qasper": 200,
}


def refuse(path):
    if "hidden" in path or "/test/" in path or path.rstrip("/").endswith("test.jsonl"):
        raise SystemExit(f"refusing {path}")


def index_dir(task):
    if task not in INDEX_NAME:
        raise SystemExit(f"refusing task {task}")
    path = os.path.join(C, "rageng/index", INDEX_NAME[task])
    refuse(path)
    return path


def train_path(task):
    path = os.path.join(ROOT, "data", task, "public", "train.jsonl")
    refuse(path)
    if not path.endswith("/public/train.jsonl"):
        raise SystemExit(f"refusing non-public train file {path}")
    return path


def dev_path(task):
    path = os.path.join(ROOT, "data", task, "public", "dev.jsonl")
    refuse(path)
    if not path.endswith("/public/dev.jsonl"):
        raise SystemExit(f"refusing non-public dev file {path}")
    return path
