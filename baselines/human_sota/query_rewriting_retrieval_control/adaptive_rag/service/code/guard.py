"""Path checks. Train and dev only. Never hidden or test."""
import os

TASKS = (
    "musique",
    "hotpotqa",
    "multihop_rag",
    "nq_open",
    "fever",
    "pubmedqa",
    "qasper",
)
MULTI_HOP = {"musique", "hotpotqa", "multihop_rag"}


def bench_root():
    root = os.environ.get("RAGENG_ROOT", "")
    if not root:
        raise SystemExit("RAGENG_ROOT is not set")
    return root


def arag_root():
    root = os.environ.get("ARAG_ROOT", "")
    if not root:
        raise SystemExit("ARAG_ROOT is not set")
    return root


def assert_task(task):
    if task not in TASKS:
        raise SystemExit(f"refuse task {task!r}; this run only covers {TASKS}")
    return task


def assert_public(path, task, split):
    """split is 'train' or 'dev'. Reject anything that looks like test or hidden."""
    if split not in ("train", "dev"):
        raise SystemExit(f"refuse split {split}")
    text = os.path.abspath(path)
    low = text.lower()
    if "hidden" in low or f"/{split}" not in text.replace("\\", "/"):
        raise SystemExit(f"refuse path {path}")
    # A public dev/train file may live under data/<task>/public/.
    # Block test files and the hidden store even if a caller also says dev.
    parts = text.replace("\\", "/").split("/")
    if "test" in parts or "hidden" in parts:
        raise SystemExit(f"refuse test/hidden path {path}")
    if f"/{task}/public/{split}.jsonl" not in text.replace("\\", "/"):
        raise SystemExit(f"refuse unexpected split file {path}")
    if not os.path.isfile(path):
        raise SystemExit(f"missing {path}")
    return text


def public_jsonl(task, split):
    path = os.path.join(bench_root(), "data", task, "public", f"{split}.jsonl")
    return assert_public(path, task, split)


def corpus_dir(task):
    path = os.path.join(bench_root(), "data", task, "public", "corpus")
    if not os.path.isdir(path):
        raise SystemExit(f"missing corpus {path}")
    return path


def info(task):
    import json
    path = os.path.join(bench_root(), "tasks", task, "info.json")
    with open(path) as f:
        return json.load(f)


def bias_label(task):
    return "C" if task in MULTI_HOP else "B"


def assert_no_secret(path):
    """Reject test and hidden path components. Does not open the file."""
    text = os.path.abspath(path).replace("\\", "/")
    parts = text.split("/")
    if "test" in parts or "hidden" in parts or "eval_test" in parts:
        raise SystemExit(f"refuse test/hidden path {path}")
    return text


def formal_jsonl(task, kind):
    """formal-v1-split train_fit or dev only. Never test."""
    assert_task(task)
    if kind not in ("train_fit", "dev"):
        raise SystemExit(f"refuse formal kind {kind}")
    path = os.path.join(
        bench_root(), "humansota", "formal-v1-split", task, f"{kind}.jsonl"
    )
    text = assert_no_secret(path)
    suffix = f"/humansota/formal-v1-split/{task}/{kind}.jsonl"
    if not text.endswith(suffix):
        raise SystemExit(f"refuse unexpected formal file {path}")
    if not os.path.isfile(text):
        raise SystemExit(f"missing {text}")
    return text


def formal_expect(task, kind):
    assert_task(task)
    if kind == "train_fit":
        return 400 if task == "pubmedqa" else 1800
    if kind == "dev":
        return 100 if task == "pubmedqa" else 200
    raise SystemExit(f"refuse formal kind {kind}")


def formal_index(task):
    """Read-only naive_bm25 hardlink. Missing built.json is a refusal, not a build."""
    assert_task(task)
    path = os.path.join(
        bench_root(), "humansota", "shared-indexes", "formal-v1", f"{task}-naive_bm25"
    )
    text = assert_no_secret(path)
    suffix = f"/shared-indexes/formal-v1/{task}-naive_bm25"
    if not text.endswith(suffix):
        raise SystemExit(f"refuse index {path}")
    if not os.path.isdir(text):
        raise SystemExit(f"missing index {text}")
    if not os.path.isfile(os.path.join(text, "built.json")):
        raise SystemExit(f"refuse missing built.json {text}")
    if os.path.exists(os.path.join(text, "emb.npy")):
        raise SystemExit(f"refuse emb.npy in naive index {text}")
    return text


def refuse_index_build():
    import ragcore

    def _refuse(*_args, **_kwargs):
        raise SystemExit("REFUSE_BUILD")

    ragcore.build = _refuse
