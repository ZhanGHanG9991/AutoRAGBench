"""formal-v1 fixed dev split. configuration. Reads only data_formal-v1/<task>/public/train.jsonl."""

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

import json, hashlib, random, os, collections, sys
ROOT = _release_path("@AUTORAGBENCH_ROOT@")
SRC = f"{ROOT}/data_formal-v1"
OUT = f"{ROOT}/humansota/formal-v1-split"
SEED = 20261019
TASKS = ["musique", "hotpotqa", "multihop_rag", "nq_open", "fever", "pubmedqa", "qasper", "webqsp", "cwq"]
DEV_N = {"pubmedqa": 100}
def strat_key(t, q):
    if t in ("fever", "pubmedqa"): return str(q["answers"][0]) if q.get("answers") else ""
    if t == "multihop_rag": return str(q.get("question_type"))
    if t == "qasper": return str(q.get("answer_type"))
    return "all"
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
man = {"seed": SEED, "decided": "configuration", "script": "make_split.py", "tasks": {}}
for t in TASKS:
    src = f"{SRC}/{t}/public/train.jsonl"
    lines = [l for l in open(src, encoding="utf-8") if l.strip()]
    Q = [json.loads(l) for l in lines]
    assert len(set(q["qid"] for q in Q)) == len(Q)
    n_dev = DEV_N.get(t, 200)
    groups = collections.defaultdict(list)
    for i, q in enumerate(Q): groups[strat_key(t, q)].append(i)
    keys = sorted(groups)
    # proportional allocation, largest remainder, ties by key name
    raw = {k: n_dev * len(groups[k]) / len(Q) for k in keys}
    alloc = {k: int(raw[k]) for k in keys}
    rest = n_dev - sum(alloc.values())
    for k in sorted(keys, key=lambda k: (-(raw[k] - alloc[k]), k))[:rest]: alloc[k] += 1
    rng = random.Random(f"{SEED}-{t}")
    dev = set()
    for k in keys:
        idx = sorted(groups[k], key=lambda i: Q[i]["qid"])
        rng.shuffle(idx)
        dev.update(idx[:alloc[k]])
    os.makedirs(f"{OUT}/{t}", exist_ok=True)
    files = {}
    for name, sel in (("train_fit", [i for i in range(len(Q)) if i not in dev]), ("dev", [i for i in range(len(Q)) if i in dev])):
        p = f"{OUT}/{t}/{name}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for i in sel: f.write(lines[i] if lines[i].endswith("\n") else lines[i] + "\n")
        pq = f"{OUT}/{t}/{name}_qids.txt"
        with open(pq, "w") as f:
            for i in sel: f.write(Q[i]["qid"] + "\n")
        files[name] = {"n": len(sel), "jsonl": f"{t}/{name}.jsonl", "jsonl_sha256": sha(p), "qids": f"{t}/{name}_qids.txt", "qids_sha256": sha(pq)}
    man["tasks"][t] = {"source": f"data_formal-v1/{t}/public/train.jsonl", "source_sha256": sha(src), "n_source": len(Q),
                       "stratify_by": {"fever": "answer label", "pubmedqa": "answer label", "multihop_rag": "question_type", "qasper": "answer_type"}.get(t, "none (random)"),
                       "strata": {k: {"total": len(groups[k]), "dev": alloc[k]} for k in keys}, **files}
    print(t, len(Q), files["train_fit"]["n"], files["dev"]["n"], {k: alloc[k] for k in keys} if len(keys) > 1 else "")
man["script_sha256"] = sha(os.path.abspath(__file__))
json.dump(man, open(f"{OUT}/split_manifest.json", "w"), ensure_ascii=False, indent=1)
