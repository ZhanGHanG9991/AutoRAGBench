"""SubgraphRAG formal-v2 CWQ training sample (configuration): 2,000 questions from
data_formal-v2/cwq/public/train.jsonl, seed 20261022 -> train_fit 1,800 + early-stop holdout 200 (never dev).
Same draw as make_split.py: sort by qid, random.Random(f"{SEED}-cwq-sgrag") shuffle; first 200 = holdout.
Lines copied verbatim. Output humansota/formal-v2-split/cwq-sgrag-train2000/."""

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

import hashlib, json, os, random
ROOT = _release_path("@AUTORAGBENCH_ROOT@")
OUT = f"{ROOT}/humansota/formal-v2-split/cwq-sgrag-train2000"
SEED = 20261022
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
src = f"{ROOT}/data_formal-v2/cwq/public/train.jsonl"
lines = [l if l.endswith("\n") else l + "\n" for l in open(src, encoding="utf-8") if l.strip()]
R = [json.loads(l) for l in lines]
idx = sorted(range(len(R)), key=lambda i: R[i]["qid"])
random.Random(f"{SEED}-cwq-sgrag").shuffle(idx)
hold, fit = sorted(idx[:200]), sorted(idx[200:2000])
os.makedirs(OUT, exist_ok=False)
man = {"seed": SEED, "source": "data_formal-v2/cwq/public/train.jsonl", "source_sha256": sha(src), "n_source": len(R),
       "decided": "configuration", "script": "make_sgrag_train_v2.py", "files": {}}
for name, sel in (("train_fit", fit), ("holdout", hold)):
    p, pq = f"{OUT}/{name}.jsonl", f"{OUT}/{name}_qids.txt"
    with open(p, "w", encoding="utf-8") as f:
        for i in sel: f.write(lines[i])
    with open(pq, "w") as f:
        for i in sel: f.write(R[i]["qid"] + "\n")
    man["files"][name] = {"n": len(sel), "jsonl_sha256": sha(p), "qids_sha256": sha(pq)}
man["script_sha256"] = sha(os.path.abspath(__file__))
json.dump(man, open(f"{OUT}/manifest.json", "w"), indent=1)
print(json.dumps({k: v["n"] for k, v in man["files"].items()}))
