"""Build task instances (Slurm CPU job or login node).

  python -m tasks.prepare --tasks all --root <RAGENG_ROOT> --out <DATA_DIR> --hidden <HIDDEN_DIR>/<version> \
         --train 2000 --test 500 --seed 20261015

Writes <DATA_DIR>/<task>/public (corpus + train) and moves the test split (private/) into the hidden store
<HIDDEN_DIR>/<version>/<task>/, read-only. The hidden store is never bound into any sandbox (agent, self_check,
service, judge), so the test can stay on the cluster and every run job evaluates right after its build.
Raw benchmark files are only read.
"""
import argparse
import json
import os
import shutil
import stat
import time
from pathlib import Path

from tasks import registry
from tasks.common import sha256_file


def lock_down(d):
    for p in sorted(Path(d).rglob("*"), reverse=True) + [Path(d)]:
        p.chmod(stat.S_IRUSR | (stat.S_IXUSR if p.is_dir() else 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden", required=True, help="hidden store of this data version")
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--test", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--rules", default="{}", help='JSON {task: {MODULE_CONSTANT: value}} overriding the sampling '
                    'rules of a task module (e.g. {"musique": {"TRAIN_MIX": "natural", "TRAIN_CAP": null}}); '
                    'recorded in the data manifest')
    a = ap.parse_args()
    rules = json.loads(Path(a.rules).read_text() if os.path.isfile(a.rules) else a.rules)
    rules = {k: v for k, v in rules.items() if not k.startswith("_")}
    names = registry.TASKS if a.tasks == "all" else a.tasks.split(",")
    hidden_root = Path(a.hidden)
    hidden_root.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(hidden_root.parent, 0o700)
    hidden_root.mkdir(exist_ok=True)
    os.chmod(hidden_root, 0o700)
    for name in names:
        t0 = time.time()
        out, hid = Path(a.out) / name, hidden_root / name
        for d in (out, hid):
            if d.exists():
                raise SystemExit(f"{d} exists: a data version is written once")
        mod = registry.load(name)
        task_rules = dict(rules.get(name, {}))
        n_train, n_test = task_rules.pop("N_TRAIN", a.train), task_rules.pop("N_TEST", a.test)  # per-task sizes
        for k, v in task_rules.items():
            if not hasattr(mod, k):
                raise SystemExit(f"{name}: unknown rule {k}")
            setattr(mod, k, v)
        m = mod.prepare(a.root, str(out), n_train, n_test, a.seed)
        pub = {"task": name, "version": hidden_root.name, "seed": a.seed, "n_corpus": m["n_corpus"],
               "rules": rules.get(name, {}), "n_requested": {"train": n_train, "test": n_test},
               "n_train": m["n"]["train"], "corpus_sha256": sha256_file(out / "public/corpus/corpus.jsonl"),
               "train_sha256": sha256_file(out / "public/train.jsonl"), "prepared": time.strftime("%FT%T")}
        (out / "public/data_manifest.json").write_text(json.dumps(pub, indent=1))
        shutil.move(str(out / "private"), str(hid))
        hm = {f.name: sha256_file(f) for f in sorted(hid.iterdir())}
        (hid / "hidden_manifest.json").write_text(json.dumps(hm, indent=1))
        lock_down(hid)
        for p in (out / "public").rglob("*"):
            p.chmod(p.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        print(json.dumps({"task": name, "seconds": round(time.time() - t0, 1),
                          **{k: v for k, v in m.items() if k != "source_sha256"}}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
