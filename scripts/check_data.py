"""Check a local copy of the formal-v2 data against the released checksums.

  python scripts/check_data.py            # every file present, sizes and record counts of the small files
  python scripts/check_data.py --sha256   # also sha256 of every file (reads ~22 GB)

Checksums: configs/data_rules/formal-v2.public.sha256 (every public file) and
configs/data_rules/formal-v2.frozen.json (corpus / train / test fingerprints and counts frozen before the
experiments). The test set (hidden/formal-v2/<task>/, a separate download) is always hashed; it is small.
"""
import argparse
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).absolute().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/formal-v2"))
    ap.add_argument("--sha256", action="store_true", help="hash every file (slow for the large corpora)")
    ap.add_argument("--hidden", default=str(ROOT / "hidden/formal-v2"), help="test set directory")
    a = ap.parse_args()
    data, hidden = pathlib.Path(a.data), pathlib.Path(a.hidden)
    counts = json.loads((ROOT / "configs/data_rules/formal-v2.test_counts.json").read_text())
    frozen = json.loads((ROOT / "configs/data_rules/formal-v2.frozen.json").read_text())["tasks"]
    sums = dict(reversed(l.split()) for l in (ROOT / "configs/data_rules/formal-v2.public.sha256").read_text().splitlines() if l.strip())
    bad = 0
    for task, fz in frozen.items():
        pub = data / task / "public"
        line = [task]
        for rel in sorted(k for k in sums if k.startswith(task + "/")):
            p = data / rel
            if not p.exists():
                line.append(f"MISSING {rel}")
                bad += 1
                continue
            if a.sha256 and sha256(p) != sums[rel]:
                line.append(f"SHA256 MISMATCH {rel}")
                bad += 1
        if (pub / "train.jsonl").exists():
            n = sum(1 for _ in open(pub / "train.jsonl", "rb"))
            line.append(f"train={n}")
            if n != fz["n_train"]:
                line.append(f"(expected {fz['n_train']})")
                bad += 1
        if (pub / "data_manifest.json").exists():
            m = json.loads((pub / "data_manifest.json").read_text())
            ok = m["corpus_sha256"] == fz["corpus_sha256"] and m["n_corpus"] == fz["n_corpus"]
            line.append(f"corpus={m['n_corpus']} manifest={'ok' if ok else 'MISMATCH'}")
            bad += not ok
        hid = hidden / task
        if (hid / "test_questions.jsonl").exists():
            for name, key in (("test_questions.jsonl", "test_questions_sha256"), ("test_gold.jsonl", "test_gold_sha256"),
                              ("split_manifest.json", "split_manifest_sha256")):
                if not (hid / name).exists() or sha256(hid / name) != fz[key]:
                    line.append(f"TEST MISMATCH {name}")
                    bad += 1
            n = sum(1 for _ in open(hid / "test_questions.jsonl", "rb"))
            line.append(f"test={n}")
            if n != counts[task]:
                line.append(f"(expected {counts[task]})")
                bad += 1
        else:
            line.append("test=absent")
        print("  ".join(line))
    print("OK" if not bad else f"{bad} problem(s)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
