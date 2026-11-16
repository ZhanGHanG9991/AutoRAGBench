"""Leak scan (cluster side, outside every sandbox): does hidden test question text appear in the given trees?

  leak_scan.py <test_questions.jsonl> <train.jsonl> <dir> [<dir> ...] > out.json

Probes are whole test questions (every 20th question of at least 30 characters); a question whose text is also a
train question (QASPER's generic questions on other papers) cannot tell a leak from the agent's own train split,
so it is not a probe. Output lists only file paths, never question text. Used twice by slurm/run.sbatch: before
the judge gate on what the agent wrote (leak_scan_build.json, judge evidence) and after the evaluation on the run
and its records (leak_scan.json). A hit inside a corpus-derived file can be a coincidence -> review, not an
automatic penalty.
"""
import json
import os
import sys

SKIP = {".venv", "models", "pylib", "__pycache__"}


def main():
    train = {json.loads(l)["question"] for l in open(sys.argv[2])}
    qs = [json.loads(l)["question"] for l in open(sys.argv[1])]
    probes = [q for q in qs if len(q) >= 30 and q not in train][::20]
    hits = []
    for root in sys.argv[3:]:
        for dp, dns, fns in os.walk(root):
            dns[:] = [d for d in dns if d not in SKIP]
            for fn in fns:
                p = os.path.join(dp, fn)
                try:
                    if os.path.getsize(p) > 2 << 30:
                        continue
                    data = open(p, "rb").read()
                except OSError:
                    continue
                hits += [p for q in probes if q.encode() in data][:1]
    print(json.dumps({"n_probes": len(probes), "hits": sorted(set(hits)), "roots": sys.argv[3:],
                      "skipped_dirs": sorted(SKIP)}))


if __name__ == "__main__":
    main()
