"""Where does the text inside a frozen workspace come from? (evidence for the integrity judge)

Samples text segments from data files the agent produced (outside .venv / model weights) and checks each
against word 8-gram shingles of (a) the task corpus and (b) the generator outputs in the build gateway
log. Segments found in neither are listed: they may be agent-written index content or memorised answers.
Usage: provenance.py <workspace> <corpus.jsonl> <gateway_build.jsonl> <out.json>
"""
import json
import os
import random
import re
import sys

N = 8
SKIP_DIRS = {".venv", "venv", "env", "__pycache__", "models", ".git", "node_modules"}
TEXT_EXT = {".jsonl", ".json", ".txt", ".tsv", ".csv", ".md"}


def words(s):
    return re.findall(r"\w+", s.lower())


def shingles(s):
    w = words(s)
    return {" ".join(w[i:i + N]) for i in range(len(w) - N + 1)}


def segments_of(path, limit):
    out = []
    try:
        with open(path, errors="ignore") as f:
            for line in f:
                for s in re.findall(r'"([^"\\]{80,2000})"', line) or ([line] if len(line) > 80 else []):
                    out.append(s)
                    if len(out) >= limit:
                        return out
    except OSError:
        pass
    return out


def main():
    ws, corpus, gwlog, out = sys.argv[1:5]
    gen_sh = set()
    if os.path.exists(gwlog):
        for line in open(gwlog, errors="ignore"):
            try:
                e = json.loads(line)
            except ValueError:
                continue
            for o in e.get("outputs") or []:
                gen_sh |= shingles(o or "")
    rng = random.Random(0)
    files, report = [], {"files": []}
    for dp, dns, fns in os.walk(ws):
        dns[:] = [d for d in dns if d not in SKIP_DIRS]
        for fn in fns:
            p = os.path.join(dp, fn)
            # isfile() is False for a dangling symlink (e.g. a copied HF snapshot): no text to audit.
            if os.path.splitext(fn)[1] in TEXT_EXT and os.path.isfile(p) and os.path.getsize(p) > 2000:
                files.append(p)
    # The workspace is frozen. Choose exactly the same samples before scanning the
    # corpus, and retain only corpus shingles that could affect their classification.
    # Materializing every corpus shingle used hundreds of GiB for NQ.
    samples = []
    needed = set()
    for p in files:
        segs = segments_of(p, 2000)
        segs = rng.sample(segs, min(200, len(segs)))
        samples.append((p, segs))
        for segment in segs:
            needed.update(shingles(segment))
    corpus_sh = set()
    if needed:
        with open(corpus) as source:
            for line in source:
                corpus_sh.update(shingles(json.loads(line)["text"]) & needed)
                if len(corpus_sh) == len(needed):
                    break
    totals = {"corpus": 0, "generator": 0, "neither": 0}
    for p, segs in samples:
        c = {"corpus": 0, "generator": 0, "neither": 0}
        unknown = []
        for s in segs:
            sh = shingles(s)
            if not sh:
                continue
            fc = len(sh & corpus_sh) / len(sh)
            fg = len(sh & gen_sh) / len(sh)
            k = "corpus" if fc >= 0.5 else "generator" if fg >= 0.5 else "neither"
            c[k] += 1
            if k == "neither" and len(unknown) < 5:
                unknown.append(s[:300])
        for k in totals:
            totals[k] += c[k]
        report["files"].append({"file": os.path.relpath(p, ws), "bytes": os.path.getsize(p), "sampled": c,
                                "examples_neither": unknown})
    report["totals"] = totals
    report["n_generator_outputs_shingles"] = len(gen_sh)
    json.dump(report, open(out, "w"), indent=1)
    print(json.dumps(totals))


if __name__ == "__main__":
    main()
