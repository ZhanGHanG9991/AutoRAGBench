"""Best-of-N judges (judges/judges.json "votes" > 1): every vote is a separate judge session writing
records/<run>/judgement_<judge>.vote<k>.json; the verdict judgement_<judge>.json takes the per-field majority.

  votes.py <records_dir> resolve <judge> [...]  write judgement_<judge>.json for the named voted judges (only
                                                these: verdicts of other judges, e.g. of older runs, stay untouched)
  votes.py <records_dir> sessions <judge> [...] print all sessions ("<judge>" or "<judge>@<k>") of the named judges
  votes.py <records_dir> units <judge> [...]    print the sessions of the named judges that still lack a complete
                                                verdict (space separated)

A vote is complete when all its boolean fields (the judge's "field" and "review_fields") are booleans. A field
whose valid votes have no strict majority of all N votes (e.g. 1 true, 1 false, 1 missing) gets no verdict, so
judges/gate.py reports the judge as missing and its incomplete votes run again.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JUDGES = {k: v for k, v in json.loads((ROOT / "judges" / "judges.json").read_text()).items() if not k.startswith("_")}


def fields(spec):
    return [spec["field"]] + list(spec.get("review_fields", []))


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return {}


def complete(v, spec):
    return all(isinstance(v.get(f), bool) for f in fields(spec))


def resolve(rec, names):
    for j in names:
        spec = JUDGES[j]
        n = spec.get("votes", 1)
        if n <= 1:
            continue
        votes = [load(rec / f"judgement_{j}.vote{k}.json") for k in range(1, n + 1)]
        out = {"votes": [{f: v.get(f) for f in fields(spec)} for v in votes]}
        for f in fields(spec):
            vals = [v.get(f) for v in votes if isinstance(v.get(f), bool)]
            for b in (True, False):
                if vals.count(b) * 2 > n:
                    out[f] = b
                    # justification of the first vote that agrees with the majority
                    out["justification_" + f] = next(v.get("justification_" + f, "") for v in votes if v.get(f) is b)
        if spec["field"] not in out:
            out["error"] = f"no majority among {n} votes"
        (rec / f"judgement_{j}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


def sessions(names):
    print(" ".join(j if JUDGES[j].get("votes", 1) <= 1 else " ".join(f"{j}@{k}" for k in range(1, JUDGES[j]["votes"] + 1))
                   for j in names))


def units(rec, names):
    todo = []
    for j in names:
        spec = JUDGES[j]
        n = spec.get("votes", 1)
        if n <= 1:
            todo += [] if complete(load(rec / f"judgement_{j}.json"), spec) else [j]
        else:
            todo += [f"{j}@{k}" for k in range(1, n + 1) if not complete(load(rec / f"judgement_{j}.vote{k}.json"), spec)]
    print(" ".join(todo))


def main():
    rec, cmd = Path(sys.argv[1]), sys.argv[2]
    if cmd == "resolve":
        resolve(rec, sys.argv[3:])
    elif cmd == "sessions":
        sessions(sys.argv[3:])
    elif cmd == "units":
        units(rec, sys.argv[3:])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
