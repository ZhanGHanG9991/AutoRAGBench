"""Judge gate of a formal agent run: the hidden test is evaluated only if every judge in judges/judges.json returned a
clean verdict. Runs on the cluster between freeze and evaluation (slurm/run.sbatch).

  gate.py <records_dir>            write <records_dir>/judge_gate.json; exit 0 passed, 1 flagged, 2 incomplete
  gate.py <records_dir> --missing  print the judges without a verdict (space separated), write nothing

A verdict is judgement_<judge>.json[<field>]: false = clean, true = flagged, anything else = no verdict. A judge's
"review_fields" (e.g. data_lookup.disallowed_benchmark_lookup) are gated the same way under the name
"<judge>.<field>"; a judge with a missing review field counts as without a verdict. For best-of-N judges
judgement_<judge>.json is the majority written by judges/votes.py.
A human can overrule a judge with <records_dir>/judge_override.json: {"<judge>": false, "by": "...", "why": "..."}
(a review field: {"<judge>.<field>": false, ...}).
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JUDGES = {k: v for k, v in json.loads((ROOT / "judges" / "judges.json").read_text()).items() if not k.startswith("_")}


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return {}


def verdicts(rec):
    """{name: True | False | None} for every gated verdict: "<judge>" and "<judge>.<review field>"."""
    out = {}
    for j, spec in JUDGES.items():
        d = load(rec / f"judgement_{j}.json")
        for name, f in [(j, spec["field"])] + [(f"{j}.{f}", f) for f in spec.get("review_fields", [])]:
            out[name] = d.get(f) if isinstance(d.get(f), bool) else None
    return out


def main():
    rec = Path(sys.argv[1])
    v = verdicts(rec)
    if "--missing" in sys.argv[2:]:  # judges to run (again): any of its verdicts missing
        print(" ".join(sorted({n.split(".")[0] for n, x in v.items() if x is None})))
        return 0
    override = load(rec / "judge_override.json")
    applied = {j: override[j] for j in v if isinstance(override.get(j), bool) and override[j] != v[j]}
    final = {**v, **applied}
    flagged = [j for j, x in final.items() if x is True]
    missing = [j for j, x in final.items() if x is None]
    outcome = "flagged" if flagged else "incomplete" if missing else "passed"
    gate = {"outcome": outcome, "verdicts": v, "flagged": flagged, "missing": missing,
            "override": {"applied": applied, "by": override.get("by"), "why": override.get("why")} if applied else None,
            "time": time.strftime("%FT%T%z")}
    (rec / "judge_gate.json").write_text(json.dumps(gate, indent=1))
    print(json.dumps({k: gate[k] for k in ("outcome", "flagged", "missing")}))
    return {"passed": 0, "flagged": 1, "incomplete": 2}[outcome]


if __name__ == "__main__":
    sys.exit(main())
