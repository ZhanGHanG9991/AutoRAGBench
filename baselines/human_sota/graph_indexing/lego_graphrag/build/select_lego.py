import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--summary", type=Path)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
if args.task == "pubmedqa":
    result = {"task": args.task, "instance": "SBE-PPR+OSAR-LLMs", "ctx": "paths",
              "selection": "pilot3-predeclared-fixed", "test_used": False}
else:
    summary = json.loads(args.summary.read_text())
    if not summary:
        raise RuntimeError("empty LEGO dev summary")
    best = max(summary, key=lambda key: (summary[key]["primary"], -summary[key].get("n_invalid", 0), key))
    instance, ctx = best.split("__", 1)
    result = {"task": args.task, "instance": instance, "ctx": ctx,
              "selection": "formal-v2-public-dev-native-primary", "selected_record": summary[best],
              "all_records": summary, "test_used": False}
args.out.write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
