"""Build the read-only evidence directory the judges work in (cluster side, after the build job).

  prepare_sandbox.py <run_dir> <records_dir> <out_dir>

out_dir/
  task/README.md, task/task.json      the rules and task the agent was given
  workspace/                          the frozen workspace without .venv and without files > 5 MB
  workspace_files.tsv                 every workspace file: size, path (including omitted ones)
  solve_parsed.txt                    the readable, secret-redacted agent trace
  agent_meta.json, freeze.json        harness / model / timing; freeze summary
  gateway_build_summary.json          generator calls during the build: counts, tokens, samples
  provenance.json                     where workspace text comes from (audit/provenance.py)
  workspace_scan.json                 model inventory and static scan of the workspace (audit/scan_workspace.py)
  leak_scan_build.json                test question text in what the agent wrote? (audit/leak_scan.py, paths only)
"""
import json
import os
import shutil
import sys
from pathlib import Path

MAX_FILE = 5_000_000
SKIP_DIRS = {".venv", "venv", "__pycache__", ".git", "node_modules"}


def copy_workspace(src, dst, listing):
    with open(listing, "w") as lf:
        for dp, dns, fns in os.walk(src):
            rel_dp = os.path.relpath(dp, src)
            skip = any(part in SKIP_DIRS for part in Path(rel_dp).parts)
            for fn in fns:
                p = os.path.join(dp, fn)
                if os.path.islink(p):
                    continue
                size = os.path.getsize(p)
                rel = os.path.relpath(p, src)
                lf.write(f"{size}\t{rel}\n")
                if skip or size > MAX_FILE:
                    continue
                q = os.path.join(dst, rel)
                os.makedirs(os.path.dirname(q), exist_ok=True)
                shutil.copyfile(p, q)


def gateway_summary(path, out):
    calls, prompt, comp, samples, first, last = 0, 0, 0, [], None, None
    if os.path.exists(path):
        rows = []
        for line in open(path, errors="ignore"):
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("event") != "call":
                continue
            calls += 1
            u = e.get("usage") or {}
            prompt += u.get("prompt_tokens", 0) or 0
            comp += u.get("completion_tokens", 0) or 0
            first = first or e.get("ts")
            last = e.get("ts")
            rows.append(e)
        step = max(1, len(rows) // 30)
        for e in rows[::step][:30]:
            msgs = (e.get("request") or {}).get("messages") or [{"content": (e.get("request") or {}).get("prompt", "")}]
            samples.append({"ts": e.get("ts"), "prompt_tail": str(msgs[-1].get("content", ""))[-1500:],
                            "output": (e.get("outputs") or [""])[0][:800]})
    json.dump({"calls": calls, "prompt_tokens": prompt, "completion_tokens": comp, "first_ts": first,
               "last_ts": last, "samples": samples}, open(out, "w"), indent=1)


def main():
    run, rec, out = map(Path, sys.argv[1:4])
    if out.exists():
        shutil.rmtree(out)
    (out / "task").mkdir(parents=True)
    for f in ("README.md", "task.json"):
        shutil.copyfile(run / "task" / f, out / "task" / f)
    copy_workspace(run / "workspace", out / "workspace", out / "workspace_files.tsv")
    for f in ("solve_parsed.txt", "agent_meta.json", "freeze.json", "provenance.json", "workspace_scan.json",
              "leak_scan_build.json"):
        if (rec / f).exists():
            shutil.copyfile(rec / f, out / f)
    gateway_summary(rec / "gateway_build.jsonl", out / "gateway_build_summary.json")
    print(f"judge sandbox ready: {out}")


if __name__ == "__main__":
    main()
