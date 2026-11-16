"""File manifests (path, size, SHA-256) and freezing.

  freeze.py hash   <dir> <manifest.json>
  freeze.py verify <dir> <manifest.json>          exit 1 and list differences if changed
  freeze.py freeze <run_dir> <records_dir>        chmod -R a-w workspace, write workspace manifest,
                                                  verify task/ against records/task_manifest.json
"""
import hashlib
import json
import os
import stat
import sys
import time

IGNORE = {".deadline"}  # written into task/ when the build clock starts


def manifest(root):
    out = {}
    for dp, dns, fns in os.walk(root, followlinks=False):
        for name in fns + [d for d in dns if os.path.islink(os.path.join(dp, d))]:
            if name in IGNORE:
                continue
            p = os.path.join(dp, name)
            rel = os.path.relpath(p, root)
            if os.path.islink(p):
                out[rel] = {"link": os.readlink(p)}
                continue
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for b in iter(lambda: f.read(1 << 20), b""):
                    h.update(b)
            out[rel] = {"size": os.path.getsize(p), "sha256": h.hexdigest()}
    return out


def diff(old, new):
    d = []
    for k in sorted(set(old) | set(new)):
        if k not in new:
            d.append(f"removed {k}")
        elif k not in old:
            d.append(f"added {k}")
        elif old[k] != new[k]:
            d.append(f"changed {k}")
    return d


def make_readonly(root):
    for dp, dns, fns in os.walk(root, followlinks=False):
        for name in fns + dns:
            p = os.path.join(dp, name)
            if not os.path.islink(p):
                m = os.stat(p).st_mode
                os.chmod(p, m & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    m = os.stat(root).st_mode
    os.chmod(root, m & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def main():
    cmd = sys.argv[1]
    if cmd == "hash":
        json.dump(manifest(sys.argv[2]), open(sys.argv[3], "w"))
    elif cmd == "verify":
        d = diff(json.load(open(sys.argv[3])), manifest(sys.argv[2]))
        print("\n".join(d[:50]) or "OK")
        sys.exit(1 if d else 0)
    elif cmd == "freeze":
        run, rec = sys.argv[2], sys.argv[3]
        ws = os.path.join(run, "workspace")
        make_readonly(ws)
        m = manifest(ws)
        json.dump(m, open(os.path.join(rec, "workspace_manifest.json"), "w"))
        task_diff = diff(json.load(open(os.path.join(rec, "task_manifest.json"))), manifest(os.path.join(run, "task")))
        info = {"time": time.strftime("%FT%T%z"), "workspace_files": len(m),
                "workspace_bytes": sum(v.get("size", 0) for v in m.values()),
                "has_start_sh": os.path.exists(os.path.join(ws, "submission", "start.sh")),
                "task_unchanged": not task_diff, "task_diff": task_diff[:50]}
        json.dump(info, open(os.path.join(rec, "freeze.json"), "w"), indent=1)
        print(json.dumps(info))


if __name__ == "__main__":
    main()
