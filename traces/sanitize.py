"""Redact secrets from record files in place (run on the cluster before records leave it).

Secrets: every *_API_KEY / *_TOKEN value in the repo .env, plus values given via --extra-env names
(e.g. VLLM_API_KEY, GATEWAY_ADMIN_TOKEN of the running job). Replaced by [REDACTED:<NAME>].
Usage: sanitize.py <file_or_dir> [...] [--env-file .env] [--extra-env NAME ...]
"""
import argparse
import os
from pathlib import Path

SUFFIXES = ("_API_KEY", "_TOKEN", "_KEY")
MIN_LEN = 8


def secrets(env_file, extra):
    out = []
    if env_file and Path(env_file).exists():
        for line in Path(env_file).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.replace("export ", "").strip()
            v = v.strip().strip('"').strip("'")
            if k.endswith(SUFFIXES) and len(v) >= MIN_LEN:
                out.append((k, v))
    for k in extra:
        v = os.environ.get(k, "")
        if len(v) >= MIN_LEN:
            out.append((k, v))
    return out


def redact_file(p, pairs):
    try:
        data = p.read_bytes()
    except OSError:
        return 0
    n = 0
    for k, v in pairs:
        b = v.encode()
        if b in data:
            n += data.count(b)
            data = data.replace(b, f"[REDACTED:{k}]".encode())
    if n:
        p.write_bytes(data)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--env-file", default=str(Path(__file__).resolve().parents[1] / ".env"))
    ap.add_argument("--extra-env", nargs="*", default=[])
    a = ap.parse_args()
    pairs = secrets(a.env_file, a.extra_env)
    total = 0
    for root in a.paths:
        root = Path(root)
        files = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file() and not p.is_symlink()]
        for p in files:
            if p.stat().st_size < 2_000_000_000:
                total += redact_file(p, pairs)
    print(f"redacted {total} occurrences of {len(pairs)} secrets")


if __name__ == "__main__":
    main()
