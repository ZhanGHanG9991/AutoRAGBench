import argparse
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path


def digest(path):
    if not path or not Path(path).is_file():
        return None
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument("--method", required=True)
parser.add_argument("--task", required=True)
parser.add_argument("--corpus", required=True)
parser.add_argument("--train", default="")
parser.add_argument("--dev", default="")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--config", action="append", default=[])
args = parser.parse_args()
manifest = {"campaign": "db5-formal-v2-20261105", "method": args.method, "task": args.task,
            "data_version": "formal-v2", "protocol": "fixed-evidence-v1",
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "node": platform.node(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "corpus": args.corpus, "corpus_sha256": digest(args.corpus),
            "train": args.train or None, "train_sha256": digest(args.train),
            "dev": args.dev or None, "dev_sha256": digest(args.dev),
            "config_files": [{"path": p, "sha256": digest(p)} for p in args.config],
            "hidden_test_used": False, "external_paid_api_used": False}
try:
    manifest["gpu"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,uuid", "--format=csv,noheader"], text=True).strip()
except Exception as exc:
    manifest["gpu_error"] = str(exc)
(args.output / "build_manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest), flush=True)
