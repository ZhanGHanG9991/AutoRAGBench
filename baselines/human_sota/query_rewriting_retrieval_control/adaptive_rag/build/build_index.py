"""Build the ragcore index for one public corpus. BM25, optional dense."""
import argparse
import os

from guard import assert_task, corpus_dir
from ragcore import build

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--dense", action="store_true")
    args = ap.parse_args()
    task = assert_task(args.task)
    out = os.path.join(HERE, "index", task)
    dense = None
    if args.dense:
        dense = os.path.join(os.environ["RAGENG_ROOT"], "envs", "models", "bge-base-en-v1.5")
        if not os.path.isdir(dense):
            raise SystemExit(f"missing dense model {dense}")
    print(f"INDEX task={task} dense={bool(dense)} -> {out}", flush=True)
    build(corpus_dir(task), out, dense_model=dense, device="cuda" if args.dense else "cpu")
    print("INDEX done", flush=True)


if __name__ == "__main__":
    main()
