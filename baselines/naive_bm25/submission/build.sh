#!/bin/bash
# Prebuild the BM25 index into <workspace>/index before the workspace is frozen (slurm/run.sbatch, baselines).
set -euo pipefail
cd "$(dirname "$0")"
exec @RAG_BASE_PY@ ragcore.py build "$CORPUS_DIR"
