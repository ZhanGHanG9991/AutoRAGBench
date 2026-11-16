#!/bin/bash
# Prebuild BM25 + bge-base embeddings into <workspace>/index before the workspace is frozen (slurm/run.sbatch).
set -euo pipefail
cd "$(dirname "$0")"
exec @RAG_BASE_PY@ ragcore.py build "$CORPUS_DIR" \
  "$(cd .. && pwd)/models/bge-base-en-v1.5"
