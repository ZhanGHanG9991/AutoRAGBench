"""Write the frozen runtime entry and non-secret experiment metadata."""
import argparse,json,os,shutil
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--framework",required=True);ap.add_argument("--bench",required=True);ap.add_argument("--run",required=True);ap.add_argument("--mode",required=True);a=ap.parse_args()
    run=Path(a.run);prep=Path(os.environ["EXP3_PREP"]);sub=run/"submission"
    if sub.exists():raise FileExistsError("submission already exists")
    sub.mkdir();shutil.copytree(prep/"lib",sub/"lib",ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(prep/"submission"/a.framework,sub/"runtime",ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(prep/"flashrag",sub/"flashrag")
    lines=["#!/bin/bash","set -euo pipefail",'HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)','RUN=$(dirname -- "$HERE")','export EXP3_PREP="$HERE"',f'export EXP3_TASK="{a.bench}"','export EXP3_CORPUS="${CORPUS_DIR:-$RUN/../task/corpus}/corpus.jsonl"','export EXP3_DEVICE=cuda HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 EXP3_DISABLE_THINKING=1 LITELLM_LOCAL_MODEL_COST_MAP=True','export EXP3_MODELS_DIR="$RUN/runtime/models"','export PYTHONPATH="$HERE/lib:$HERE/runtime:$RUN/runtime/vendor"']
    if a.framework=="autorag":lines += ['export EXP3_LANCE_DIR="$RUN/serve-index" EXP3_TRIAL_JSON="$RUN/winning-trial.json"']
    elif a.framework=="dspy":lines += ['export EXP3_BM25_DIR="$RUN/bm25" EXP3_DSPY_PROGRAM="$RUN/hop.json"']
    else:
        winner=(run/"winner_id.txt").read_text().strip();candidates=json.loads((prep/"flashrag/candidates.json").read_text())["candidates"];c=next(c for c in candidates if c["id"]==winner)
        model={"e5":"e5-large-v2","bge":"bge-large-en-v1.5"}.get(c["retriever"],"")
        lines += [f'export EXP3_FLASHRAG_CANDIDATE="{winner}"','export EXP3_BM25_DIR="$RUN/indexes/bm25"',f'export EXP3_DENSE_DIR="$RUN/indexes/{model}"']
    lines += [f'exec "$RUN/runtime/env/bin/python" "$HERE/runtime/exp3_{a.framework}_server.py"']
    (sub/"start.sh").write_text("\n".join(lines)+"\n");(sub/"start.sh").chmod(0o755)
    (run/"x9_metadata.json").write_text(json.dumps({"framework":a.framework,"bench":a.bench,"mode":a.mode,"data_version":"formal-v2","split_seed":20261027,"optimizer_model":"deepseek-flash","retrieval_generation":"platform generator Qwen3.5-4B","partition":os.environ.get("SLURM_JOB_PARTITION",""),"gpu_name":os.environ.get("GPU_NAME",""),"gpus":1,"test_evaluation":False,"external_readonly_dependencies":["platform standard envs/rag-base and envs/python only"]},indent=2))
if __name__=="__main__":main()
