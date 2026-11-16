#!/bin/bash
# Formal-v1 T5-large evaluator training on the filtered balanced pairs.
# Split like pilot into two 4-epoch parts. No ablation. No sbatch from here.
# Usage: sbatch run_train.sbatch <1|2>
# Part 2 must be submitted with --dependency=afterok:<part1> --kill-on-invalid-dep=yes
#SBATCH --job-name=crag-fv1-train
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
ROOT=${AUTORAGBENCH_ROOT:?}
C=$ROOT/humansota/CorrectiveRAG
FORMAL=$C/rageng/formal-v1
PY=$C/.venv/bin/python
PART=${1:?usage: run_train.sbatch <1|2>}
case "$PART" in
  1|2) ;;
  *) echo "REFUSE part $PART"; exit 1 ;;
esac
# Formal-v1 only. Never train the mh-unfiltered ablation from this entrypoint.
unset CRAG_ABLATION
export CRAG_FORMAL=1
mkdir -p "$FORMAL/logs" "$FORMAL/results" "$C/models/evaluator-rageng-formal-v1"
avail=$(df -BG --output=avail ${BUILD_TMP:?} | tail -1 | tr -dc '0-9')
echo "DISK_AVAIL_GB $avail"
if [ "${avail:-0}" -lt 200 ]; then echo "DISK_LOW $avail"; exit 1; fi
test -f "$FORMAL/pairs/evaluator_formal_v1_pairs.jsonl"
test -f "$FORMAL/pairs/pair_stats.json"
if [ "$PART" = 2 ]; then
  test -f "$C/models/evaluator-rageng-formal-v1/resume.pt"
fi
cd "$C/rageng"
echo "JOB_START $(date -Is) job=$SLURM_JOB_ID host=$(hostname) partition=$SLURM_JOB_PARTITION part=$PART formal=1"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
"$PY" "$C/rageng/train_evaluator_rageng.py" --part "$PART"
echo "TRAIN_PART${PART}_DONE $(date -Is)"
