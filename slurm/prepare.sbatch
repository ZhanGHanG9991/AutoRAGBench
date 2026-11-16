#!/bin/bash
#SBATCH --job-name=rageng-prepare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
# CPU-only: build task instances of one data version ($RAGENG_DATA_VERSION, not "pilot"):
# $DATA_DIR/<task>/public (corpus + train) and the hidden test in $HIDDEN_DIR/<version>/<task>/.
# Sampling rules of the version, if any: configs/data_rules/<version>.json (tasks.prepare --rules).
#   prepare.sbatch <tasks> <train> <test> <seed>
set -euo pipefail
source "${RAGENG_ROOT:?export RAGENG_ROOT=<repository root> before sbatch}/configs/pilot.env"
[ "$DATA_VERSION" != pilot ] || { echo "set RAGENG_DATA_VERSION (pilot data is not rebuilt)"; exit 2; }
cd $RAGENG_ROOT
RULES=()
[ -f configs/data_rules/$DATA_VERSION.json ] && RULES=(--rules configs/data_rules/$DATA_VERSION.json)
echo "JOB_START $(date -Is) host=$(hostname) job=$SLURM_JOB_ID version=$DATA_VERSION tasks=$1"
$RAG_BASE_PY -m tasks.prepare --tasks "$1" --root $RAGENG_ROOT --out $DATA_DIR --hidden $HIDDEN_DIR/$DATA_VERSION \
  --train "$2" --test "$3" --seed "$4" ${RULES[@]+"${RULES[@]}"}
echo "JOB_END $(date -Is)"
