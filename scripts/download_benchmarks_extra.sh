#!/bin/bash
# Non-HF parts of FinanceBench / MultiHop-RAG / ALCE-ASQA: official GitHub repos (code, eval, FinanceBench PDFs)
# and the DPR Wikipedia corpus (psgs_w100, stored under ALCE-ASQA/corpus; tasks/nq_open uses it as the NQ corpus).
# Needs access to GitHub and dl.fbaipublicfiles.com (set a proxy in your shell if required).
#   bash scripts/download_benchmarks_extra.sh [FinanceBench MultiHop-RAG ALCE-ASQA]
set -uo pipefail
source "$(dirname "$0")/../configs/pilot.env"

declare -A GIT
GIT[FinanceBench]=https://github.com/patronus-ai/financebench.git
GIT[MultiHop-RAG]=https://github.com/yixuantt/MultiHop-RAG.git
GIT[ALCE-ASQA]=https://github.com/princeton-nlp/ALCE.git
DPR_URL=https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz

for name in ${@:-FinanceBench MultiHop-RAG ALCE-ASQA}; do
  D=$RAGENG_ROOT/benchmarks/$name; mkdir -p $D/meta; echo "== $name"
  if [ ! -d $D/repo/.git ]; then
    for i in 1 2 3; do git clone --depth 1 "${GIT[$name]}" $D/repo.part && mv $D/repo.part $D/repo && break; rm -rf $D/repo.part; sleep 10; done
  fi
  [ -d $D/repo/.git ] && printf '%s\t%s\n' "${GIT[$name]}" "$(git -C $D/repo rev-parse HEAD)" > $D/meta/repo_commit.tsv && cat $D/meta/repo_commit.tsv
  if [ $name = ALCE-ASQA ]; then
    C=$D/corpus; mkdir -p $C
    if [ ! -s $C/psgs_w100.tsv.gz ]; then
      for i in 1 2 3 4 5; do curl -sSL --fail -C - --retry 3 --connect-timeout 20 -o $C/psgs_w100.tsv.gz.part "$DPR_URL" && mv $C/psgs_w100.tsv.gz.part $C/psgs_w100.tsv.gz && break; sleep 10; done
    fi
    if [ -s $C/psgs_w100.tsv.gz ]; then
      gzip -t $C/psgs_w100.tsv.gz && echo "gzip -t OK"
      printf '%s\t%s\t%s\n' "$DPR_URL" "$(stat -c %s $C/psgs_w100.tsv.gz)" "$(sha256sum $C/psgs_w100.tsv.gz | cut -d' ' -f1)" > $D/meta/corpus_manifest.tsv
      cat $D/meta/corpus_manifest.tsv
    else echo "FAILED $DPR_URL"; fi
  fi
done
echo "EXTRA_DONE $(date -Is)"
