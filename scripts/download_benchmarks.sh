#!/bin/bash
# Download raw benchmark files from Hugging Face ($HF_ENDPOINT or $HF_MIRROR) into benchmarks/<Name>/raw/.
# Raw files are never modified; every file gets a sha256 and the dataset revision is recorded.
#   bash scripts/download_benchmarks.sh [NQ-open PopQA PubMedQA FEVER QASPER FinanceBench MultiHop-RAG ALCE-ASQA]
# GitHub repos / non-HF corpora are fetched separately by scripts/download_benchmarks_extra.sh (needs vpn).
set -uo pipefail
source "$(dirname "$0")/../configs/pilot.env"
HF=${HF_ENDPOINT:-$HF_MIRROR}
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

declare -A REPO FILES
REPO[NQ-open]="google-research-datasets/nq_open Tevatron/wikipedia-nq"
FILES[google-research-datasets/nq_open]="nq_open/train-00000-of-00001.parquet nq_open/validation-00000-of-00001.parquet"
FILES[Tevatron/wikipedia-nq]="nq-dev.jsonl.gz nq-test.jsonl.gz nq-train.jsonl.gz"
REPO[PopQA]="akariasai/PopQA"
FILES[akariasai/PopQA]="test.tsv"
REPO[PubMedQA]="qiaojin/PubMedQA"
FILES[qiaojin/PubMedQA]="pqa_labeled/train-00000-of-00001.parquet pqa_unlabeled/train-00000-of-00001.parquet pqa_artificial/train-00000-of-00001.parquet"
REPO[FEVER]="copenlu/fever_gold_evidence"
FILES[copenlu/fever_gold_evidence]="train.jsonl valid.jsonl test.jsonl"
REPO[QASPER]="allenai/qasper"
FILES[allenai/qasper]="@parquet:qasper/train/0.parquet @parquet:qasper/validation/0.parquet @parquet:qasper/test/0.parquet"
REPO[FinanceBench]="PatronusAI/financebench"
FILES[PatronusAI/financebench]="financebench_merged.jsonl"
REPO[MultiHop-RAG]="yixuantt/MultiHopRAG"
FILES[yixuantt/MultiHopRAG]="MultiHopRAG.json corpus.json"
REPO[ALCE-ASQA]="princeton-nlp/ALCE-data din0s/asqa"
FILES[princeton-nlp/ALCE-data]="ALCE-data.tar"
FILES[din0s/asqa]="data/train-00000-of-00001-87b7d64f7913b544.parquet data/dev-00000-of-00001-58a9a40c6e69f07b.parquet"

fetch() {  # fetch <url> <dest>
  mkdir -p "$(dirname "$2")"
  for i in 1 2 3; do
    curl -sSL --fail --retry 3 --connect-timeout 20 -o "$2.part" "$1" && mv "$2.part" "$2" && return 0
    sleep 5
  done
  echo "FAILED $1"; return 1
}

for name in "${@:-NQ-open PopQA PubMedQA FEVER QASPER FinanceBench MultiHop-RAG ALCE-ASQA}"; do
  for name1 in $name; do
    D=$RAGENG_ROOT/benchmarks/$name1; mkdir -p $D/raw $D/meta
    echo "== $name1"
    : > $D/meta/manifest.tsv
    for repo in ${REPO[$name1]}; do
      sha=$(curl -sL --max-time 30 "$HF/api/datasets/$repo" | python3 -c "import json,sys; print(json.load(sys.stdin).get('sha','?'))" 2>/dev/null)
      for f in ${FILES[$repo]}; do
        if [[ $f == @parquet:* ]]; then
          p=${f#@parquet:}; url="$HF/api/datasets/$repo/parquet/$p"; dest=$D/raw/${repo//\//__}/parquet/$p
        else
          url="$HF/datasets/$repo/resolve/$sha/$f"; dest=$D/raw/${repo//\//__}/$f
        fi
        [ -s "$dest" ] || fetch "$url" "$dest" || continue
        printf '%s\t%s\t%s\t%s\t%s\n' "$repo" "$sha" "$f" "$(stat -c %s "$dest")" "$(sha256sum "$dest" | cut -d' ' -f1)" >> $D/meta/manifest.tsv
      done
    done
    column -t -s $'\t' $D/meta/manifest.tsv
  done
done
echo "DOWNLOAD_DONE $(date -Is)"
