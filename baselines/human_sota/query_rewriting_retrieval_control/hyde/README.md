# HyDE

Paper: L. Gao, X. Ma, J. Lin, J. Callan. *Precise Zero-Shot Dense Retrieval without Relevance Labels* (HyDE). ACL 2023.
Upstream: https://github.com/texttron/hyde.git
Commit: `a2fd8734307612cb0225d71ffbf26e0d225986b8`.
The upstream commit has no LICENSE file; no upstream source is vendored here.

The AutoRAGBench adapter encodes the question and eight hypothetical passages with
`facebook/contriever`, averages their vectors, and returns the top 10 original
corpus passages by exact inner product. One Qwen3.5-4B request at `$LLM_BASE_URL`
generates all eight hypotheses (`n=8`, temperature 0.7, 512 tokens each, seed
20261015). Generated hypotheses never appear in `/retrieve` evidence.
The encoder uses attention-mask mean pooling, length 512, no normalization,
bfloat16 build inference, and float16 stored vectors/search.

| Tasks evaluated | Index / prompt |
|---|---|
| CWQ, WebQSP | 400-token windows; general passage prompt |
| MultiHop-RAG | 400-token windows; news passage prompt |
| QASPER | 400-token windows; scientific passage prompt; `meta.doc_id` scope |
| MuSiQue, HotpotQA | Whole-document encoding; general passage prompt |
| PubMedQA | Whole-document encoding; scientific passage prompt |
| FEVER | Whole-document encoding; claim-verification prompt |

NQ was not evaluated in this release batch. Whole-document inputs are truncated
to the encoder's 512-token limit. Shared files are stored once; `build/layout.json`
selects exact task configs, including aliases to identical configurations.
Legacy `/answer` code remains in the frozen source; evaluation uses `/retrieve`.

## Rebuild

Follow [workspace assembly](../../_common/fixed_evidence_adapter/README.md), using
this directory as `$M`, a pinned upstream checkout as `$U`, and a new workspace `$W`.
Create a Python 3.12 environment at `$W/.venv`; install `build/requirements.lock`
and upstream dependencies. No upstream patch is needed.
Download `facebook/contriever` revision `2bd46a25019aeea091fd42d1f0fd4801675cf699`
into `$W/models/contriever`; `build/download_model.py` downloads and verifies it
under its adjacent `models/` directory, which must then be copied into `$W/models/`.
With the public corpus and a GPU available, run:

```bash
"$W/.venv/bin/python" "$M/build/rebuild.py" --workspace "$W" --task "$TASK" \
  --corpus "$CORPUS_DIR/corpus.jsonl" --out "$W/index"
bash "$W/submission/start.sh"
```

The portable wrapper calls the archived `prepare`, `encode`, and `verify`
functions directly. Historical build scripts remain in `build_sources/` after
assembly; their direct CLI requires `AUTORAGBENCH_ROOT` and the original relative
`humansota/HyDE/rageng/` and `data_formal-v*/` layout. The wrapper needs neither.
Earlier byte-identical indexes were reused for FEVER, HotpotQA and PubMedQA;
`build/legacy/` preserves that earlier builder. Models, arrays, environments and
corpora are omitted. No model fine-tuning is required.

## VI-G starting package

HotpotQA and PubMedQA packages reuse the evaluated server, encoder and configs.
Assemble with `--starting-package`; only the launcher, environment setup and
whole-document `code/rebuild_index.py` are additional. The supplied rebuild script
expects an existing `index/manifest.json`; first build the index as above, then
`bash "$W/build.sh" NEW_INDEX` can rebuild into a new directory. It joins title and
body with a space, while the archived builder uses a newline; that source detail
is preserved. The encoder tokenizes both as whitespace. No starting index is shipped.
The supplied build.sh requires a Slurm allocation and BUILD_PARTITION; the portable
rebuild wrapper above has no scheduler requirement.
