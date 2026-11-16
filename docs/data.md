# Data card

Seed 20261015. Train ≤ 10,000 and test ≤ 1,000 questions per task (fewer when the source has fewer). Corpora are the
full benchmark corpora, independent of which questions were drawn. Document and question ids are seeded random UUIDs
that carry no information about the source data. Sampling rules: `configs/data_rules/formal-v2.json`; fingerprints
frozen before the experiments: `configs/data_rules/formal-v2.frozen.json`; checksums of every public file:
`configs/data_rules/formal-v2.public.sha256`.

| Task | Train source | Test source | Corpus |
|-|-|-|-|
| NQ (`nq_open`) | Tevatron `nq-train` (DPR train questions with a positive passage) | official NQ-open test (HF split `validation`, 3,610 questions) | DPR Wikipedia passages `psgs_w100` (21,015,324) |
| HotpotQA (`hotpotqa`) | official train, `level=hard` | official distractor dev (all hard), stratified by type | fullwiki corpus: introductory paragraphs of enwiki-20171001 (5,233,235) |
| MuSiQue (`musique`) | official MuSiQue-Ans train | official MuSiQue-Ans dev (the official test has no answers), stratified by hop count | every paragraph of the Ans train, dev and test files, deduplicated (117,534) |
| MH-RAG (`multihop_rag`) | all remaining questions except those whose evidence set equals a test question's (1,680) | 500 questions drawn at random, stratified by question type, ≤ 3 per identical evidence set | the 609 news articles, one document per article with source, date, author and category |
| PubMedQA (`pubmedqa`) | the 500 PQA-L questions outside the official test | official PQA-L test (500) | abstracts of the PQA-L, PQA-U and PQA-A subsets, deduplicated by PubMed id (273,518); labelled sections without title and conclusion, as in the reasoning-required setting |
| QASPER (`qasper`) | every question of the official train + validation papers (3,598) | official test papers, ≤ 3 questions per paper | full text of every paper, one document per paper (1,585); `meta.doc_id` names the target paper |
| FEVER (`fever`) | official train | official `paper_test`, stratified by label | introductory sections of the FEVER Wikipedia dump (5,396,106 pages) |
| WebQSP (`webqsp`) | official train + validation (answerable questions, 2,945) | official test | union of the RoG Freebase subgraphs, one document per entity with its outgoing triples (665,334) + `triples.tsv` |
| CWQ (`cwq`) | official RoG train | official test | union of the RoG subgraphs of train, validation and test (1,310,816 entities, 6.95M triples) + `triples.tsv` |

Train keeps the official train split's own label / type / hop distribution; the test follows the natural mix of its
official pool. KGQA questions carry `meta.topic_entities` (names as in the graph). Train rows carry
`supporting_doc_ids` where the source labels supporting evidence.

## Known properties

* FEVER: train and test label distributions differ (train ≈ 55% SUPPORTS / 20% REFUTES / 25% NOT ENOUGH INFO; test
  one third each).
* MuSiQue: train 2/3/4-hop ≈ 72/22/6%, test ≈ 52/31/17%.
* NQ: train contains only questions for which DPR retrieves a positive passage; the test has no such filter.
* MH-RAG: answers are concentrated (two answers account for about 19% of all questions); about 72% of test questions
  share at least one evidence sentence with a training question (no identical evidence set is shared).
* CWQ: about 72% of test questions have a topic entity that also occurs in training questions (inherited from the
  source data).
* QASPER: 133 test questions have the same text as a training question (generic questions about different papers).
* HotpotQA: supporting articles of test questions can also support training questions (inherited, reported in the
  split manifest, not filtered).

## Rebuilding from the raw releases

Raw files go to `benchmarks/<Name>/raw/` (`scripts/download_benchmarks.sh` for the Hugging Face releases,
`scripts/download_benchmarks_extra.sh` for the DPR passages; the HotpotQA Wikipedia dump, MuSiQue, FEVER, the
PubMedQA test ids and the RoG subgraphs are fetched from their official releases into the paths named at the top of
each `tasks/<task>/task.py`). Then:

```bash
RAGENG_DATA_VERSION=formal-v2 sbatch slurm/prepare.sbatch all 10000 1000 20261015
# or directly: python -m tasks.prepare --tasks all --root . --out data/formal-v2 --hidden hidden/formal-v2 \
#   --train 10000 --test 1000 --seed 20261015 --rules configs/data_rules/formal-v2.json
```

`tasks.prepare` writes `data/formal-v2/<task>/public/` and moves the test split into `hidden/formal-v2/<task>/`
(`test_questions.jsonl`, `test_gold.jsonl`, `split_manifest.json`).

The released test set is exactly the one written by this step for the paper (sha256 in
`configs/data_rules/formal-v2.frozen.json`: `test_questions_sha256`, `test_gold_sha256`, `split_manifest_sha256`).
