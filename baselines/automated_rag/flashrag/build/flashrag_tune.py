"""Registered search, stopping globally when the existing token meter fires."""
import argparse
import gc
import json
import os
import sys
from pathlib import Path
from exp3lib.budget import BudgetError
from exp3lib.tune_scorer import score_hits
from search_stop import search, save


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('Slurm allocation required')
    ap = argparse.ArgumentParser()
    ap.add_argument('--bench', required=True)
    ap.add_argument('--run', required=True)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    prep = Path(os.environ['EXP3_PREP'])
    run = Path(a.run)
    rows = json.loads((prep/'splits'/a.bench/'tune.json').read_text())
    assert len(rows) == 200
    if a.smoke:
        assert a.bench == 'qasper'
        rows = rows[:45]
    candidates = json.loads((prep/'flashrag/candidates.json').read_text())['candidates']
    sys.path.insert(0, str(prep/'submission/flashrag'))
    from exp3_flashrag_server import build_retrieve

    def build(candidate):
        os.environ['EXP3_FLASHRAG_CANDIDATE'] = candidate['id']
        model = {'e5': 'e5-large-v2', 'bge': 'bge-large-en-v1.5'}.get(candidate['retriever'], '')
        os.environ['EXP3_DENSE_DIR'] = str(run/'indexes'/model)
        os.environ['EXP3_BM25_DIR'] = str(run/'indexes/bm25')
        return build_retrieve()

    def checkpoint(scores, receipt):
        receipt['smoke'] = a.smoke
        receipt['split'] = 'train-tune'
        receipt['test_accessed'] = False
        save(run/'tune_scores.json', scores)
        save(run/'search_stop.json', receipt)

    def release():
        gc.collect()
        import torch
        torch.cuda.empty_cache()

    scores, receipt = search(candidates, rows, build, score_hits, BudgetError,
                             a.bench, checkpoint=checkpoint, release=release)
    if receipt['stopped_early']:
        print('stopped early on budget: attempted', len(receipt['attempted_candidates']),
              'candidates; fully evaluated', len(receipt['completed_candidates']), '; remaining candidates skipped', flush=True)
    if not scores:
        raise RuntimeError('No complete configuration available at budget stop')


if __name__ == '__main__':
    main()
