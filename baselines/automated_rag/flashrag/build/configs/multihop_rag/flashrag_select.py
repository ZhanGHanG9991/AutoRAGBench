"""Freeze the best complete configuration available when registered search stopped."""
import argparse
import json
import os
from pathlib import Path
from exp3lib.selector import select
from search_stop import select_as_of_stop, save


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bench', required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--split', default='tune')
    a = p.parse_args()
    if a.split != 'tune':
        raise SystemExit('selector only accepts the train tune split')
    prep = Path(os.environ['EXP3_PREP'])
    run = Path(a.run)
    candidates = json.loads((prep/'flashrag/candidates.json').read_text())['candidates']
    scores = json.loads((run/'tune_scores.json').read_text())
    receipt = json.loads((run/'search_stop.json').read_text())
    assert receipt['questions_per_candidate'] == (45 if receipt['smoke'] else 200)
    winner = select_as_of_stop(candidates, scores, receipt, registered_selector=select)
    with (run/'winner_id.txt').open('x') as f:
        f.write(winner['id'] + '\n')
    save(run/'selection_report.json', {'winner_id': winner['id'],
         'policy': receipt['policy'], 'stopped_early': receipt['stopped_early'],
         'attempted_candidates': len(receipt['attempted_candidates']),
         'completed_candidates': len(receipt['completed_candidates']),
         'registered_candidates': len(candidates), 'candidate_exclusion_used': receipt['candidate_exclusion_used'], 'skipped_candidates': receipt['skipped_candidates'],
         'partial_score_used': False, 'original_selector_used': True,
         'split': 'train-tune', 'test_accessed': False})
    print(winner['id'])


if __name__ == '__main__':
    main()
