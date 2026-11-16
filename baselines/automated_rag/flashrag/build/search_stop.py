"""Stop the entire registered search at the first retrieval token cap.

No candidate invalidation, exclusion, reordering, continuation, budget change,
partial score, or altered tie break. Select only a completed prefix as of stop.
"""
import json
import math
import shutil
from datetime import datetime, timezone

POLICY = 'stop_entire_search_at_first_retrieval_token_cap_v1'


def save(path, value):
    if path.exists():
        archive = path.parent / '_archive' / 'search-checkpoints'
        archive.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        shutil.copy2(path, archive / (path.name + '-' + stamp))
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def search(candidates, rows, build_retrieve, score_hits, budget_error_type,
           bench, *, checkpoint, release):
    scores = {}
    receipt = {'policy': POLICY, 'registered_candidates': [c['id'] for c in candidates],
               'completed_candidates': [], 'attempted_candidates': [],
               'questions_per_candidate': len(rows), 'stopped_early': False,
               'stop': None, 'budget': {'calls': 4, 'tokens': 32000, 'seconds': 60},
               'candidate_exclusion_used': False, 'remaining_candidates_attempted': False}
    checkpoint(scores, receipt)
    for candidate in candidates:
        receipt['attempted_candidates'].append(candidate['id'])
        retrieve = build_retrieve(candidate)
        values = []
        try:
            for index, row in enumerate(rows):
                try:
                    hits, corpus = retrieve(row['question'], row.get('meta') or {})
                except budget_error_type as exc:
                    if exc.kind != 'tokens':
                        raise
                    meter = None
                    tb = exc.__traceback__
                    while tb is not None:
                        candidate_meter = tb.tb_frame.f_locals.get('self')
                        if tb.tb_frame.f_code.co_name == 'after_call' and hasattr(candidate_meter, 'tokens'):
                            meter = {'tokens': candidate_meter.tokens, 'calls': candidate_meter.calls,
                                     'max_tokens': candidate_meter.max_tokens, 'max_calls': candidate_meter.max_calls,
                                     'max_seconds': candidate_meter.max_seconds}
                        tb = tb.tb_next
                    receipt['observed_meter_at_stop'] = meter
                    receipt['stopped_at'] = datetime.now(timezone.utc).isoformat()
                    receipt['stopped_early'] = True
                    receipt['stop'] = {'reason': 'retrieval_token_cap',
                                       'candidate_id': candidate['id'],
                                       'question_index_1based': index + 1,
                                       'completed_rows_in_interrupted_candidate': len(values),
                                       'partial_score_used': False}
                    checkpoint(scores, receipt)
                    return scores, receipt
                # Reader/metric errors retain their original failure behavior.
                values.append(score_hits(row, hits, corpus, bench))
            scores[candidate['id']] = sum(values) / len(values)
            receipt['completed_candidates'].append(candidate['id'])
            checkpoint(scores, receipt)
            print('candidate complete', candidate['id'], 'questions', len(values), flush=True)
        finally:
            del retrieve
            release()
    return scores, receipt


def select_as_of_stop(candidates, scores, receipt, *, registered_selector):
    if receipt.get('policy') != POLICY:
        raise ValueError('missing search stop policy')
    ids = [c['id'] for c in candidates]
    complete = receipt['completed_candidates']
    attempted = receipt['attempted_candidates']
    if receipt.get('registered_candidates') != ids or complete != ids[:len(complete)]:
        raise ValueError('registered candidate order changed')
    if set(scores) != set(complete) or not complete:
        raise ValueError('no complete configuration or inconsistent scores')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in scores.values()):
        raise ValueError('invalid complete score')
    if receipt['stopped_early']:
        stop = receipt['stop']
        if (len(complete) >= len(ids) or attempted != ids[:len(complete)+1]
                or stop.get('candidate_id') != ids[len(complete)]
                or stop.get('reason') != 'retrieval_token_cap'
                or stop.get('partial_score_used') is not False
                or not 0 <= stop['completed_rows_in_interrupted_candidate'] < receipt['questions_per_candidate']
                or stop['question_index_1based'] != stop['completed_rows_in_interrupted_candidate'] + 1):
            raise ValueError('invalid stop receipt or continued search')
    elif complete != ids or attempted != ids or receipt['stop'] is not None:
        raise ValueError('incomplete search without a stop')
    if receipt.get('candidate_exclusion_used') is not False or receipt.get('remaining_candidates_attempted') is not False:
        raise ValueError('candidate exclusion/continuation refused')
    return registered_selector(candidates[:len(complete)], scores, split='tune')
