"""Cell-local policy: skip only over-limit candidates, retaining the registered order."""
import json,math,shutil
from datetime import datetime,timezone
POLICY='skip_over_limit_candidate_continue_131072_v1'
def save(path,value):
    if path.exists():
        archive=path.parent/'_archive'/'search-checkpoints';archive.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,archive/(path.name+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')))
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def search(candidates,rows,build_retrieve,score_hits,budget_error_type,bench,*,checkpoint,release):
    scores={};receipt={'policy':POLICY,'registered_candidates':[c['id'] for c in candidates],
      'completed_candidates':[],'attempted_candidates':[],'skipped_candidates':[],
      'questions_per_candidate':len(rows),'stopped_early':False,'stop':None,
      'budget':{'calls':4,'tokens':131072,'seconds':60},'candidate_exclusion_used':True,
      'remaining_candidates_attempted':True,'tune_reader_budget_unchanged':{'calls':1,'tokens':32000,'seconds':60,'max_output_tokens':512}}
    checkpoint(scores,receipt)
    for candidate in candidates:
        receipt['attempted_candidates'].append(candidate['id']);retrieve=build_retrieve(candidate);values=[];skipped=False
        try:
            for index,row in enumerate(rows):
                try:hits,corpus=retrieve(row['question'],row.get('meta') or {})
                except budget_error_type as exc:
                    if exc.kind not in ('tokens','context'):raise
                    receipt['skipped_candidates'].append({'candidate_id':candidate['id'],'reason':exc.kind,
                      'question_index_1based':index+1,'completed_rows_discarded':len(values),'partial_score_used':False})
                    skipped=True;checkpoint(scores,receipt)
                    print('candidate over-limit/skipped',candidate['id'],'reason',exc.kind,'question',index+1,flush=True);break
                # The original tune scorer and its 32000-token cap are deliberately unchanged.
                values.append(score_hits(row,hits,corpus,bench))
            if not skipped:
                scores[candidate['id']]=sum(values)/len(values);receipt['completed_candidates'].append(candidate['id'])
                checkpoint(scores,receipt);print('candidate complete',candidate['id'],'questions',len(values),flush=True)
        finally:del retrieve;release()
    return scores,receipt

def select_as_of_stop(candidates,scores,receipt,*,registered_selector):
    ids=[c['id'] for c in candidates];complete=receipt['completed_candidates'];skip=[x['candidate_id'] for x in receipt['skipped_candidates']]
    assert receipt['policy']==POLICY and receipt['registered_candidates']==ids and receipt['attempted_candidates']==ids
    assert set(scores)==set(complete) and complete and not set(complete)&set(skip)
    assert set(complete)|set(skip)==set(ids)
    assert complete==[x for x in ids if x in complete]
    assert all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in scores.values())
    assert all(x['reason'] in ('tokens','context') and x['partial_score_used'] is False for x in receipt['skipped_candidates'])
    return registered_selector([c for c in candidates if c['id'] in complete],scores,split='tune')
