"""One frozen agent delivery, all held-out readers. Called after the original judge gate."""
import argparse,json,os,pathlib,subprocess,sys,time
from types import SimpleNamespace
from .pipeline import ROOT,paths,atomic,extract,seal,read,filehash
from .contract import PROTOCOL

PANEL=['gemma4_e4b','gemma3_4b','phi4_mini','smollm3_3b','llama32_3b']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-id',required=True);a=ap.parse_args()
    rec=ROOT/'records'/a.run_id;meta=json.loads((rec/'run.json').read_text())
    assert meta.get('evaluation_protocol')==PROTOCOL
    experiment=a.run_id;hidden,public=paths(experiment);adapter=rec/'adapter.json'  # reference systems / trusted adapters have no agent judge gate
    agent=json.loads(adapter.read_text()).get('agent',False) if adapter.exists() else True
    case=dict(id=meta['task'],task=meta['task'],run_id=a.run_id,source_run=a.run_id,agent=agent)
    cases=public/'cases.json'
    if cases.exists():assert json.loads(cases.read_text())==[case]
    else:atomic(cases,[case])
    dest=hidden/case['id'];start=time.time()
    if not (dest/'raw.jsonl').exists():
        extract(SimpleNamespace(experiment=experiment,cases=str(cases),only='',split='test',limit=0,eval_seconds=int(os.environ.get('EVAL_SECONDS','14400'))))
    if not (dest/'evidence.jsonl').exists():
        seal(SimpleNamespace(experiment=experiment,only='',readers=','.join(PANEL)))
    # Resume only finished reader artifacts that match this exact sealed evidence.
    evidence=json.loads((dest/'meta.json').read_text());scores=[]
    assert filehash(dest/'evidence.jsonl')==evidence['evidence_sha256']
    for reader in PANEL:
        score_path=dest/reader/'scores.json'
        if not score_path.exists():
            read(SimpleNamespace(experiment=experiment,only=case['id'],reader=reader,concurrency=4))
        score=json.loads(score_path.read_text())
        assert score['evidence_sha256']==evidence['evidence_sha256'] and score['reader']==reader
        scores.append({k:v for k,v in score.items() if k!='per_question'})
    result={'protocol':PROTOCOL,'run_id':a.run_id,'task':meta['task'],'readers':scores,'reader_macro_primary':sum(s['primary'] for s in scores)/len(scores),
            'retrieval':evidence,'elapsed_seconds_this_attempt':time.time()-start,'n_invalid_total':sum(s['n_invalid'] for s in scores)}
    atomic(rec/'multireader_scores.json',result)
    atomic(rec/'run_result.json',{'outcome':'evaluated_multireader','protocol':PROTOCOL,'primary':result['reader_macro_primary'],'readers':PANEL,'job':os.environ.get('SLURM_JOB_ID'),'n_invalid_total':result['n_invalid_total'],'time':time.strftime('%FT%T%z')})
    print('MULTIREADER_COMPLETE',a.run_id,'reader_macro',result['reader_macro_primary'],'invalid',result['n_invalid_total'],flush=True)

if __name__=='__main__':main()
