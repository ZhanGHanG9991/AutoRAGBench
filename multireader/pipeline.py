"""Extract once, seal evidence for the entire reader panel, answer, then score.

Hidden questions/evidence/answers stay under hidden/. records/ holds only aggregate metadata.
No target reader is exposed to a RAG service, and no target answer feeds back into retrieval.
"""
import argparse,concurrent.futures,hashlib,json,os,pathlib,subprocess,time,importlib.metadata
import httpx
from .contract import PROTOCOL,sha,messages,clean_answer,verify_corpus,pack_for_readers
from .server import ROOT,Server,config
from .efficiency import generation_counts
from tasks import registry
from tasks.score import score

def atomic(path,value):
    path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(path)

def jsonl(path):
    with open(path,encoding='utf-8') as f:return [json.loads(l) for l in f if l.strip()]

def write_jsonl(path,rows):
    path=pathlib.Path(path);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    tmp.replace(path)

def filehash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def paths(experiment):
    if not experiment or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in experiment):raise ValueError('Unsafe experiment name')
    hidden=ROOT/'hidden/multireader'/experiment;public=ROOT/'records/multireader'/experiment
    hidden.mkdir(parents=True,exist_ok=True,mode=0o700);public.mkdir(parents=True,exist_ok=True)
    return hidden,public

def extract(a):
    cases=json.loads(pathlib.Path(a.cases).read_text());hidden,public=paths(a.experiment)
    selected=[c for c in cases if not a.only or c['id'] in a.only.split(',')]
    if not selected:raise ValueError('No cases selected')
    with Server('qwen35_4b',public/'retrieval-services',retrieval=True) as server:
        for case in selected:
            name=case['id'];dest=hidden/name;rec=public/name;dest.mkdir(exist_ok=True);rec.mkdir(exist_ok=True)
            if (dest/'raw.jsonl').exists():raise FileExistsError('Evidence already exists: '+name)
            source=ROOT/'runs'/case['run_id'];ws=source/'workspace';orig=ROOT/'records'/case['source_run']
            if case.get('agent'):
                gate=json.loads((orig/'judge_gate.json').read_text())
                if gate.get('outcome')!='passed':raise RuntimeError('Original agent did not pass judge gate')
            hid=ROOT/'hidden/formal-v2'/case['task']
            qpath=hid/'test_questions.jsonl';gpath=hid/'test_gold.jsonl'
            if a.split=='train':
                qs=jsonl(source/'task/train.jsonl')[:a.limit];write_jsonl(dest/'gold.jsonl',qs)
                write_jsonl(dest/'questions.jsonl',[{k:q[k] for k in ['qid','question','meta'] if k in q} for q in qs]);qpath=dest/'questions.jsonl';gpath=dest/'gold.jsonl'
            qs=jsonl(qpath)
            if len({q['qid'] for q in qs})!=len(qs):raise ValueError('Duplicate question IDs')
            manifest=ROOT/'records'/case['run_id']/'workspace_manifest.json'
            cmd=[str(ROOT/'envs/vllm/bin/python'),str(ROOT/'scripts/freeze.py'),'verify',str(ws),str(manifest)]
            subprocess.run(cmd,check=True,stdout=(rec/'workspace-pre.txt').open('w'))
            env=server.gateway_env();env.update(RAGENG_RESPONSE_MODE='evidence',EVAL_WALL_SECONDS=str(a.eval_seconds))
            out=dest/'extract'
            subprocess.run(['bash',str(ROOT/'scripts/eval_system.sh'),sha(name)[:10],str(ws),str(source/'task/corpus'),str(qpath),str(out)],env=env,check=True)
            subprocess.run(cmd,check=True,stdout=(rec/'workspace-post.txt').open('w'))
            preds=jsonl(out/'predictions.jsonl');by={r['qid']:r for r in preds}
            if set(by)!={q['qid'] for q in qs}:raise ValueError('Question/evidence coverage mismatch')
            rows=[]
            for q in qs:
                p=by[q['qid']]
                row={'qid':q['qid'],'question':q['question'],'meta':q.get('meta',{}),'passages':p.get('passages',[]),
                     'error':p.get('error'),'over_budget':p.get('over_budget',False),
                     'retrieval_latency':p.get('latency'),'retrieval_tokens':(p.get('gateway') or {}).get('total_tokens',0),
                     'retrieval_calls':(p.get('gateway') or {}).get('calls',0)}
                if registry.info(case['task']).get('uses_meta_doc_id') and any(x['doc_id']!=str(row['meta'].get('doc_id')) for x in row['passages']):row['error']='wrong_document';row['passages']=[]
                rows.append(row)
            # line-level corpus evidence; document-scoped tasks keep lines in the cited document
            check=verify_corpus(rows,source/'task/corpus/corpus.jsonl',line_scope='claimed' if registry.info(case['task']).get('uses_meta_doc_id') else 'any')
            write_jsonl(dest/'raw.jsonl',rows)
            summary=json.loads((out/'predictions.jsonl.summary.json').read_text())
            meta={'protocol':PROTOCOL,'case':case,'split':a.split,'n':len(rows),'retrieval_model':'qwen35_4b',
                  'question_sha256':filehash(qpath),'gold_path':str(gpath),'workspace_manifest_sha256':filehash(manifest),
                  'raw_sha256':filehash(dest/'raw.jsonl'),'corpus_validation':check,'extraction':summary,
                  'n_errors':sum(bool(x['error'] or x['over_budget']) for x in rows)}
            atomic(dest/'meta.json',meta);atomic(rec/'extraction.json',meta)
            print('EXTRACTED',name,len(rows),'errors',meta['n_errors'],flush=True)

def token_count(tok,msg):
    return len(tok.apply_chat_template(msg,tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False))

def seal(a):
    # Bound the tokenizer's batch worker pool to this Slurm step's CPU budget.
    os.environ.setdefault('RAYON_NUM_THREADS',os.environ.get('SLURM_CPUS_PER_TASK','1'))
    from transformers import AutoTokenizer
    hidden,public=paths(a.experiment);cfg=config();panel=a.readers.split(',')
    tokenizer_start=time.monotonic()
    print('TOKENIZERS_LOADING',len(dict.fromkeys(['qwen35_4b']+panel)),flush=True)
    toks={k:AutoTokenizer.from_pretrained(cfg['readers'][k]['path'],local_files_only=True,trust_remote_code=False) for k in dict.fromkeys(['qwen35_4b']+panel)}
    tokenizer_load_seconds=time.monotonic()-tokenizer_start
    print('TOKENIZERS_READY',round(tokenizer_load_seconds,3),flush=True)
    for dest in sorted(hidden.iterdir()):
        if not (dest/'raw.jsonl').exists() or (a.only and dest.name not in a.only.split(',')):continue
        if (dest/'evidence.jsonl').exists():raise FileExistsError('Sealed evidence already exists')
        meta=json.loads((dest/'meta.json').read_text());task=meta['case']['task'];fmt=registry.info(task)['answer_format']
        if filehash(dest/'raw.jsonl')!=meta['raw_sha256']:raise ValueError('Raw evidence changed after extraction')
        rows=jsonl(dest/'raw.jsonl');packed=[];truncated=0;packing_start=time.monotonic()
        print('PACKING_START',dest.name,len(rows),flush=True)
        for i,row in enumerate(rows,1):
            ps=pack_for_readers(row['question'],row['passages'],fmt,toks['qwen35_4b'],list(toks.values()),batch=True)
            truncated+=ps!=row['passages']
            msg=messages(row['question'],ps,fmt)
            out={**row,'passages':ps,'messages':msg};out['evidence_sha256']=sha({'qid':out['qid'],'messages':msg,'passages':ps,'protocol':PROTOCOL})
            packed.append(out)
            if i%100==0 or i==len(rows):
                print('PACKING_PROGRESS',dest.name,i,len(rows),'seconds',round(time.monotonic()-packing_start,3),flush=True)
        packing_seconds=time.monotonic()-packing_start
        write_jsonl(dest/'evidence.jsonl',packed)
        meta.update(panel=panel,evidence_sha256=filehash(dest/'evidence.jsonl'),n_context_truncated=truncated,
                    packing_implementation='batched-prefix-v1',packing_seconds=packing_seconds,
                    tokenizer_load_seconds=tokenizer_load_seconds)
        atomic(dest/'meta.json',meta);atomic(public/dest.name/'evidence.json',meta)
        print('SEALED',dest.name,len(packed),'context_truncated',truncated,flush=True)

def read(a):
    hidden,public=paths(a.experiment);profile=config()['readers'][a.reader]
    model_manifest=ROOT/'configs/model-manifest.json'  # sha256 of the reader weights used in the paper
    fingerprint=json.loads(model_manifest.read_text())[a.reader] if model_manifest.exists() else None
    versions={k:importlib.metadata.version(k) for k in ['vllm','transformers','torch']}
    with Server(a.reader,public/('reader-services-'+a.reader)) as server:
        with httpx.Client(base_url=server.url,headers={'Authorization':'Bearer '+server.key},timeout=60,trust_env=False) as c:
            smoke=c.post('/v1/chat/completions',json={'model':'generator','messages':[{'role':'user','content':'Reply with only the word READY.'}],'max_tokens':32,'temperature':0,'chat_template_kwargs':{'enable_thinking':False}})
            smoke.raise_for_status();s=smoke.json();text=s['choices'][0]['message'].get('content')
            if not text or s['choices'][0]['finish_reason']=='length':raise RuntimeError('Reader smoke has empty/truncated output')
            atomic(public/('smoke-'+a.reader+'.json'),{'reader':a.reader,'output':text,'finish_reason':s['choices'][0]['finish_reason']})
            warm=c.post('/v1/chat/completions',json={'model':'generator','messages':[{'role':'user','content':('A library contains books on many subjects. '*1500)+'\nReply with only READY.'}],'max_tokens':32,'temperature':0,'chat_template_kwargs':{'enable_thinking':False}})
            warm.raise_for_status()
        for dest in sorted(hidden.iterdir()):
            if not (dest/'evidence.jsonl').exists() or (a.only and dest.name not in a.only.split(',')):continue
            meta=json.loads((dest/'meta.json').read_text())
            if a.reader not in meta['panel']:raise ValueError('Reader was not included in common packing panel')
            if filehash(dest/'evidence.jsonl')!=meta['evidence_sha256']:raise ValueError('Evidence changed after sealing')
            out=dest/a.reader;out.mkdir(exist_ok=True)
            if (out/'scores.json').exists():raise FileExistsError('Reader results already exist')
            rows=jsonl(dest/'evidence.jsonl');start=time.time()
            def answer(row):
                rec={'qid':row['qid'],'answer':'','error':row['error'],'over_budget':row['over_budget'],'evidence_sha256':row['evidence_sha256']}
                expected=sha({'qid':row['qid'],'messages':row['messages'],'passages':row['passages'],'protocol':PROTOCOL})
                if expected!=row['evidence_sha256']:raise ValueError('Row evidence checksum mismatch')
                if rec['error'] or rec['over_budget']:return rec
                t=time.time()
                try:
                    with httpx.Client(timeout=60,trust_env=False) as c:
                        response=c.post(server.url+'/v1/chat/completions',headers={'Authorization':'Bearer '+server.key},json={
                            'model':'generator','messages':row['messages'],'temperature':0,'max_tokens':512,
                            'chat_template_kwargs':{'enable_thinking':False}})
                    response.raise_for_status();d=response.json();choice=d['choices'][0];raw=choice['message'].get('content') or ''
                    rec.update(answer=clean_answer(raw),raw_answer=raw,finish_reason=choice['finish_reason'],usage=d.get('usage',{}))
                    if not raw.strip():rec['error']='empty_generation'
                    elif choice['finish_reason']=='length':rec['error']='truncated_generation'
                    elif choice['message'].get('reasoning') or choice['message'].get('reasoning_content') or '<think>' in raw:rec['error']='unexpected_thinking'
                except Exception as e:rec['error']=type(e).__name__
                rec['latency']=time.time()-t
                return rec
            preds=[]
            with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as pool,(out/'predictions.jsonl.part').open('w') as f:
                for i,p in enumerate(pool.map(answer,rows),1):
                    preds.append(p);f.write(json.dumps(p,ensure_ascii=False)+'\n');f.flush()
                    if i%100==0:print('READER_PROGRESS',a.reader,dest.name,i,len(rows),flush=True)
            (out/'predictions.jsonl.part').replace(out/'predictions.jsonl')
            gold=jsonl(meta['gold_path'])
            if len(preds)!=len(gold) or len({p['qid'] for p in preds})!=len(preds) or {p['qid'] for p in preds}!={g['qid'] for g in gold}:raise ValueError('Prediction/gold coverage mismatch')
            result=score(meta['case']['task'],{p['qid']:p for p in preds},gold)
            elapsed=time.time()-start
            lat=sorted(p['latency'] for p in preds if 'latency' in p)
            usage={k:sum((p.get('usage') or {}).get(k,0) or 0 for p in preds) for k in ['prompt_tokens','completion_tokens','total_tokens']}
            result.update(reader=a.reader,reader_path=profile['path'],protocol=PROTOCOL,source=meta['case'],
                          evidence_sha256=meta['evidence_sha256'],elapsed_seconds=elapsed,usage=usage,
                          latency_mean=sum(lat)/len(lat) if lat else None,latency_p50=lat[len(lat)//2] if lat else None,
                          latency_p95=lat[min(len(lat)-1,int(len(lat)*.95))] if lat else None,
                          completion_tokens_per_second=usage['completion_tokens']/elapsed,
                          concurrency=a.concurrency,
                          max_output_tokens=512,temperature=0,thinking=False,dtype='bfloat16')
            result.update(generation_counts(preds,elapsed))
            result.update(model_fingerprint=fingerprint,software_versions=versions,model_startup_seconds=server.startup_seconds,
                          slurm_job=os.environ.get('SLURM_JOB_ID'),node=os.uname().nodename)
            atomic(out/'scores.json',result)
            atomic(public/dest.name/('scores-'+a.reader+'.json'),{k:v for k,v in result.items() if k!='per_question'})
            print('SCORED',a.reader,dest.name,result['primary'],'invalid',result['n_invalid'],flush=True)
            if result['n_generation_errors']:print('READER_ERRORS',a.reader,dest.name,result['n_generation_errors'],'retained and scored as invalid',flush=True)

def main():
    os.umask(0o077)
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='command',required=True)
    for name in ['extract','seal','read']:
        p=sub.add_parser(name);p.add_argument('--experiment',required=True);p.add_argument('--only',default='')
        if name=='extract':
            p.add_argument('--cases',required=True);p.add_argument('--split',choices=['train','test'],default='test');p.add_argument('--limit',type=int,default=20);p.add_argument('--eval-seconds',type=int,default=14400)
        elif name=='seal':p.add_argument('--readers',default=','.join(config()['readers']))
        else:p.add_argument('--reader',choices=config()['readers'],required=True);p.add_argument('--concurrency',type=int,default=4)
    a=ap.parse_args();globals()[a.command](a)

if __name__=='__main__':main()
