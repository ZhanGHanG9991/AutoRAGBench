"""Build-only Qwen reader. Never reads hidden data or serves an alternative model."""
import argparse,json,os,pathlib,sys,time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import httpx
from multireader.contract import pack_for_readers,messages,clean_answer,QWEN_TOKENIZER
from tasks import registry

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--questions',required=True);ap.add_argument('--evidence',required=True);ap.add_argument('--out',required=True);ap.add_argument('--gateway',required=True);ap.add_argument('--task',required=True);a=ap.parse_args()
    questions={q['qid']:q for q in map(json.loads,open(a.questions))}
    rows=[json.loads(l) for l in open(a.evidence)];fmt=registry.info(a.task)['answer_format']
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(QWEN_TOKENIZER,local_files_only=True,trust_remote_code=False)
    with httpx.Client(base_url=a.gateway,timeout=60,trust_env=False) as c,open(a.out,'w') as f:
        for row in rows:
            result={**row,'answer':None};qid=row['qid'];opened=False;t=time.time()
            try:
                if not row['error'] and not row['over_budget']:
                    c.post('/admin/open',json={'qid':qid+':reader'},headers={'X-Admin-Token':os.environ['GATEWAY_ADMIN_TOKEN']}).raise_for_status();opened=True
                    ps=pack_for_readers(questions[qid]['question'],row.get('passages',[]),fmt,tok,[tok])
                    response=c.post('/v1/chat/completions',json={'model':'generator','messages':messages(questions[qid]['question'],ps,fmt),'temperature':0,'max_tokens':512,'chat_template_kwargs':{'enable_thinking':False}})
                    response.raise_for_status();choice=response.json()['choices'][0];result['answer']=clean_answer(choice['message'].get('content') or '')
                    if not result['answer']:result['error']='empty_generation'
                    elif choice['finish_reason']=='length':result['error']='truncated_generation'
            except Exception as e:result['error']='reader:'+type(e).__name__
            finally:
                if opened:
                    stats=c.post('/admin/close',json={'qid':qid+':reader'},headers={'X-Admin-Token':os.environ['GATEWAY_ADMIN_TOKEN']}).json()
                    result['reader_gateway']=stats;result['over_budget']|=bool(stats.get('over_tokens') or stats.get('over_seconds'))
            result['reader_latency']=time.time()-t;f.write(json.dumps(result,ensure_ascii=False)+'\n');f.flush()

if __name__=='__main__':main()
