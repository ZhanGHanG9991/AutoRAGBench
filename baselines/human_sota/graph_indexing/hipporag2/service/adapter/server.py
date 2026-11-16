import json
import os
import time
import traceback
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from runtime import BGE, GatewayLLM, config, instrument, load_rag, retrieval_query
from fe_corpus import corpus_evidence, check_limits

W=Path(__file__).resolve().parents[1]
INFO=json.loads((W/'submission/task.json').read_text())
mapping=json.loads((W/'index/mapping.json').read_text())
embed=BGE(W/'models/bge-base-en-v1.5')
llm=GatewayLLM(os.environ['LLM_BASE_URL'],config(W/'index'))
current={}
def get_rag(key):
    if key not in mapping: raise ValueError('Unknown/missing required meta.doc_id')
    if key not in current:
        current.clear()
        r=instrument(load_rag(W/'index/indexes'/mapping[key],llm,embed,readonly=True))
        r.prepare_retrieval_objects()
        current[key]=r
    return current[key]
if INFO['name']!='qasper': get_rag('global')

def retrieve(question,meta):
    started=time.monotonic()
    llm.calls=llm.tokens=0; llm.events=[]; llm.deadline=started+57
    key=str(meta.get('doc_id','')) if INFO['name']=='qasper' else 'global'
    rag=get_rag(key); rag.trace={}
    linked_query=retrieval_query(question,meta,INFO['name'])
    result=rag.retrieve([linked_query],num_to_retrieve=5)[0]
    if INFO['name']=='qasper':
        assert all(str(m['doc_id'])==key for m in result.doc_metadata)
    evidence=corpus_evidence();passages=[]
    for text,item in zip(result.docs,result.doc_metadata):
        passages.append(evidence.chunk_passage(text,str(item['doc_id'])))
    trace=dict(task=INFO['name'],doc_id=key,question=question,retrieval_query=linked_query,
        citations=[p['doc_id'] for p in passages],seconds=time.monotonic()-started,
        calls=llm.calls,tokens=llm.tokens,dense_fallback=rag.trace.get('run_ppr',0)==0,**rag.trace)
    print('HIPPO_FIXED_EVIDENCE '+json.dumps(trace),flush=True)
    if llm.tokens>32000: raise RuntimeError('Token budget exceeded')
    return {'passages':check_limits(passages)}

def answer(question,meta):
    started=time.monotonic()
    llm.calls=llm.tokens=0; llm.events=[]; llm.deadline=started+57
    key=str(meta.get('doc_id','')) if INFO['name']=='qasper' else 'global'
    rag=get_rag(key); rag.trace={}
    linked_query=retrieval_query(question,meta,INFO['name'])
    result=rag.retrieve([linked_query],num_to_retrieve=5)[0]
    if INFO['name']=='qasper':
        assert all(m['doc_id']==key for m in result.doc_metadata)
    context='\n\n'.join(f'Passage {i+1}:\n{d}' for i,d in enumerate(result.docs))
    messages=[{'role':'system','content':'Answer the question using the provided passages. Return only the final answer. '+INFO['answer_format']},
        {'role':'user','content':context+'\n\nQuestion: '+question}]
    raw,usage,_=llm.infer(messages,max_new_tokens=512)
    answer=raw.strip()
    trace=dict(task=INFO['name'],doc_id=key,question=question,retrieval_query=linked_query,answer=answer,
        citations=[m.get('doc_id') for m in result.doc_metadata],
        seconds=time.monotonic()-started,calls=llm.calls,tokens=llm.tokens,
        dense_fallback=rag.trace.get('run_ppr',0)==0,**rag.trace)
    print('HIPPO_TRACE '+json.dumps(trace),flush=True)
    if not answer: raise RuntimeError('Empty fixed-model final output')
    if llm.tokens>32000: raise RuntimeError('Token budget exceeded')
    return {'answer':answer,'citations':trace['citations']}

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path=='/health' else 404);self.end_headers()
    def do_POST(self):
        try:
            if self.path not in ('/answer','/retrieve'): raise ValueError('Unknown route')
            data=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            fn=retrieve if self.path=='/retrieve' else answer
            body=json.dumps(fn(data['question'],data.get('meta') or {})).encode()
            self.send_response(200)
        except Exception:
            traceback.print_exc();body=b'{"error":"answer_failed"}';self.send_response(500)
        self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(body)

HTTPServer(('127.0.0.1',int(os.environ['PORT'])),Handler).serve_forever()
