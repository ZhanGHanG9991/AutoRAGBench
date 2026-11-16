import json,os,time,traceback
from pathlib import Path
from http.server import BaseHTTPRequestHandler,HTTPServer
from runtime import BGE,GatewayLLM,config,instrument,load_rag,retrieval_query
from provenance import Evidence,bounded
W=Path(__file__).resolve().parents[1]
INFO=json.loads((W/'submission/task.json').read_text())
mapping=json.loads((W/'index/mapping.json').read_text())
embed=BGE(W/'models/bge-base-en-v1.5')
llm=GatewayLLM(os.environ['LLM_BASE_URL'],config(W/'index'))
rag=instrument(load_rag(W/'index/indexes'/mapping['global'],llm,embed,readonly=True))
rag.prepare_retrieval_objects()
evidence=Evidence(hippo=True)
def retrieve(question,meta):
    started=time.monotonic();llm.calls=llm.tokens=0;llm.events=[];llm.deadline=started+57
    rag.trace={}
    result=rag.retrieve([retrieval_query(question,meta,INFO['name'])],num_to_retrieve=5)[0]
    passages=bounded([evidence.hippo_passage(d) for d in result.docs])
    if not passages:raise ValueError('empty retrieval')
    if llm.calls>4 or llm.tokens>32000:raise ValueError('retrieval budget exceeded')
    print(json.dumps({'event':'retrieve','passages':len(passages),'calls':llm.calls,'tokens':llm.tokens,'ppr_calls':rag.trace.get('run_ppr',0)}),flush=True)
    return {'passages':passages}
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def send(self,code,body):
        raw=json.dumps(body).encode();self.send_response(code);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
    def do_GET(self):self.send(200 if self.path=='/health' else 404,{'status':'ok','method':'HippoRAG2'})
    def do_POST(self):
        if self.path!='/retrieve':self.send(404,{'error':'route'});return
        try:
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            self.send(200,retrieve(body['question'],body.get('meta') or {}))
        except Exception as exc:
            print('RETRIEVE_ERROR '+type(exc).__name__,flush=True);self.send(500,{'error':type(exc).__name__})
HTTPServer(('127.0.0.1',int(os.environ['PORT'])),Handler).serve_forever()
