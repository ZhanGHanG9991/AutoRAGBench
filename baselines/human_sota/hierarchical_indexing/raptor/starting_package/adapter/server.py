import os
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel,Field
import uvicorn
from runtime import Engine
from provenance import Evidence,bounded
engine=Engine(Path(__file__).resolve().parents[1]);evidence=Evidence();app=FastAPI()
class Request(BaseModel):
    question:str
    meta:dict=Field(default_factory=dict)
@app.get('/health')
def health():return {'status':'ok','method':'RAPTOR'}
@app.post('/retrieve')
def retrieve(request:Request):
    _,selected,tree=engine.retrieve(request.question,request.meta)
    external_id=str(request.meta['doc_id']);seen=set();leaves=[]
    def descend(idx):
        if idx in seen:return
        seen.add(idx);node=tree.all_nodes[idx]
        if not node.children:leaves.append(node);return
        for child in sorted(node.children):descend(child)
    for item in selected:descend(item['node_index'])
    passages=bounded([evidence.leaf_passage(external_id,node.text) for node in leaves])
    if not passages:raise ValueError('empty leaf retrieval')
    return {'passages':passages}
if __name__=='__main__':uvicorn.run(app,host='127.0.0.1',port=int(os.environ['PORT']),access_log=False)
