"""Minimal evidence contract; replace empty evidence with real corpus retrieval."""
import os
import uvicorn
from fastapi import FastAPI
app=FastAPI()
@app.get('/health')
def health():return {'ok':True}
@app.post('/retrieve')
def retrieve(req:dict):return {'passages':[]}
if __name__=='__main__':uvicorn.run(app,host='127.0.0.1',port=int(os.environ['PORT']),log_level='warning')
