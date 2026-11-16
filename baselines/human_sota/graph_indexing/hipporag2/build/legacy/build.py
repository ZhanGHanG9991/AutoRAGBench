import argparse
import atexit
from collections import Counter
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path
from runtime import BGE, GatewayLLM, chunks, config, load_rag, normalize_openie_unicode

p=argparse.ArgumentParser()
p.add_argument('--task',required=True)
p.add_argument('--corpus',required=True)
p.add_argument('--out',required=True)
p.add_argument('--models',required=True)
p.add_argument('--llm',required=True)
p.add_argument('--limit',type=int,default=0)
p.add_argument('--data-version',default='formal-v1')
a=p.parse_args()
out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
assert not (out/'COMPLETE').exists(), 'Completed index must not be overwritten'
t=time.monotonic()
started=time.time()
embed=BGE(a.models)
llm=GatewayLLM(a.llm,config(out),out/'call-cache')
attempt=out/('attempt-'+os.environ.get('HIPPO_JOB_ID','manual')+'-'+str(int(started))+'.json')
def save_attempt():
    attempt.write_text(json.dumps(dict(started=started,ended=time.time(),seconds=time.monotonic()-t,
        job_id=os.environ.get('HIPPO_JOB_ID'),gpu_count=1,hardware=os.environ.get('HIPPO_GPU_NAME'),
        calls=llm.calls,tokens=llm.tokens,complete=(out/'COMPLETE').exists()),indent=2))
atexit.register(save_attempt)
signal.signal(signal.SIGTERM,lambda *unused: sys.exit(143))
if a.task == 'qasper':
    rows=[]
    with open(a.corpus) as f:
        for line in f:
            rows.append(json.loads(line))
            if a.limit and len(rows)>=a.limit: break
    source_documents=len(rows)
    groups=[(str(r['doc_id']), chunks(r, a.task)) for r in rows]
else:
    # The global index still receives the exact same ordered chunk list, but do
    # not retain a second, multi-million-item list of decoded corpus rows while
    # building it.  This is only a representation/lifetime optimization; it
    # does not shard the graph or alter OpenIE, embeddings, edges, or PPR.
    docs=[]
    source_documents=0
    with open(a.corpus) as f:
        for line in f:
            docs.extend(chunks(json.loads(line), a.task))
            source_documents+=1
            if a.limit and source_documents>=a.limit: break
    groups=[('global', docs)]
mapping={}; stats=[]; overall_edges=Counter()
for key,docs in groups:
    sub=hashlib.sha256(key.encode()).hexdigest()[:24] if a.task=='qasper' else 'global'
    idx=out/'indexes'/sub
    rag=load_rag(idx,llm,embed)
    # A previous interrupted build may have saved vectors before graph creation.
    # Explicitly rebuild the missing derived graph, reusing only same-manifest
    # embeddings and cached model responses. Never delete or replace old results.
    if not rag._graph_state_available and rag.chunk_embedding_store.get_all_ids():
        rag.global_config.force_index_from_scratch=True
    # Fail closed on OpenIE parse/transport errors; do not publish a silently incomplete graph.
    original=rag.openie.batch_openie
    def checked(batch):
        ner,triples=original(batch)
        errors=[{'chunk':k,'stage':stage,'error':r.metadata['error']} for stage,d in [('ner',ner),('triples',triples)] for k,r in d.items() if r.metadata.get('error')]
        if errors:
            (out/'openie_errors.json').write_text(json.dumps(errors,indent=2))
            raise RuntimeError(f'{len(errors)} OpenIE errors; see openie_errors.json')
        repairs=normalize_openie_unicode(ner,triples)
        if repairs:
            print(f'OPENIE_UNICODE_REPAIRS={repairs}',flush=True)
        return ner,triples
    rag.openie.batch_openie=checked
    rag.index(docs)
    rag.prepare_retrieval_objects()
    edges=Counter()
    for edge in rag.graph.es:
        for typ in str(edge['edge_kind']).split('+'):
            edges[typ]+=1
    overall_edges.update(edges)
    s=dict(key=key,passages=len(rag.passage_node_keys),phrases=len(rag.entity_node_keys),facts=len(rag.fact_node_keys),nodes=rag.graph.vcount(),edges=rag.graph.ecount(),edge_attributes=rag.graph.es.attributes(),edge_types=dict(edges))
    # A one-passage QASPER paper can legitimately have no pair of phrases
    # above the official synonym threshold.  Require semantic/context edges
    # per independent paper, and enforce synonym edges over the complete
    # benchmark index below.
    for required in ('fact', 'passage'):
        if not edges[required]:
            raise RuntimeError(f'Missing required {required} edges in {key}: {dict(edges)}')
    stats.append(s); mapping[key]=sub
    (out/'graphs.json').write_text(json.dumps(stats,indent=2))
    (out/'mapping.json').write_text(json.dumps(mapping))
    rag.close()
total={k:sum(s[k] for s in stats) for k in ('passages','phrases','facts','nodes','edges')}
assert total['passages'] and total['phrases'] and total['facts'] and total['edges'], total
if not overall_edges['synonym']:
    raise RuntimeError(
        f'Missing required synonym edges in complete index: {dict(overall_edges)}')
corpus_hash=hashlib.sha256()
with open(a.corpus,'rb') as f:
    for block in iter(lambda: f.read(8 << 20), b''):
        corpus_hash.update(block)
summary=dict(task=a.task,data_version=a.data_version,source_documents=source_documents,limit=a.limit,complete=True,
    corpus_sha256=corpus_hash.hexdigest(),
    seconds=time.monotonic()-t,calls=llm.calls,tokens=llm.tokens,
    embedding_parameters=embed.parameters,**total)
summary['storage_bytes']=sum(f.stat().st_size for f in out.rglob('*') if f.is_file())
prior=[json.loads(f.read_text()) for f in out.glob('attempt-*.json')]
summary['cumulative_seconds']=summary['seconds']+sum(x['seconds'] for x in prior)
summary['cumulative_calls']=llm.calls+sum(x['calls'] for x in prior)
summary['cumulative_tokens']=llm.tokens+sum(x['tokens'] for x in prior)
summary['cumulative_index_gpu_hours']=summary['cumulative_seconds']/3600
summary['index_bytes']=sum(f.stat().st_size for f in (out/'indexes').rglob('*') if f.is_file())
(out/'index_stats.json').write_text(json.dumps(summary,indent=2))
(out/'COMPLETE').write_text('Complete corpus subset' if a.limit else 'Complete corpus')
print('INDEX_COMPLETE',json.dumps(summary),flush=True)
