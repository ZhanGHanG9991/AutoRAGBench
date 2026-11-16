"""Bounded RAPTOR forest for full public corpora, no QA-driven filtering.

All leaves and every recursive summary participate in collapsed retrieval.
Partition boundaries are deterministic corpus-order chunks, a documented
scalability adaptation. No test queries or answers are accepted by this program.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sqlite3
import time
import numpy as np
import faiss
from build import build_chunks, chunk_documents
from runtime import SBERT


def make_index(out, config):
    vectors=np.memmap(out/'vectors.f32',dtype='float32',mode='r').reshape(-1,768)
    n=len(vectors)
    faiss.omp_set_num_threads(8)
    if n <= 200000:
        index=faiss.IndexFlatIP(768)
        description={'type':'IndexFlatIP','exact':True}
    else:
        nlist=2048
        quantizer=faiss.IndexFlatIP(768)
        index=faiss.IndexIVFScalarQuantizer(quantizer,768,nlist,faiss.ScalarQuantizer.QT_fp16,faiss.METRIC_INNER_PRODUCT)
        index.cp.seed=config['seed']; index.cp.niter=20
        rng=np.random.default_rng(config['seed'])
        selected=np.sort(rng.choice(n,size=min(n,131072),replace=False))
        index.train(np.asarray(vectors[selected],dtype='float32'))
        index.nprobe=64
        description={'type':'IndexIVFScalarQuantizer','exact':False,'quantization':'fp16',
                     'nlist':nlist,'nprobe':64,'training_nodes':len(selected),'seed':config['seed']}
    for start in range(0,n,32768):
        index.add(np.asarray(vectors[start:start+32768],dtype='float32'))
    faiss.write_index(index,str(out/'search.faiss'))
    loaded=faiss.read_index(str(out/'search.faiss'))
    assert loaded.ntotal==n
    (out/'search_config.json').write_text(json.dumps(description,indent=2))
    return description


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--corpus',required=True);p.add_argument('--out',required=True)
    p.add_argument('--models',required=True);p.add_argument('--config',required=True)
    p.add_argument('--task',required=True)
    a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE').exists(): raise RuntimeError('Completed index must not be overwritten')
    config=json.loads(Path(a.config).read_text())
    block_size=config['collection_leaf_block_size']
    embedding=SBERT(Path(a.models)/'multi-qa-mpnet-base-cos-v1','cuda')
    spool=out/'trees';spool.mkdir(exist_ok=True)
    start=time.time();blocks=0;documents=0;leaves=0;pending=[]
    digest=hashlib.sha256()

    def flush(items):
        nonlocal blocks,leaves
        dest=spool/f'{blocks:07d}.pkl'
        item_hash=hashlib.sha256(json.dumps(items,ensure_ascii=False).encode()).hexdigest()
        if dest.exists():
            saved=pickle.load(dest.open('rb'))
            assert saved['input_sha256']==item_hash and saved['config']==config
        else:
            with (spool/f'{blocks:07d}.calls.jsonl').open('w') as log:
                tree=build_chunks(items,embedding,config,log)
            saved={'input_sha256':item_hash,'config':config,'tree':tree}
            temp=dest.with_suffix('.tmp')
            with temp.open('wb') as f:pickle.dump(saved,f,protocol=5)
            temp.rename(dest)
        blocks+=1;leaves+=len(items)
        (out/'progress.json').write_text(json.dumps({'task':a.task,'stage':'trees',
            'documents_seen':documents,'blocks':blocks,'leaves':leaves,'seconds_current_job':time.time()-start,
            'job_id':os.environ.get('SLURM_JOB_ID')}))
        print('RAPTOR_BLOCK',blocks,'leaves',leaves,'seconds',round(time.time()-start,1),flush=True)

    with open(a.corpus,'rb') as f:
        for line in f:
            digest.update(line);doc=json.loads(line);documents+=1
            if a.task in ('webqsp','cwq'):doc['kg']=True
            pending.extend(chunk_documents([doc],config))
            while len(pending)>=block_size:
                flush(pending[:block_size]);pending=pending[block_size:]
    if pending:flush(pending)
    manifest_path=Path(a.corpus).parent.parent/'data_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    assert digest.hexdigest()==manifest['corpus_sha256']
    assert documents==manifest['n_corpus']
    # KG file is the identical edge corpus in tab form. Validate count and digest,
    # never traverse the KG or selectively expand it using question entities.
    kg_info=None
    if a.task in ('webqsp','cwq'):
        triples=Path(a.corpus).with_name('triples.tsv');h=hashlib.sha256();edges=0
        with triples.open('rb') as f:
            for line in f:
                h.update(line);assert len(line.rstrip(b'\n').split(b'\t'))==3;edges+=1
        corpus_edges=0
        with open(a.corpus) as f:
            # Preparation joins facts with a literal LF. ``str.splitlines``
            # also treats Unicode NEL (U+0085) inside a Freebase entity label
            # as a row boundary, over-counting nine CWQ and six WebQSP facts.
            # Count only the delimiter used by tasks/kgqa.py so this validates
            # the actual triples.tsv rows without changing indexed content.
            for line in f:corpus_edges+=len(json.loads(line)['text'].split('\n'))
        assert corpus_edges==edges,(corpus_edges,edges)
        kg_info={'triples':edges,'triples_sha256':h.hexdigest(),'entity_text_lines':corpus_edges}
    dbpath=out/'nodes.sqlite'
    # Assembly can be safely rerun from immutable tree checkpoints.
    if dbpath.exists():dbpath.unlink()
    db=sqlite3.connect(dbpath)
    db.execute('CREATE TABLE nodes(id INTEGER PRIMARY KEY,text TEXT,layer INTEGER,children TEXT,doc_ids TEXT)')
    roots=[];counts={};offset=0
    with (out/'vectors.f32').open('wb') as vector_file:
        for file in sorted(spool.glob('*.pkl')):
            tree=pickle.load(file.open('rb'))['tree']
            mapping={n.index:k for k,ns in tree.layer_to_nodes.items() for n in ns}
            records=[];vectors=[]
            for idx,n in sorted(tree.all_nodes.items()):
                assert idx==len(vectors)
                records.append((offset+idx,n.text,mapping[idx],json.dumps(sorted(offset+c for c in n.children)),json.dumps(n.doc_ids)))
                vectors.append(n.embeddings['SBERT'])
                level=str(mapping[idx]);counts[level]=counts.get(level,0)+1
            db.executemany('INSERT INTO nodes VALUES(?,?,?,?,?)',records);db.commit()
            np.asarray(vectors,dtype='float32').tofile(vector_file)
            roots.extend(offset+i for i in tree.root_nodes)
            offset+=len(tree.all_nodes)
    db.close()
    assert sum(n for k,n in counts.items() if int(k)>0)>0
    np.save(out/'roots.npy',np.asarray(roots,dtype='int64'))
    search=make_index(out,config)
    stats={'task':a.task,'scope':'full','documents':documents,'leaves':leaves,'trees':blocks,
           'layer_nodes':counts,'max_summary_depth':max(map(int,counts)),
           'corpus_sha256':digest.hexdigest(),'kg':kg_info,'config':config,'search':search,
           'seconds_current_job':time.time()-start,'job_id':os.environ.get('SLURM_JOB_ID'),
           'embedding_parameters':embedding.parameter_count,'serialization_verified':True}
    (out/'index_stats.json').write_text(json.dumps(stats,indent=2))
    (out/'catalog.json').write_text(json.dumps({'task':a.task,'scope':'full','storage':'sqlite_faiss','nodes':offset}))
    (out/'COMPLETE').write_text('Full corpus coverage, recursive summaries and search reload checked.\n')

if __name__=='__main__':main()
