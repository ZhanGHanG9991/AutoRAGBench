"""Rebuild corpus-specific SubgraphRAG vectors, preserving the learned MLP."""
import hashlib,json,os,shutil,sys
from pathlib import Path
import numpy as np
from encoder import load_encoder,embed_texts
from graph import KG
W=Path(__file__).resolve().parents[1];out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
triples=Path(os.environ['CORPUS_DIR'])/'triples.tsv'
assert 'hidden' not in triples.parts and 'test' not in triples.name
kg=KG(str(triples));tok,model,n=load_encoder(str(W/'models/encoder'),'cuda')
def digest_names(values):
    h=hashlib.sha256()
    for value in values:h.update(value.encode());h.update(b'\n')
    return h.hexdigest()
texts=[kg.names[i] for i in kg.text_ids];rels=sorted(kg.rel_names)
np.save(out/'ent_emb.f16.npy',embed_texts(tok,model,texts,batch=128,max_length=128))
np.save(out/'rel_emb.f16.npy',embed_texts(tok,model,rels,batch=128,max_length=128))
shutil.copy2(W/'artifacts/ckpt.pt',out/'ckpt.pt')
meta=json.loads((W/'artifacts/meta.json').read_text())
meta.update(n_text=len(texts),n_rel=len(rels),n_triples=len(kg.heads),text_sha256=digest_names(texts),rel_sha256=digest_names(rels),view='rebuilt corpus embeddings; inherited train-only MLP')
h=hashlib.sha256()
with triples.open('rb') as f:
    for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
meta['triples_sha256']=h.hexdigest();(out/'meta.json').write_text(json.dumps(meta,indent=2))
