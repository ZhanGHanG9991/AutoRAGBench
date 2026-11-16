"""Build registered dense models with source-exact 256/32 windows; Slurm only."""
import argparse,json,os
from pathlib import Path
from build_bm25_index import ROOT
from exp3lib.chunking import chunk_spans

def main():
    if not os.environ.get("SLURM_JOB_ID"): raise SystemExit("Slurm allocation required")
    ap=argparse.ArgumentParser(); ap.add_argument("--bench",required=True);ap.add_argument("--out",required=True);ap.add_argument("--limit",type=int,default=0);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    import subprocess,sys
    subprocess.run([sys.executable,str(Path(__file__).with_name("build_bm25_index.py")),"--bench",a.bench,"--out",str(out/"bm25"),"--limit",str(a.limit)],check=True)
    import faiss,numpy as np
    from sentence_transformers import SentenceTransformer
    from exp3lib.prefixes import apply_prefix
    prep=Path(os.environ["EXP3_PREP"])
    corpus=Path(os.environ["EXP3_CORPUS"])
    for name in ("e5-large-v2","bge-large-en-v1.5"):
        model=SentenceTransformer(str(Path(os.environ["EXP3_MODELS_DIR"])/name),local_files_only=True,device="cuda")
        assert model.tokenizer.is_fast
        index=None;ids=[];spans=[];texts=[];pending=[]
        def flush():
            nonlocal index
            if not texts:return
            vectors=np.asarray(model.encode(texts,batch_size=64,normalize_embeddings=True,show_progress_bar=False),dtype=np.float32)
            if index is None:index=faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
            for did,a,b in pending:ids.append(did);spans.append([a,b])
            texts.clear();pending.clear()
        with corpus.open() as handle:
            for n,line in enumerate(handle):
                if a.limit and n>=a.limit:break
                row=json.loads(line);text=row["text"]
                for start,end in chunk_spans(text,model.tokenizer):
                    texts.append(apply_prefix(text[start:end],name,"passage",bool(getattr(model,"prompts",None))))
                    pending.append((str(row["doc_id"]),start,end))
                    if len(texts)>=256:flush()
        flush()
        if index is None:raise RuntimeError("empty corpus")
        dest=out/name;dest.mkdir(parents=True,exist_ok=True)
        faiss.write_index(index,str(dest/"index.faiss"));(dest/"doc_ids.txt").write_text("\n".join(ids)+"\n");(dest/"spans.json").write_text(json.dumps(spans))
        print("dense index complete",name,"chunks",len(ids),flush=True)
        del model,index
        import gc,torch
        gc.collect();torch.cuda.empty_cache()
if __name__=="__main__":main()
