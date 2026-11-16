"""Corpus-only offline construction, official extraction + advanced ER, checkpointed stages."""
from __future__ import annotations
import argparse, json, os, time, pickle, hashlib, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from runtime import *


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--task',required=True); ap.add_argument('--corpus',required=True)
    ap.add_argument('--out',required=True); ap.add_argument('--models',required=True); ap.add_argument('--llm',required=True)
    ap.add_argument('--limit',type=int,default=0); ap.add_argument('--seconds',type=int,default=7200)
    ap.add_argument('--workers',type=int,default=128); ap.add_argument('--er-scope',choices=['document','corpus'],default='document'); args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    (out/'COMPLETE').unlink(missing_ok=True)
    start=time.time(); deadline=start+args.seconds
    llm=FixedLLM(args.llm,out/'calls.jsonl',output_tokens=4000,build_mode=True)
    cfg=GraphConfig(extractor_type='llm',image_description_force=False,max_gleaning=0)
    tree,node_doc,docs,leaves=build_tree(args.corpus,out,args.task,args.limit)
    corpus_sha=hashlib.sha256(Path(args.corpus).read_bytes()).hexdigest()
    if (out/'index_stats.json').exists():
        previous=json.loads((out/'index_stats.json').read_text())
        if previous['corpus_sha256']!=corpus_sha or previous['limited_documents']!=args.limit:
            raise RuntimeError('Refusing extraction-cache reuse across different corpus hashes or document limits')
    tree.save_to_file()
    with (out/'tree_meta.pkl').open('wb') as f: pickle.dump((node_doc,docs,leaves),f)
    extractor=KGExtractor(cfg,llm=llm,vlm=None,save_path=str(out))
    # Include source title in extraction context, while the saved tree remains verbatim corpus text.
    def extract(nid):
        if time.time()>deadline-180: raise TimeoutError('build budget reached during extraction')
        n=tree.nodes[nid]
        contextual=TreeNode(n.meta_info.model_dump()); contextual.type=n.type; contextual.index_id=nid
        contextual.meta_info.content=docs[node_doc[nid]].get('title','')+'\n'+n.meta_info.content
        return extractor.extract_kg(contextual)
    done=0; errors=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(extract,nid):nid for nid in leaves}
        for future in as_completed(futures):
            nid=futures[future]
            try: future.result()
            except Exception as e: errors.append({'nid':nid,'error':str(e)})
            done+=1
            if done%100==0 or done==len(leaves):
                state=dict(stage='extraction',done=done,total=len(leaves),errors=len(errors),elapsed=time.time()-start)
                (out/'progress.json').write_text(json.dumps(state)); print(json.dumps(state),flush=True)
    with (out/'extraction_attempts.jsonl').open('a') as f:
        f.write(json.dumps(dict(job=os.environ.get('SLURM_JOB_ID'),elapsed_seconds=time.time()-start,done=done,errors=len(errors)))+'\n')
    if errors:
        (out/'errors.json').write_text(json.dumps(errors,indent=2)); raise RuntimeError(f'{len(errors)} failed extractions')
    extraction_seconds=time.time()-start
    # Extraction success is not an index-complete marker. Fail closed on incomplete graph stages.
    model_load_started=time.time()
    embedder=Embedder(Path(args.models)/'bge-base-en-v1.5')
    reranker=Reranker(Path(args.models)/'bge-reranker-base')
    model_load_seconds=time.time()-model_load_started
    refinement_started=time.time()
    if args.er_scope=='document':
        from document_er import refine_documents
        graph,refiner=refine_documents(llm,embedder,reranker,extractor,leaves,node_doc,out,deadline)
        graph.remove_self_loops()
    else:
        graph=FastGraph(save_path=str(out))
        if (out/'er_vdb').exists():
            import shutil; shutil.rmtree(out/'er_vdb')  # task-owned scratch; extraction cache retained
        refiner=Refiner(llm,embedder,reranker,graph,out)
        # Precompute embeddings in batches, avoiding thousands of tiny GPU forwards in the ER loop.
        names=[]
        for nid in leaves:
            res=extractor.load_tmp_res(nid)
            names.extend(graph.get_node_name_from_entity(e) for e in res['entities'])
        embedder.embed_texts(names)
        for i,nid in enumerate(leaves):
            if time.time()>deadline-90:
                (out/'progress.json').write_text(json.dumps(dict(stage='er_budget_exceeded',done=i,total=len(leaves))))
                raise TimeoutError('advanced ER exceeded build budget; no final index published')
            res=extractor.load_tmp_res(nid)
            refiner.advanced_kg_refiner(res['entities'],res['relations'],nid)
            if i%100==0:
                p=dict(stage='entity_resolution',done=i,total=len(leaves),nodes=len(graph.kg),edges=graph.kg.number_of_edges(),elapsed=time.time()-start)
                (out/'progress.json').write_text(json.dumps(p)); print(json.dumps(p),flush=True)
        refiner.refine_entities(); graph.remove_self_loops()
    refinement_seconds=time.time()-refinement_started
    # Keep actual upstream graph + bidirectional source mappings; no training/test information enters it.
    entities=sorted(graph.kg.nodes)
    vectors=embedder.embed_texts(entities).astype('float32')
    with (out/'bookindex.pkl').open('wb') as f: pickle.dump(dict(tree=tree,graph=graph,node_doc=node_doc,docs=docs,leaves=leaves,entities=entities),f)
    np.save(out/'entities.npy',vectors)
    # Verify every entity/source mapping in both directions before publishing.
    for name,data in graph.kg.nodes(data=True):
        for nid in data['source_ids']:
            assert name in graph.tree2kg[nid], (name,nid,'missing forward GT-Link')
    for nid,names in graph.tree2kg.items():
        for name in names:
            assert name in graph.kg and nid in graph.kg.nodes[name]['source_ids'], (name,nid,'stale GT-Link')
    graph.save_graph()
    raw_calls=[json.loads(line) for line in (out/'calls.jsonl').open()] if (out/'calls.jsonl').exists() else []
    generation_cost=dict(calls=len(raw_calls),truncated_calls=sum(x['finish_reason']=='length' for x in raw_calls),prompt_tokens=sum(x['usage']['prompt_tokens'] for x in raw_calls),completion_tokens=sum(x['usage']['completion_tokens'] for x in raw_calls))
    stats=dict(task=args.task,generation_cost=generation_cost,er_scope=args.er_scope,corpus_sha256=corpus_sha,documents=len(docs),tree_nodes=len(tree.nodes),leaves=len(leaves),
               graph_entities=len(graph.kg),graph_edges=graph.kg.number_of_edges(),gt_links=sum(map(len,graph.tree2kg.values())),
               gradient_searches=refiner.gradient_searches,gradient_decisions=refiner.gradient_decisions,alias_merges=refiner.alias_merges,
               embedding_params=embedder.params,reranker_params=reranker.params,generator='Qwen3.5-4B',
               elapsed_seconds=time.time()-start,extraction_seconds=extraction_seconds,model_load_seconds=model_load_seconds,refinement_seconds=refinement_seconds,seconds_budget=args.seconds,limited_documents=args.limit,
               complete=True,source_revision='5ddf9eb4b4806cddf3c2a98e195f72866b1ab694')
    (out/'index_stats.json').write_text(json.dumps(stats,indent=2)); print(json.dumps(stats),flush=True)
    (out/'COMPLETE').write_text('complete\n')
if __name__=='__main__':
    import logging
    logging.basicConfig(level=logging.ERROR)
    main()
