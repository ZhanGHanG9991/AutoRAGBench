"""Keep the paper's clean, within-document ER scope; parallelize independent documents."""
from runtime import *
from concurrent.futures import ThreadPoolExecutor, as_completed

def refine_documents(llm,embedder,reranker,extractor,leaves,node_doc,out,deadline,workers=32):
    # Precompute the name embeddings once; per-document graph state remains independent.
    by_doc=defaultdict(list); names=[]
    for nid in leaves:
        by_doc[node_doc[nid]].append(nid)
        for e in extractor.load_tmp_res(nid)['entities']:
            names.append(f'Name: {e.entity_name}\nType: {e.entity_type}')
    embedder.embed_texts(names)
    merged=FastGraph(save_path=str(out)); counts=defaultdict(int)
    def work(item):
        did,nids=item
        if time.time()>deadline-90: raise TimeoutError('document ER budget exceeded')
        graph=FastGraph(); refiner=Refiner(llm,embedder,reranker,graph,out,memory=True)
        for nid in nids:
            res=extractor.load_tmp_res(nid)
            refiner.advanced_kg_refiner(res['entities'],res['relations'],nid)
        refiner.refine_entities()
        return graph,refiner.gradient_searches,refiner.gradient_decisions,refiner.alias_merges
    started=time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(work,item) for item in by_doc.items()]
        for i,f in enumerate(as_completed(futures)):
            graph,searches,decisions,aliases=f.result()
            union_graph(merged,graph)
            counts['gradient_searches']+=searches; counts['gradient_decisions']+=decisions; counts['alias_merges']+=aliases
            if i%100==0 or i+1==len(by_doc):
                p=dict(stage='document_er',documents_done=i+1,documents_total=len(by_doc),nodes=len(merged.kg),edges=merged.kg.number_of_edges(),elapsed=time.time()-started)
                (Path(out)/'progress.json').write_text(json.dumps(p)); print(json.dumps(p),flush=True)
    return merged,SimpleNamespace(**counts)
