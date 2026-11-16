"""AutoRAGBench transport/model adapters. Uses upstream BookRAG operators with the recorded compatibility patches."""
from __future__ import annotations
import os
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
from pathlib import Path
import sys, json, time, threading
sys.path.insert(0,str(Path(__file__).resolve().parent / "vendor"))
from types import SimpleNamespace
from collections import defaultdict
import numpy as np
import httpx
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from Core.Index.Tree import DocumentTree, TreeNode, NodeType
from Core.Index.Graph import Graph, Entity, Relationship, SetEncoder
from Core.pipelines.kg_extractor import KGExtractor
from Core.pipelines.kg_refiner import KGRefiner
from Core.provider.vdb import VectorStore
from Core.configs.graph_config import GraphConfig
from Core.rag.gbc_plan import TaskPlanner, PlanResult
from Core.rag.gbc_retrieval import Retriever
from Core.rag.gbc_utils import filter_tree_nodes

class FixedLLM:
    """Only evaluator-provided generator endpoint; no external providers or automatic eval retries."""
    def __init__(self, base, log=None, output_tokens=3000, build_mode=False):
        self.client = httpx.Client(base_url=base.rstrip('/'), timeout=120, trust_env=False)
        self.config = SimpleNamespace(max_tokens=8000)
        self.output_tokens = output_tokens
        self.build_mode=build_mode
        self.log = Path(log) if log else None
        self.lock = threading.Lock()
        self.calls = 0
    def request(self, prompt, schema=None, max_tokens=None):
        if not isinstance(prompt, str):
            messages = [m.to_dict() for m in prompt.storage]
        else:
            messages = [{'role':'user', 'content':prompt}]
        data = dict(model='generator', messages=messages, temperature=0,
                    max_tokens=max_tokens or self.output_tokens, seed=20261022,
                    chat_template_kwargs={'enable_thinking':False})
        if self.build_mode:
            data['repetition_penalty']=1.08
        if schema:
            data['structured_outputs'] = {'json':schema.model_json_schema()}
        started = time.time()
        r = self.client.post('/chat/completions', json=data)
        r.raise_for_status()
        obj = r.json()
        text = obj['choices'][0]['message']['content'] or ''
        rec = dict(time=time.time(), elapsed=time.time()-started, usage=obj.get('usage'),
                   finish_reason=obj['choices'][0].get('finish_reason'), prompt=messages, output=text)
        with self.lock:
            self.calls += 1
            if self.log:
                with self.log.open('a') as f: f.write(json.dumps(rec,ensure_ascii=False)+'\n')
        if rec['finish_reason'] == 'length' and not self.build_mode:
            raise RuntimeError('Generator output truncated during query execution')
        return text
    def get_completion(self, prompt, json_response=False):
        if self.build_mode and isinstance(prompt,str) and '<|COMPLETE|>' in prompt and '"entity"' in prompt:
            prompt += '\nOutput each entity only ONCE, combining its attributes into one concise description of at most 30 words. Output each relationship only ONCE. Finish with the required completion delimiter. Do not repeat previously listed records.'
        return self.request(prompt)
    def get_json_completion(self, prompt, schema, **kwargs):
        return schema.model_validate_json(self.request(prompt, schema=schema))

class Embedder:
    MM_EMBEDDER = False
    def __init__(self, path):
        self.model = SentenceTransformer(str(path), device='cuda')
        self.model.max_seq_length = 512
        self.params = sum(p.numel() for p in self.model.parameters())
        assert self.params <= 10**9
        self.cache = {}
        self.encode_lock=threading.RLock()
    def embed_texts(self,texts):
        with self.encode_lock:
            return self._embed_texts(texts)
    def _embed_texts(self, texts):
        texts = list(texts)
        missing = list(dict.fromkeys(t for t in texts if t not in self.cache))
        if missing:
            a = self.model.encode(missing,batch_size=128,normalize_embeddings=True,convert_to_numpy=True,
                                  show_progress_bar=False)
            for t,v in zip(missing,a): self.cache[t]=v
        return np.stack([self.cache[t] for t in texts]) if texts else np.empty((0,768),dtype=np.float32)
    def compute_texts_sim(self,a,b):
        x=self.embed_texts([a,b]); return float(x[0] @ x[1])
    def close(self): pass

class Reranker:
    max_length = 512
    def __init__(self,path):
        self.model = CrossEncoder(str(path),device='cuda',max_length=512)
        self.params = sum(p.numel() for p in self.model.model.parameters())
        self.calls=0
        self.predict_lock=threading.Lock()
        self.local=threading.local()
        assert self.params <= 10**9
    def rerank(self,query,documents,instruction=None):
        if not documents: return []
        self.calls+=1
        self.local.calls=getattr(self.local,"calls",0)+1
        with self.predict_lock:
            return self.model.predict([(query,x) for x in documents],batch_size=64,
                                  show_progress_bar=False, activation_fn=torch.nn.Sigmoid()).tolist()
    def clean_cache(self): pass
    def close(self): pass

class Refiner(KGRefiner):
    """Inject compliant <=1B encoders into the original advanced ER implementation."""
    def __init__(self,llm,embedder,reranker,graph,path,memory=False):
        self.llm=llm; self.graph_index=graph; self.graph_config=GraphConfig()
        self.similar_search_topk=10; self.similar_search_distance_threshold=.2
        self.similar_search_mink=1; self.similar_search_g=.6
        self.embedder=embedder; self.reranker=reranker
        self.vdb_path=str(Path(path)/'er_vdb')
        self.vdb=MemoryVectorStore(embedder) if memory else VectorStore(embedder,self.vdb_path,'kg_collection')
        self.entity_to_vdb_id=defaultdict(str); self.entity_alias_map=defaultdict(str)
        self.gradient_searches=0; self.gradient_decisions=0; self.alias_merges=0
    def search_similar_entities(self,entity):
        self.gradient_searches+=1
        before=getattr(self.reranker.local,"calls",0)
        result=super().search_similar_entities(entity)
        self.gradient_decisions+=int(getattr(self.reranker.local,"calls",0)>before)
        return result
    def entity_merge(self,old_entity,new_entity,merged_to_old_entity=False):
        if (old_entity.entity_name,old_entity.entity_type)!=(new_entity.entity_name,new_entity.entity_type):
            self.alias_merges+=1
        return super().entity_merge(old_entity,new_entity,merged_to_old_entity)

class FastGraph(Graph):
    def update_entity(self,old_entity_name,old_entity_type,new_entity):
        # Upstream updates only new_source_ids, leaving stale links after a renamed/third-entity merge.
        old_name=self.get_node_name_from_str(old_entity_name,old_entity_type)
        new_name=self.get_node_name_from_entity(new_entity)
        sources=set(new_entity.source_ids) | set(self.kg.nodes[old_name]['source_ids'])
        if new_name in self.kg: sources |= set(self.kg.nodes[new_name]['source_ids'])
        repaired=new_entity.model_copy(update={'source_ids':sources})
        super().update_entity(old_entity_name,old_entity_type,repaired)
        for nid in sources:
            if old_name!=new_name: self.tree2kg[nid].discard(old_name)
            self.tree2kg[nid].add(new_name)
    def add_and_link(self,tree_node_id,entities):
        # Fix upstream's <=10-entity bootstrap path losing reverse GT-links on repeated names.
        if isinstance(entities,Entity): entities=[entities]
        previous={self.get_node_name_from_entity(e):set(self.kg.nodes[self.get_node_name_from_entity(e)]['source_ids'])
                  for e in entities if self.get_node_name_from_entity(e) in self.kg}
        super().add_and_link(tree_node_id,entities)
        for e in entities:
            name=self.get_node_name_from_entity(e)
            self.kg.nodes[name]['source_ids']=set(self.kg.nodes[name]['source_ids']) | previous.get(name,set()) | {tree_node_id}
    def get_all_nodes(self):
        # A NetworkX NodeView preserves membership semantics without allocating a full set per entity.
        return self.kg.nodes


def build_tree(corpus,out,task,limit=0):
    import re
    tree=DocumentTree({'content':'Corpus','file_name':task},SimpleNamespace(save_path=str(out)))
    node_doc={}; docs={}; leaf_ids=[]
    def add(parent,content,kind):
        n=TreeNode({'content':content,'page_idx':0}); n.type=kind
        tree.add_node(n); parent.add_child(n); return n
    for i,line in enumerate(Path(corpus).open()):
        if limit and i>=limit: break
        d=json.loads(line); did=d['doc_id']; docs[did]=d
        doc=add(tree.root_node,d.get('title') or did,NodeType.TITLE)
        doc.outline_node=True; node_doc[doc.index_id]=did
        parent=doc
        # Only explicit headings in the provided corpus are interpreted as sections.
        if task=='qasper':
            blocks=re.split(r'(?m)^## (.+)\n',d['text'])
            sections=[('Abstract',blocks[0].removeprefix('Abstract\n'))]
            sections.extend(zip(blocks[1::2],blocks[2::2]))
        else: sections=[(None,d['text'])]
        for title,body in sections:
            if title:
                parent=add(doc,title,NodeType.TITLE); parent.outline_node=True; node_doc[parent.index_id]=did
            else: parent=doc
            # Preserve source text and order; no synthetic knowledge/chapters.
            for para in body.split('\n'):
                words=para.split()
                for j in range(0,len(words),200):
                    n=add(parent,' '.join(words[j:j+200]),NodeType.TEXT)
                    n.summary=d.get('title','')
                    node_doc[n.index_id]=did; leaf_ids.append(n.index_id)
    tree.max_depth=max(n.depth for n in tree.nodes)
    return tree,node_doc,docs,leaf_ids


class AuditedRetriever(Retriever):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.graph_calls=self.text_calls=self.skyline_calls=0
        import Core.rag.gbc_retrieval as mod
        original=mod.calculate_skyline
        def audited(scores):
            self.skyline_calls+=1
            return original(scores)
        mod.calculate_skyline=audited
    def graph_reranker(self,*args,**kwargs):
        self.graph_calls+=1
        return super().graph_reranker(*args,**kwargs)
    def text_reranker(self,*args,**kwargs):
        self.text_calls+=1
        return super().text_reranker(*args,**kwargs)

def init_retriever(embedder,reranker):
    return AuditedRetriever(variant='standard',reranker=reranker,embedder=embedder,
                     alpha=.5,topk_ent=5,x_percentile=.85,topk=10)

class MemoryVectorStore:
    """Exact cosine store for one document; same interface/distances as Chroma's cosine store."""
    def __init__(self,embedder):
        self.embedding_model=embedder; self.items={}; self.collection=self
    def count(self): return len(self.items)
    def add_texts(self,texts,metadatas):
        import uuid
        vectors=self.embedding_model.embed_texts(texts)
        ids=[str(uuid.uuid4()) for _ in texts]
        self.items.update({i:(t,m,v) for i,t,m,v in zip(ids,texts,metadatas,vectors)})
        return ids
    def delete_text_by_ids(self,ids):
        for i in ids: self.items.pop(i,None)
    def search(self,query_text,top_k=10):
        if not self.items: return []
        ids=list(self.items); vals=list(self.items.values())
        vectors=np.stack([x[2] for x in vals]); q=self.embedding_model.embed_texts([query_text])[0]
        distances=1-vectors@q
        return [dict(id=ids[i],distance=float(distances[i]),content=vals[i][0],metadata=vals[i][1])
                for i in np.argsort(distances,kind='stable')[:top_k]]

def union_graph(target,source):
    """Union document KGs by identical typed entity names, retaining all source GT-links."""
    for name,attributes in source.kg.nodes(data=True):
        if name not in target.kg:
            target.kg.add_node(name,**dict(attributes))
        else:
            existing=target.kg.nodes[name]
            existing['source_ids']=set(existing['source_ids']) | set(attributes['source_ids'])
            if len(attributes.get('description',''))>len(existing.get('description','')):
                existing['description']=attributes['description']
        for nid in attributes['source_ids']: target.tree2kg[nid].add(name)
    for u,v,attributes in source.kg.edges(data=True):
        if target.kg.has_edge(u,v):
            old=target.kg.edges[u,v]
            old['source_ids']=set(old.get('source_ids',set())) | set(attributes.get('source_ids',set()))
            old['weight']=max(old.get('weight',1),attributes.get('weight',1))
        else: target.kg.add_edge(u,v,**dict(attributes))
