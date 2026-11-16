"""Formal-v2 public-train search: released SmartGridSearch.search + all six rerankers.

No inter-configuration score/answer cache. Only scoring/generator/provenance
hooks differ: native MuSiQue F1, fixed metered Qwen, and hard-budget correction.
"""
import argparse
import hashlib
import importlib.util
import json
import logging
import os
import random
import resource
import sys
import time
import types
from pathlib import Path

import faiss
import httpx
import numpy as np
import torch
from FlagEmbedding import BGEM3FlagModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).resolve().parent
C = Path(os.environ.get('CARROT_UPSTREAM', HERE.parent.parent)).resolve()
MODEL_ROOT = Path(os.environ.get('CARROT_MODEL_ROOT', HERE / 'models')).resolve()
sys.path[:0] = [str(C / 'rageng'), str(C)]
from runtime import BudgetMCTS, Pipeline, token_count, checked
from optimize import Meter

# The release omits core/config.py; importing the label-search module imports
# ModelLoader, even though its provider setup is unused. No provider is created.
config = types.ModuleType('core.config'); config.API_KEY_1 = ''; config.API_OPENROUTER = ''
sys.modules.setdefault('core.config', config)
from agents.model_loader import ModelLoader
SPEC = importlib.util.spec_from_file_location('carrot_native_labelsearch', C / 'data/data_preprocess_wikipassage_smart.py')
NATIVE = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(NATIVE)

ORDER = ['jina-reranker-v1-turbo-en', 'jina-reranker-v2-base-multilingual',
         'bge-reranker-v2-m3', 'bge-reranker-large', 'bge-reranker-base',
         'gte-multilingual-reranker-base']
REPOS = dict(zip(ORDER, ['jinaai/jina-reranker-v1-turbo-en', 'jinaai/jina-reranker-v2-base-multilingual',
                       'BAAI/bge-reranker-v2-m3', 'BAAI/bge-reranker-large', 'BAAI/bge-reranker-base',
                       'Alibaba-NLP/gte-multilingual-reranker-base']))
GTE_CODE_REVISION = '40ced75c3017eb27626c9d4ea981bde21a2662f4'
GTE_OFFLINE_DIR = Path(os.environ.get(
    'CARROT_GTE_OFFLINE_DIR',
    Path(os.environ.get('HF_HOME', MODEL_ROOT)) .parent / 'gte-multilingual-reranker-base-offline',
)).resolve()
ACTIVE = None


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''): h.update(block)
    return h.hexdigest()


def dump(path, obj): Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=1))


class ObservedMCTS(BudgetMCTS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.simulations = self.scored_pairs = self.expanded_nodes = 0
        self.scoring_seconds = 0.
        self.started = time.perf_counter()
        ACTIVE.current = self

    def expand(self, parent):
        nodes = super().expand(parent); self.expanded_nodes += len(nodes); return nodes

    def simulate(self, query, nodes):
        t0 = time.perf_counter()
        result = super().simulate(query, nodes)
        self.simulations += 1; self.scored_pairs += len(nodes)
        self.scoring_seconds += time.perf_counter() - t0
        return result


# Source files stay unchanged. The native search calls this instrumented class,
# whose search, UCB and backpropagation still come from the released source.
NATIVE.MCTS = ObservedMCTS


class Probe(NATIVE.SmartGridSearch):
    def __init__(self, logger, pipeline, meter, output, scorer):
        super().__init__(logger, None, {})
        self.pipeline, self.meter, self.output, self.scorer = pipeline, meter, output, scorer
        self.row = self.current = None
        self.phase = 'coarse'; self.trials = []; self.trial_offset = 0; self.model_names = {}

    def get_ground_truth_answer(self, *args, **kwargs):
        # Public benchmark gold replaces the native generated reference.
        return self.row['answers'][0]

    def evaluate_config(self, question, context, ground_truth):
        m = self.current
        stats = {'algorithm': 'upstream.core.mcts.MCTS.search',
                 'configuration': {'reranker': self.model_names[id(m.model)], 'C': m.C,
                                   'iterations': m.total_trials, 'lambda': m.lambda_},
                 'budget_cl100k': m.budget, 'context_cl100k': token_count(context),
                 'context_words_ucb_cost': len(context.split()), 'mcts_seconds': time.perf_counter()-m.started,
                 'retrieval_seconds': self.retrieval_seconds,
                 'simulations': m.simulations, 'expanded_nodes': m.expanded_nodes,
                 'reranker_pairs_evaluated': m.scored_pairs, 'reranker_scoring_seconds': m.scoring_seconds,
                 'phase': self.phase, 'torch_allocated_mib': torch.cuda.memory_allocated()/2**20,
                 'torch_reserved_mib': torch.cuda.memory_reserved()/2**20,
                 'torch_peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
                 'torch_peak_reserved_mib': torch.cuda.max_memory_reserved()/2**20}
        assert stats['context_cl100k'] <= m.budget
        # The local gateway occasionally loses a keep-alive connection to the
        # colocated vLLM process during this very long search.  Retry only
        # transport/HTTP-500 failures; semantic/model failures still fail
        # closed.  These are fresh generator calls, never cached answers.
        retry_errors = []
        retry_gateway = {'calls': 0, 'prompt_tokens': 0, 'completion_tokens': 0}
        for attempt in range(3):
            try:
                rec = self.meter.evaluate(self.pipeline, self.row, context, stats)
            except httpx.TransportError as exc:
                error = f'{type(exc).__name__}: {exc}'
                if attempt == 2:
                    raise
                retry_errors.append(error)
                print('LOCAL_GENERATOR_RETRY', json.dumps({
                    'qid': self.row['qid'], 'attempt': attempt + 1, 'error': error,
                }), flush=True)
                time.sleep(3 * (attempt + 1))
                continue
            error = str(rec.get('error') or '')
            transient = any(marker in error for marker in (
                '500 Internal Server Error', 'httpcore.ReadError',
                'RemoteProtocolError', 'ConnectError', 'ConnectionResetError'))
            if not error or not transient or attempt == 2:
                break
            retry_errors.append(error)
            gateway = rec.get('gateway') or {}
            for key in retry_gateway:
                retry_gateway[key] += int(gateway.get(key) or 0)
            print('LOCAL_GENERATOR_RETRY', json.dumps({
                'qid': self.row['qid'], 'attempt': attempt + 1, 'error': error,
            }), flush=True)
            time.sleep(3 * (attempt + 1))
        if retry_errors:
            gateway = rec.setdefault('gateway', {})
            for key, value in retry_gateway.items():
                gateway[key] = int(gateway.get(key) or 0) + value
            rec['infrastructure_retries'] = len(retry_errors)
            rec['infrastructure_retry_errors'] = retry_errors
        score = self.scorer(rec['answer'], self.row) if not rec['error'] and not rec['over_budget'] else 0.
        rec.update({'native_f1': score, 'trial_index': self.trial_offset + len(self.trials) + 1,
                    'context_sha256': hashlib.sha256(context.encode()).hexdigest()})
        self.trials.append(rec)
        with (self.output/'trials.jsonl').open('a') as stream:
            stream.write(json.dumps(rec, ensure_ascii=False)+'\n')
        if rec['error']: raise RuntimeError('generator evaluation failed: '+rec['error'])
        return score, rec['answer']


class PhaseHandler(logging.Handler):
    def emit(self, record):
        msg = record.getMessage()
        if msg.startswith('Phase 1:'): ACTIVE.phase = 'coarse'
        elif msg.startswith('  Refining '): ACTIVE.phase = 'quartile'
        elif msg.startswith('Phase 2:'): ACTIVE.phase = 'fine'
        elif msg.startswith('Best overall:'): ACTIVE.phase = 'reranker_comparison'


def load_native_bank(models):
    # Invoke the exact released loader method with only repository names mapped
    # to pinned offline snapshots. It never calls _setup_llm/_setup_embedding.
    paths = {repo: str(models/name) for name, repo in REPOS.items()}
    gte_repo = REPOS['gte-multilingual-reranker-base']
    required_gte_files = ('config.json', 'configuration.py', 'modeling.py', 'model.safetensors')
    missing_gte_files = [name for name in required_gte_files if not (GTE_OFFLINE_DIR / name).exists()]
    if missing_gte_files:
        raise FileNotFoundError(
            f'incomplete offline GTE package {GTE_OFFLINE_DIR}: {missing_gte_files}'
        )
    paths[gte_repo] = str(GTE_OFFLINE_DIR)
    original_model = AutoModelForSequenceClassification.from_pretrained
    original_tokenizer = AutoTokenizer.from_pretrained
    def model_local(name, *a, **k):
        repo = str(name)
        if repo == gte_repo:
            # The campaign-local package contains the exact new-impl revision
            # beside the unchanged model weights and uses local auto_map
            # entries, so Transformers never resolves remote code at runtime.
            k.pop('code_revision', None)
            k.setdefault('local_files_only', True)
        return original_model(paths.get(repo, name), *a, **k)
    def token_local(name, *a, **k): return original_tokenizer(paths.get(str(name), name), *a, **k)
    AutoModelForSequenceClassification.from_pretrained = model_local
    AutoTokenizer.from_pretrained = token_local
    loader = ModelLoader.__new__(ModelLoader)
    try: loader._setup_rerankers()
    finally:
        AutoModelForSequenceClassification.from_pretrained = original_model
        AutoTokenizer.from_pretrained = original_tokenizer
    models_tuple = [loader.jina_model_v1, loader.jina_model_v2, loader.flag_reranker_v2,
                    loader.flag_reranker_large, loader.flag_reranker_base, loader.gte_model]
    rerankers = {name: (model, loader.gte_tokenizer if name.startswith('gte-') else None)
                 for name, model in zip(ORDER, models_tuple)}
    counts = {name: checked(model.model if name.startswith('bge-') else model)
              for name, (model, _) in rerankers.items()}
    return loader, rerankers, counts


def train_small_predictor(labeled, output):
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    from torch.utils.data import DataLoader
    from agents.data_enhanced import EmbeddingPairsDatasetEnhanced
    from agents.network_enhanced import SiameseNetworkEnhanced, ContrastiveAndRegressionLossEnhanced
    t0 = time.perf_counter()
    df = pd.DataFrame(labeled)
    encoder = LabelEncoder().fit(ORDER)
    ds = EmbeddingPairsDatasetEnhanced(df, 'question_embedding', 'nodes_query_embedding', 'max_reranker',
                                      df[['C_param', 'iteration', 'lambda']], encoder, use_chunk=True)
    batches = DataLoader(ds, batch_size=32, shuffle=True)
    net = SiameseNetworkEnhanced(embed_dim=1024, num_classes=6, use_chunk_fusion=True).cuda()
    loss_fn = ContrastiveAndRegressionLossEnhanced(margin=1.)
    optimizer = torch.optim.Adam(net.parameters(), lr=.001)
    losses = []
    for epoch in range(60):
        net.train()
        for data in batches:
            q1,c1,q2,c2,label,y1,y2,p1,p2 = data
            q1,c1,q2,c2,label,p1,p2 = [x.float().cuda() for x in (q1,c1,q2,c2,label,p1,p2)]
            out1,pred1 = net(q1,c1); out2,pred2 = net(q2,c2)
            loss = loss_fn(out1,out2,label,out1,out2,y1.cuda(),y2.cuda(),pred1,pred2,p1,p2)
            assert torch.isfinite(loss)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append({'epoch':epoch+1,'joint_loss':float(loss.detach())})
    torch.save(net.state_dict(), output/'predictor.pt')
    dump(output/'training_loss.json',losses)
    return {'epochs':60,'samples':len(labeled),'seconds':time.perf_counter()-t0,
            'label_encoder_classes':encoder.classes_.tolist(),'final_loss':losses[-1]['joint_loss']}


def main():
    global ACTIVE
    parser = argparse.ArgumentParser(); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--public',type=Path,required=True); parser.add_argument('--index',type=Path,required=True)
    parser.add_argument('--train',type=Path,default=None)
    parser.add_argument('--task',required=True); parser.add_argument('--count',type=int,default=100)
    parser.add_argument('--top-k',type=int,default=60); parser.add_argument('--seed',type=int,default=20261022)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    if not args.resume:
        assert not (args.output/'trials.jsonl').exists(), 'fresh output directory required'
    seed=args.seed;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(8); assert torch.cuda.device_count()==1
    root=Path(os.environ['CARROT_RAGENG_ROOT']);sys.path.insert(0,str(root))
    from tasks import registry
    from core.mcts import MCTS
    assert Probe.search is NATIVE.SmartGridSearch.search
    assert ObservedMCTS.search is MCTS.search
    assert ObservedMCTS.backpropagate is MCTS.backpropagate
    task_module=registry.load(args.task); primary_metric=registry.info(args.task)['primary_metric']
    scorer=lambda prediction,row: task_module.score_one(prediction,row)[primary_metric]
    train_path=args.train or (args.public/'train.jsonl')
    rows=[json.loads(l) for l in train_path.open() if l.strip()]
    random.Random(seed).shuffle(rows);rows=rows[:args.count];assert len(rows)==args.count
    # Corpus index is immutable and reused, so timing reports its exclusion.
    pipeline=Pipeline.__new__(Pipeline);pipeline.root=HERE;pipeline.task=args.task;pipeline.device='cuda'
    started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    t0=time.perf_counter()
    pipeline.embedder=BGEM3FlagModel(str(MODEL_ROOT/'bge-m3'),use_fp16=True,device='cuda')
    counts={'bge-m3':checked(pipeline.embedder.model)}
    pipeline.chunks=json.loads((args.index/'chunks.json').read_text())
    pipeline.embeddings=np.load(args.index/'embeddings.npy',mmap_mode='r')
    pipeline.index=faiss.IndexFlatIP(pipeline.embeddings.shape[1]);pipeline.index.add(np.asarray(pipeline.embeddings,dtype='float32'))
    pipeline.by_doc={}
    for chunk_id, chunk in enumerate(pipeline.chunks):
        pipeline.by_doc.setdefault(str(chunk['doc_id']), []).append(chunk_id)
    embedding_load_seconds=time.perf_counter()-t0
    t0=time.perf_counter();loader,rerankers,reranker_counts=load_native_bank(MODEL_ROOT)
    load_seconds=time.perf_counter()-t0;counts.update(reranker_counts)
    print('ALL_SIX_NATIVE_RERANKERS_LOADED',json.dumps(counts),'seconds',round(load_seconds,2),flush=True)
    meter=Meter();logger=logging.getLogger('carrot-profile-native');logger.setLevel(logging.INFO)
    filehandler=logging.FileHandler(args.output/'native_search.log');logger.addHandler(filehandler)
    ACTIVE=Probe(logger,pipeline,meter,args.output,scorer);logger.addHandler(PhaseHandler())
    ACTIVE.model_names={id(m):name for name,(m,t) in rerankers.items()}
    provenance={'seed':seed,'task':args.task,'train_used':args.count,'train_available':len([l for l in train_path.open() if l.strip()]),'dev_used':0,'test_used':0,
                'train_qids':[r['qid'] for r in rows],'train_sha256':digest(train_path),
                'corpus_sha256':digest(args.public/'corpus/corpus.jsonl'),
                'index_embeddings_sha256':digest(args.index/'embeddings.npy'),
                'native_label_script_sha256':digest(C/'data/data_preprocess_wikipassage_smart.py'),
                'native_mcts_sha256':digest(C/'core/mcts.py'),'top_k':args.top_k,'context_budget':1024,
                'metric':f'{args.task} {primary_metric} on public train gold','generator':'Qwen3.5-4B',
                'score_cache':False,'answer_cache':False,'rerankers_order':ORDER,'model_parameters':counts,
                'native_search_method':'SmartGridSearch.search (unchanged)',
                'native_model_loading':'ModelLoader._setup_rerankers (offline path mapping only)',
                'hard_budget_adapter':'previously validated BudgetMCTS feasibility correction',
                'native_reranker_token_lengths':'released defaults; GTE tokenizer branch max_length=8192',
                'embedding_model_load_seconds':embedding_load_seconds,'six_reranker_load_seconds':load_seconds,
                'chunk_embedding':'native batch-10 re-encoding and mean over all top-60 chunks'}
    dump(args.output/'provenance.json',provenance)
    summaries=[];labeled=[];existing_trials=0
    if args.resume:
        summaries=json.loads((args.output/'query_summaries.json').read_text())
        labeled=json.loads((args.output/'training_labels.json').read_text())
        assert len(summaries)==len(labeled)<len(rows)
        assert [s['qid'] for s in summaries] == [r['qid'] for r in rows[:len(summaries)]]
        assert [x['qid'] for x in labeled] == [r['qid'] for r in rows[:len(labeled)]]
        with (args.output/'trials.jsonl').open() as stream:
            existing_trials=sum(1 for line in stream if line.strip())
        assert existing_trials == sum(int(s['config_evaluations']) for s in summaries)
        ACTIVE.trial_offset=existing_trials
        print('RESUME_NATIVE_PROFILE',json.dumps({
            'completed_queries':len(summaries),'remaining_queries':len(rows)-len(summaries),
            'existing_trials':existing_trials,
        }),flush=True)
    label_started=time.perf_counter()
    for row in rows[len(summaries):]:
        t0=time.perf_counter();rt=time.perf_counter();nodes,q,c=pipeline.retrieve(row['question'],row.get('meta',{}),args.top_k)
        # Match the released preprocessing: re-encode every retrieved chunk in
        # batches of ten and mean-pool all sixty, rather than top-ten pooling.
        chunk_vectors=[]
        for i in range(0,len(nodes),10):
            vectors=NATIVE.get_dense_embeddings([n.node.text for n in nodes[i:i+10]],pipeline.embedder)
            chunk_vectors.append(np.asarray(vectors).reshape(-1,1024))
        c=np.vstack(chunk_vectors).mean(axis=0)
        ACTIVE.retrieval_seconds=time.perf_counter()-rt;ACTIVE.row=row;before=len(ACTIVE.trials)
        result=ACTIVE.search(row['question'],nodes,rerankers,'','',budget=1024)
        trials=ACTIVE.trials[before:];expected=len(result['all_results'])+6
        assert len(trials)==expected and expected in (197,222), (len(trials),expected)
        cfg=result['best_config'];best_name=cfg['reranker']
        labeled.append({'qid':row['qid'],'question_embedding':q.tolist(),'nodes_query_embedding':c.tolist(),
                        'max_reranker':best_name,'C_param':cfg['C'],'iteration':cfg['iterations'],'lambda':cfg['lambda']})
        summary={'qid':row['qid'],'query_index':len(summaries)+1,'seconds':time.perf_counter()-t0,
                 'config_evaluations':len(trials),'generator_calls':sum(r['gateway']['calls'] for r in trials),
                 'prompt_tokens':sum(r['gateway']['prompt_tokens'] for r in trials),
                 'completion_tokens':sum(r['gateway']['completion_tokens'] for r in trials),
                 'reranker_pairs':sum(r['carrot']['reranker_pairs_evaluated'] for r in trials),
                 'mcts_seconds':sum(r['carrot']['mcts_seconds'] for r in trials),
                 'reranker_scoring_seconds':sum(r['carrot']['reranker_scoring_seconds'] for r in trials),
                 'over_budget_trials':sum(r['over_budget'] for r in trials),'errors':sum(bool(r['error']) for r in trials),
                 'best_native_f1':cfg['score'],'best_configuration':{k:cfg[k] for k in ('reranker','C','lambda','iterations')},
                 'final_reranker_scores':result['reranker_scores']}
        summaries.append(summary);dump(args.output/'query_summaries.json',summaries);dump(args.output/'training_labels.json',labeled)
        print('QUERY_PROFILE',json.dumps(summary),flush=True)
    label_seconds=time.perf_counter()-label_started
    training=train_small_predictor(labeled,args.output)
    total={'status':'completed','task':args.task,'queries':args.count,'config_evaluations':existing_trials+len(ACTIVE.trials),
           'labeling_seconds':label_seconds,'predictor_training':training,'process_seconds':time.perf_counter()-started,
           'generator_calls':sum(r['generator_calls'] for r in summaries),'prompt_tokens':sum(r['prompt_tokens'] for r in summaries),
           'completion_tokens':sum(r['completion_tokens'] for r in summaries),
           'mcts_seconds':sum(r['mcts_seconds'] for r in summaries),'reranker_pairs':sum(r['reranker_pairs'] for r in summaries),
           'over_budget_trials':sum(r['over_budget_trials'] for r in summaries),'generator_errors':sum(r['errors'] for r in summaries),
           'torch_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
           'torch_peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,
           'python_peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
           'hidden_test_used':False,'external_api_calls':0,'index_rebuilt':False}
    dump(args.output/'summary.json',total);print('FULL_NATIVE_PROFILE_COMPLETE',json.dumps(total),flush=True)


if __name__=='__main__':main()
