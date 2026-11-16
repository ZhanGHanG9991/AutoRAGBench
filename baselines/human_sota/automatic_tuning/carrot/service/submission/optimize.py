"""Train CARROT's configuration agent on public train; budget-select on dev.

Only bounded exhaustive label search, native paired dataset/network/joint loss.
Public train labels and development evaluation are never mixed with hidden test.
"""
import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
import httpx
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from runtime import Pipeline, build_index, RERANKERS, PARAMS
from agents.data_enhanced import EmbeddingPairsDatasetEnhanced
from agents.network_enhanced import SiameseNetworkEnhanced, ContrastiveAndRegressionLossEnhanced


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=1))


class Meter:
    def __init__(self):
        self.url = os.environ['CARROT_GATEWAY_URL']
        self.base = self.url + '/v1'
        self.headers = {'X-Admin-Token': Path(os.environ['CARROT_ADMIN_TOKEN_FILE']).read_text().strip()}
        self.client = httpx.Client(timeout=15, trust_env=False)
        self.count = 0

    def evaluate(self, pipeline, row, context, statistics):
        key = row['qid'] + ':' + str(self.count)
        self.count += 1
        self.client.post(self.url + '/admin/open', json={'qid': key}, headers=self.headers).raise_for_status()
        t0 = time.perf_counter()
        error, answer = None, ''
        try:
            answer, usage = pipeline.generate(row['question'], context, self.base)
        except Exception as e:
            error = type(e).__name__ + ':' + str(e)[:160]
        finally:
            r = self.client.post(self.url + '/admin/close', json={'qid': key}, headers=self.headers)
            r.raise_for_status(); gateway = r.json()
        # Include search/embedding latency in the benchmark's 60-second limit.
        latency = time.perf_counter() - t0 + statistics.get('retrieval_seconds', 0) + statistics['mcts_seconds']
        return {'qid': row['qid'], 'answer': answer, 'error': error,
                'over_budget': bool(gateway.get('over_tokens') or gateway.get('over_seconds') or latency > 60),
                'gateway': gateway, 'latency': latency, 'carrot': statistics}


def train_predictor(pipeline, labeled, root):
    df = pd.DataFrame(labeled)
    encoder = LabelEncoder().fit(RERANKERS)
    # RERANKERS is sorted: head assignment is consistent with LabelEncoder.
    assert encoder.classes_.tolist() == RERANKERS
    dataset = EmbeddingPairsDatasetEnhanced(df, 'question_embedding', 'nodes_query_embedding',
                                            'max_reranker', df[['C_param', 'iteration', 'lambda']],
                                            encoder, use_chunk=True)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    net = SiameseNetworkEnhanced(embed_dim=1024, num_classes=6, use_chunk_fusion=True).to(pipeline.device)
    loss_fn = ContrastiveAndRegressionLossEnhanced(margin=1.0)
    optimizer = torch.optim.Adam(net.parameters(), lr=.001)
    losses = []
    for epoch in range(60):
        net.train(); totals = []
        for data in loader:
            q1, c1, q2, c2, label, y1, y2, p1, p2 = data
            q1, c1, q2, c2, label, p1, p2 = [x.float().to(pipeline.device) for x in (q1, c1, q2, c2, label, p1, p2)]
            y1, y2 = y1.to(pipeline.device), y2.to(pipeline.device)
            out1, pred1 = net(q1, c1); out2, pred2 = net(q2, c2)
            loss = loss_fn(out1, out2, label, out1, out2, y1, y2, pred1, pred2, p1, p2)
            if not torch.isfinite(loss):
                raise RuntimeError('non-finite native predictor loss')
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            totals.append(float(loss.detach()))
        losses.append({'epoch': epoch + 1, 'joint_loss': float(np.mean(totals))})
    torch.save(net.state_dict(), root / 'predictor.pt')
    dump(root / 'training_loss.json', losses)
    net.eval(); pipeline.network = net
    print('NATIVE_PREDICTOR_TRAINED', len(df), 'epochs=60', 'final_loss=', losses[-1], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--public', type=Path, required=True)
    parser.add_argument('--train-limit', type=int, default=64)
    parser.add_argument('--dev-limit', type=int, default=100)
    parser.add_argument('--seconds', type=int, default=7200)
    args = parser.parse_args()
    root = args.root; root.mkdir(parents=True, exist_ok=True)
    seed = 20261014
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    sysroot = Path(os.environ['CARROT_RAGENG_ROOT'])
    import sys
    sys.path.insert(0, str(sysroot))
    from tasks import registry
    from tasks.score import score
    from tasks.common import read_jsonl
    module = registry.load(args.task)
    metric = registry.info(args.task)['primary_metric']
    train = list(read_jsonl(args.public / 'train.jsonl'))
    dev = list(read_jsonl(args.public / 'dev.jsonl'))[:args.dev_limit]
    random.Random(seed).shuffle(train); train = train[:args.train_limit]
    assert not {r['qid'] for r in train} & {r['qid'] for r in dev}
    budget_start = time.monotonic()
    deadline = budget_start + args.seconds
    if not (root / 'index/embeddings.npy').exists():
        manifest = build_index(root, args.public / 'corpus')
        dump(root / 'index_manifest.json', manifest)
    else:
        manifest = json.loads((root / 'index_manifest.json').read_text())
    pipeline = Pipeline(root, args.task)
    meter = Meter()
    labels = []
    label_path = root / 'train_trials.jsonl'
    # Resumption only reuses public train computations with identical protocol.
    previous = []
    if (root / 'training_labels.json').exists():
        previous = json.loads((root / 'training_labels.json').read_text())
    done = {x['qid'] for x in previous}
    labels.extend(previous)
    grid = [{'reranker': r, 'C': c, 'iterations': i, 'lambda': lam}
            for r in RERANKERS for c, i, lam in PARAMS]
    for row in train:
        if row['qid'] in done: continue
        if time.monotonic() > deadline:
            raise TimeoutError('build budget exhausted before training labels completed')
        rt = time.perf_counter()
        nodes, qe, ce = pipeline.retrieve(row['question'], row.get('meta', {}))
        retrieval_seconds = time.perf_counter() - rt
        trials, caches = [], {r: {} for r in RERANKERS}
        for cfg in grid:
            ctx, stats = pipeline.search(row['question'], nodes, cfg, 1024, caches[cfg['reranker']])
            stats['retrieval_seconds'] = retrieval_seconds
            rec = meter.evaluate(pipeline, row, ctx, stats)
            value = module.score_one(rec['answer'], row)[metric] if not rec['error'] and not rec['over_budget'] else 0.
            rec['native_score'] = value
            with label_path.open('a') as stream: stream.write(json.dumps(rec, ensure_ascii=False) + '\n')
            trials.append(rec)
        # Tie: least generation tokens, then least MCTS iterations, deterministic
        # catalog order. Ground truth is public TRAIN only.
        best = max(trials, key=lambda r: (r['native_score'], -r['gateway'].get('total_tokens', 0),
                                          -r['carrot']['mcts_iterations']))
        cfg = best['carrot']['configuration']
        labels.append({'qid': row['qid'], 'question_embedding': qe.tolist(),
                       'nodes_query_embedding': ce.tolist(), 'max_reranker': cfg['reranker'],
                       'C_param': cfg['C'], 'iteration': cfg['iterations'], 'lambda': cfg['lambda'],
                       'best_train_score': best['native_score']})
        dump(root / 'training_labels.json', labels)
        print('TRAIN_LABEL', args.task, len(labels), len(train), cfg, best['native_score'], flush=True)
    train_predictor(pipeline, labels, root)
    development = []
    for budget in (512, 1024, 2048):
        preds = []
        path = root / f'dev_b{budget}.jsonl'
        with path.open('w') as stream:
            for row in dev:
                if time.monotonic() > deadline:
                    raise TimeoutError('build budget exhausted before development selection completed')
                rt = time.perf_counter()
                nodes, qe, ce = pipeline.retrieve(row['question'], row.get('meta', {}))
                cfg, raw = pipeline.predict_config(qe, ce)
                retrieval_seconds = time.perf_counter() - rt
                ctx, stats = pipeline.search(row['question'], nodes, cfg, budget)
                stats.update({'retrieval_seconds': retrieval_seconds, 'predictor_raw_params': raw})
                rec = meter.evaluate(pipeline, row, ctx, stats)
                preds.append(rec); stream.write(json.dumps(rec, ensure_ascii=False) + '\n'); stream.flush()
        scores = score(args.task, {x['qid']: x for x in preds}, dev)
        dump(root / f'dev_b{budget}_scores.json', scores)
        info = {'budget': budget, 'dev_primary': scores['primary'], 'dev_n': scores['n'],
                'invalid': scores['n_invalid'], 'mean_context_cl100k': np.mean([r['carrot']['context_cl100k'] for r in preds]),
                'mean_generator_tokens': np.mean([r['gateway'].get('total_tokens', 0) for r in preds]),
                'mean_reranker_pairs': np.mean([r['carrot']['reranker_pairs_evaluated'] for r in preds]),
                'mean_latency': np.mean([r['latency'] for r in preds])}
        development.append(info)
        print('DEV_BUDGET', args.task, json.dumps(info), flush=True)
    best = max(development, key=lambda x: (x['dev_primary'], -x['mean_generator_tokens'], -x['budget']))
    artifact = {'task': args.task, 'budget': best['budget'], 'method': 'CARROT-learned-MCTS',
                'seed': seed, 'train_available': 500, 'train_used': len(labels), 'dev_used': len(dev),
                'train_qids': [r['qid'] for r in train], 'dev_qids': [r['qid'] for r in dev],
                'train_sha256': hashlib.sha256((args.public / 'train.jsonl').read_bytes()).hexdigest(),
                'dev_sha256': hashlib.sha256((args.public / 'dev.jsonl').read_bytes()).hexdigest(),
                'index': manifest, 'model_parameters': pipeline.model_parameters,
                'catalog': RERANKERS, 'label_grid': grid, 'top_k': 20, 'epochs': 60,
                'development_tradeoff': development, 'generator_calls_build': meter.count,
                'build_budget_seconds': args.seconds, 'elapsed_seconds': time.monotonic() - budget_start,
                'hidden_test_used': False, 'budget_tokenizer': 'cl100k_base',
                'ucb_cost_unit': 'native whitespace words', 'generator': 'Qwen3.5-4B',
                'adaptations': ['hard budget enforcement in subclass', 'native AutoRAGBench scoring for label search',
                                'two official rerankers', 'native enhanced trainer network at inference',
                                'QASPER meta.doc_id prefilter', 'fixed metered generator']}
    dump(root / 'artifact.json', artifact)
    print('OPTIMIZATION_COMPLETE', args.task, json.dumps(best), flush=True)


if __name__ == '__main__':
    main()
