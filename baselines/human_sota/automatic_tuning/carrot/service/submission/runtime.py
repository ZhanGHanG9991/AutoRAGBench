"""CARROT adapter. Native MCTS.search/UCB/backprop and native learned network.

The budget subclass repairs the official tokenizer=None budget bypass. No query
expansion, additional retrieval rounds, answer voting, or external generators.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import faiss
import httpx
import numpy as np
import tiktoken
import torch
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from transformers import AutoTokenizer

BASE = Path(__file__).resolve().parent
UPSTREAM = BASE.parent / 'upstream' if (BASE.parent / 'upstream').exists() else BASE.parent
sys.path.insert(0, str(UPSTREAM))
from core.mcts import MCTS
from agents.network_enhanced import SiameseNetworkEnhanced

ENC = tiktoken.get_encoding('cl100k_base')
RERANKERS = ['bge-reranker-base', 'bge-reranker-v2-m3']
PARAMS = [(2.0, 10, 0.0), (3.0, 20, 0.6), (4.0, 30, 1.0)]


def token_count(text):
    return len(ENC.encode(text, disallowed_special=()))


class BudgetMCTS(MCTS):
    """Paper-required feasible expansion/selection; other methods are upstream."""
    def _filter_by_budget(self, nodes):
        return [n for n in nodes if token_count(n.concat_text) <= self.budget]

    def expand(self, parent):
        children = super().expand(parent)
        valid = self._filter_by_budget(children)
        parent.children[:] = self._filter_by_budget(parent.children)
        return valid

    def _is_fully_expanded(self, node):
        feasible = sum(token_count(node.concat_text + ' ' + c.node.text) <= self.budget
                       for c in self.nodes_query if c not in node.state)
        return len(node.children) >= feasible

    def select_best_node(self, nodes, parent_visits):
        valid = self._filter_by_budget(nodes)
        if not valid:
            raise RuntimeError('no budget-feasible child')
        return super().select_best_node(valid, parent_visits)

    def find_best_node_recursive(self, root):
        # With feasible expansion every node passed to native global selection
        # is feasible; native all-depth reward/visits/layer tie-breaking remains.
        best = super().find_best_node_recursive(root)
        assert token_count(best.concat_text) <= self.budget
        return best


class CountingReranker:
    def __init__(self, model, cache=None):
        self.model = model
        self.cache = {} if cache is None else cache
        self.requested = self.evaluated = self.batches = 0

    def compute_score(self, pairs):
        keys = [tuple(p) for p in pairs]
        missing = list(dict.fromkeys(k for k in keys if k not in self.cache))
        self.requested += len(keys)
        if missing:
            scores = self.model.compute_score([list(p) for p in missing], batch_size=16,
                                              max_length=self.model.max_length)
            if isinstance(scores, (float, int)):
                scores = [scores]
            self.cache.update(zip(missing, [float(s) for s in scores]))
            self.evaluated += len(missing)
            self.batches += (len(missing) + 15) // 16
        return [self.cache[k] for k in keys]


def checked(model):
    count = sum(p.numel() for p in model.parameters())
    if count > 1_000_000_000:
        raise ValueError(f'retrieval model exceeds 1B: {count}')
    return count


class Pipeline:
    def __init__(self, root, task):
        self.root, self.task = Path(root), task
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.set_num_threads(min(8, int(os.environ.get('OMP_NUM_THREADS', '8'))))
        self.embedder = BGEM3FlagModel(str(self.root / 'models/bge-m3'),
                                    use_fp16=self.device == 'cuda', device=self.device)
        self.model_parameters = {'bge-m3': checked(self.embedder.model)}
        self.rerankers = {}
        for name in RERANKERS:
            model = FlagReranker(str(self.root / 'models' / name),
                                 use_fp16=self.device == 'cuda', device=self.device)
            model.max_length = 512 if name == 'bge-reranker-base' else 8192
            self.model_parameters[name] = checked(model.model)
            self.rerankers[name] = model
        self.chunks = json.loads((self.root / 'index/chunks.json').read_text())
        self.embeddings = np.load(self.root / 'index/embeddings.npy', mmap_mode='r')
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(np.asarray(self.embeddings, dtype='float32'))
        self.by_doc = {}
        for i, c in enumerate(self.chunks):
            self.by_doc.setdefault(c['doc_id'], []).append(i)
        self.network = None
        if (self.root / 'predictor.pt').exists():
            self.network = SiameseNetworkEnhanced(embed_dim=1024, num_classes=6,
                                                  use_chunk_fusion=True).to(self.device)
            self.network.load_state_dict(torch.load(self.root / 'predictor.pt',
                                                   map_location=self.device, weights_only=True))
            self.network.eval()

    def encode(self, texts):
        x = self.embedder.encode(texts, batch_size=32, max_length=8192,
                                 return_dense=True, return_sparse=False, return_colbert_vecs=False)
        return np.asarray(x['dense_vecs'], dtype='float32')

    def retrieve(self, question, meta, top_k=20):
        q = self.encode([question])[0]
        if self.task == 'qasper':
            doc = (meta or {}).get('doc_id')
            if not doc or doc not in self.by_doc:
                raise ValueError('QASPER requires a valid meta.doc_id')
            ids = np.asarray(self.by_doc[doc])
            chosen = ids[np.argsort(-(self.embeddings[ids] @ q))[:top_k]].tolist()
        else:
            _, ids = self.index.search(q.reshape(1, -1), min(top_k, len(self.chunks)))
            chosen = ids[0].tolist()
        nodes = [SimpleNamespace(node=SimpleNamespace(text=self.chunks[i]['text']),
                                 chunk_id=i) for i in chosen]
        mean = np.asarray(self.embeddings[chosen[:10]]).mean(axis=0)
        return nodes, q, mean

    def predict_config(self, query_embedding, chunk_embedding):
        if self.network is None:
            raise RuntimeError('learned predictor is required')
        with torch.no_grad():
            logits, params = self.network(
                torch.tensor(query_embedding[None], device=self.device),
                torch.tensor(chunk_embedding[None], device=self.device))
        # Native network has six heads; only these two catalog entries trained.
        name = RERANKERS[int(logits[0, :len(RERANKERS)].argmax())]
        raw = params[0].cpu().numpy().astype(float)
        cfg = {'reranker': name, 'C': float(np.clip(raw[0], 2., 4.)),
               'iterations': int(np.clip(round(raw[1]), 10, 30)),
               'lambda': float(np.clip(raw[2], 0., 1.))}
        return cfg, raw.tolist()

    def search(self, question, nodes, config, budget, cache=None):
        t0 = time.perf_counter()
        scorer = CountingReranker(self.rerankers[config['reranker']], cache)
        mcts = BudgetMCTS(nodes_query=nodes, model=scorer, tokenizer=None,
                          C=config['C'], lambda_=config['lambda'], budget=budget)
        best = mcts.search(question, max_iterations=config['iterations'])
        ids = [n.chunk_id for n in best.state]
        stats = {'algorithm': 'upstream.core.mcts.MCTS.search', 'configuration': config,
                 'budget_cl100k': budget, 'context_cl100k': token_count(best.concat_text),
                 'context_words_ucb_cost': best.token_cost(), 'selected_chunks': ids,
                 'selected_doc_ids': list(dict.fromkeys(self.chunks[i]['doc_id'] for i in ids)),
                 'selected_reward': best.reranker_score, 'selected_depth': best.layer,
                 'mcts_iterations': mcts.total_trials, 'reranker_pairs_requested': scorer.requested,
                 'reranker_pairs_evaluated': scorer.evaluated, 'reranker_batches': scorer.batches,
                 'mcts_seconds': time.perf_counter() - t0}
        assert stats['context_cl100k'] <= budget
        return best.concat_text.strip(), stats

    def generate(self, question, context, base_url):
        hints = {
            'musique': 'a short answer phrase, no explanation',
            'hotpotqa': 'a short answer phrase or yes/no, no explanation',
            'nq_open': 'a short answer span of at most five words, no explanation',
            'pubmedqa': 'exactly one word: yes, no, or maybe',
            'fever': 'exactly one label: SUPPORTS, REFUTES, or NOT ENOUGH INFO',
            'qasper': 'a concise extracted span; Yes/No for yes-no questions; Unanswerable if the paper does not answer it',
        }
        prompt = (f'Use the passages to answer the question.\nAnswer format: {hints[self.task]}.\n'
                  f'Reply with the answer only.\n\nPassages:\n{context}\n\nQuestion: {question}\nAnswer:')
        with httpx.Client(timeout=50, trust_env=False) as client:
            r = client.post(base_url.rstrip('/') + '/chat/completions', json={
                'model': 'generator', 'temperature': 0, 'max_tokens': 256,
                'chat_template_kwargs': {'enable_thinking': False},
                'messages': [{'role': 'user', 'content': prompt}]})
        r.raise_for_status()
        body = r.json()
        content = (body['choices'][0]['message'].get('content') or '').strip()
        # Only formatting allowed; never make an answer from the reranker.
        return content, body.get('usage', {})

    def answer(self, question, meta, budget, base_url):
        t0 = time.perf_counter()
        nodes, q, c = self.retrieve(question, meta)
        cfg, raw = self.predict_config(q, c)
        ctx, stats = self.search(question, nodes, cfg, budget)
        answer, usage = self.generate(question, ctx, base_url)
        stats.update({'predictor_raw_params': raw, 'generator_usage': usage,
                      'retrieval_and_answer_seconds': time.perf_counter() - t0,
                      'question_sha256': hashlib.sha256(question.encode()).hexdigest()})
        return answer, stats


def build_index(root, corpus):
    root, corpus = Path(root), Path(corpus)
    (root / 'index').mkdir(parents=True, exist_ok=True)
    from data.build_faiss_index import FAISSIndexBuilder
    builder = FAISSIndexBuilder(model_name=str(root / 'models/bge-m3'), chunk_size=256,
                                batch_size=32, use_fp16=True)
    checked(builder.model.model)
    chunks = []
    for line in (corpus / 'corpus.jsonl').open():
        doc = json.loads(line)
        for c in builder.chunk_text(doc['text'], doc['doc_id'], 'document'):
            c['title'] = doc.get('title', '')
            c['text'] = (c['title'] + '\n' + c['text']).strip()
            # A too-long singleton must fit the smallest tested cost budget.
            # Native 256-word chunking can exceed 512 tokenizer tokens. Split
            # only that chunk; retain all corpus text and title provenance.
            toks = ENC.encode(c['text'], disallowed_special=())
            for start in range(0, max(len(toks), 1), 480):
                entry = dict(c); entry['text'] = ENC.decode(toks[start:start + 480])
                chunks.append(entry)
    embeddings = builder.encode_chunks_dense([c['text'] for c in chunks])
    faiss.normalize_L2(embeddings)
    (root / 'index/chunks.json').write_text(json.dumps(chunks, ensure_ascii=False))
    np.save(root / 'index/embeddings.npy', embeddings)
    return {'documents': len({c['doc_id'] for c in chunks}), 'chunks': len(chunks),
            'corpus_sha256': hashlib.sha256((corpus / 'corpus.jsonl').read_bytes()).hexdigest(),
            'embedding_model': 'BAAI/bge-m3', 'chunk_words': 256, 'overlap_words': 64,
            'long_chunk_split_cl100k': 480, 'embedding_parameters': checked(builder.model.model)}
