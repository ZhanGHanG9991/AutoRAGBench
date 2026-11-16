"""Pilot adapter. Retrieval mathematics are inherited from pinned upstream.

Upstream's sentence nodes live in the sentence store/mention adjacency, not
the igraph PPR projection. No generator is constructed by this module.
"""
import json
import math
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import igraph as ig
from sentence_transformers import SentenceTransformer
from src.LinearRAG import LinearRAG
from src.config import LinearRAGConfig
from src.ner import SpacyNER

COMMIT = 'bcc94e66c221f798801255efba09311d6fbcd8d6'
PARAMS = dict(max_iterations=3, iteration_threshold=0.4, passage_ratio=2,
              top_k_sentence=3, retrieval_top_k=5, damping=0.5,
              passage_node_weight=0.05, use_vectorized_retrieval=False,
              enable_hybrid_attribute_fallback=False, batch_size=128, max_workers=8)


class BoundedNER(SpacyNER):
    def batch_ner(self, hash_id_to_passage, max_workers):
        # Same upstream extraction; bounded spaCy batches avoid the upstream
        # len(corpus)//workers batch and repeated O(N) key-list construction.
        keys = list(hash_id_to_passage)
        p2e, s2e = {}, defaultdict(list)
        for i, (key, doc) in enumerate(zip(keys, self.spacy_model.pipe(
                hash_id_to_passage.values(), batch_size=16))):
            p, s = self.extract_entities_sentences(doc, key)
            p2e.update(p)
            for sent, ents in s.items():
                for ent in ents:
                    if ent not in s2e[sent]:
                        s2e[sent].append(ent)
            if i % 1000 == 0:
                print('NER_PROGRESS', i, len(keys), flush=True)
        return p2e, s2e


class AuditedLinearRAG(LinearRAG):
    def __init__(self, cfg, ner):
        # Same state as upstream constructor, sharing the loaded NER model.
        import torch
        self.config = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset_name = cfg.dataset_name
        self.load_embedding_store()
        self.llm_model = None
        self.spacy_ner = ner
        self.graph = ig.Graph(directed=False)
        self.trace = {}

    def calculate_entity_scores(self, *args):
        weights, activated = super().calculate_entity_scores(*args)
        seeds = set(args[3])
        novel = set(activated) - seeds
        self.trace.update(seed_count=len(seeds), activated_count=len(activated),
                          new_entity_count=len(novel),
                          new_entities=[self.entity_embedding_store.hash_id_to_text[k]
                                        for k in sorted(novel)[:10]])
        return weights, activated

    def run_ppr(self, node_weights):
        self.trace['ppr_calls'] = self.trace.get('ppr_calls', 0) + 1
        t = time.monotonic()
        reset_prob = np.where(np.isnan(node_weights) | (node_weights < 0), 0, node_weights)
        pagerank_scores = self.graph.personalized_pagerank(
            vertices=range(len(self.node_name_to_vertex_idx)),
            damping=self.config.damping, directed=False, weights='weight',
            reset=reset_prob, implementation='prpack')
        doc_scores = np.asarray(pagerank_scores, dtype=np.float64)[self._passage_nodes]
        k = min(self.config.retrieval_top_k, len(doc_scores))
        if k:
            candidates = np.argpartition(doc_scores, -k)[-k:]
            order = candidates[np.argsort(doc_scores[candidates])[::-1]]
        else:
            order = np.empty(0, dtype=np.int64)
        self.trace['ppr_seconds'] = time.monotonic() - t
        return ([self.passage_hash_ids[i] for i in order],
                doc_scores[order].tolist())

    def prepare(self):
        for kind in ('entity', 'passage', 'sentence'):
            store = getattr(self, kind + '_embedding_store')
            setattr(self, kind + '_hash_ids', list(store.hash_id_to_text))
            setattr(self, kind + '_embeddings', np.asarray(store.embeddings, dtype=np.float32))
        self.node_name_to_vertex_idx = {v['name']: v.index for v in self.graph.vs}
        self.vertex_idx_to_node_name = {v.index: v['name'] for v in self.graph.vs}
        self.passage_node_indices = [self.node_name_to_vertex_idx[h] for h in self.passage_hash_ids]
        self._passage_nodes = np.asarray(self.passage_node_indices, dtype=np.int64)
        self._passage_idx = {h: i for i, h in enumerate(self.passage_hash_ids)}
        self._dense_cache_key = None
        self._dense_cache_scores = None

    def _passage_similarities(self, question_embedding):
        key = np.asarray(question_embedding, dtype=np.float32).tobytes()
        if key != self._dense_cache_key:
            q = np.asarray(question_embedding, dtype=np.float32).reshape(-1, 1)
            self._dense_cache_scores = np.dot(self.passage_embeddings, q).ravel()
            self._dense_cache_key = key
        return self._dense_cache_scores

    def dense_passage_retrieval(self, question_embedding):
        """Exact dense top-k without sorting every passage in the corpus."""
        t = time.monotonic()
        similarities = self._passage_similarities(question_embedding)
        k = min(self.config.retrieval_top_k, len(similarities))
        if k:
            candidates = np.argpartition(similarities, -k)[-k:]
            order = candidates[np.argsort(similarities[candidates])[::-1]]
        else:
            order = np.empty(0, dtype=np.int64)
        self.trace['dense_seconds'] = self.trace.get('dense_seconds', 0.0) + time.monotonic() - t
        return order, similarities[order].tolist()

    def calculate_passage_scores(self, question, question_embedding, actived_entities):
        """Equivalent upstream scoring using the Tri-Graph mention adjacency.

        The upstream loop lowercases and scans every passage for every query.
        Indexed entity mentions already define the Entity-Passage candidates;
        only those passages can receive a non-zero entity bonus. Dense reset
        weights and raw occurrence bonuses retain the upstream equations.
        """
        t = time.monotonic()
        dense = self._passage_similarities(question_embedding)
        low, high = float(np.min(dense)), float(np.max(dense))
        normalized = np.ones_like(dense) if high == low else (dense - low) / (high - low)
        entity_bonus = np.zeros(len(dense), dtype=np.float32)

        for entity_hash_id, (_, entity_score, tier) in actived_entities.items():
            entity_lower = self.entity_embedding_store.hash_id_to_text[entity_hash_id].lower()
            entity_node = self.node_name_to_vertex_idx[entity_hash_id]
            denom = tier if tier >= 1 else 1
            for neighbor in self.graph.neighbors(entity_node):
                passage_hash_id = self.vertex_idx_to_node_name[neighbor]
                passage_idx = self._passage_idx.get(passage_hash_id)
                if passage_idx is None:
                    continue
                occurrences = self.passage_embedding_store.hash_id_to_text[
                    passage_hash_id].lower().count(entity_lower)
                if occurrences:
                    entity_bonus[passage_idx] += (
                        entity_score * math.log(1 + occurrences) / denom)

        passage_scores = (self.config.passage_ratio * normalized
                          + np.log1p(entity_bonus))
        weights = np.zeros(len(self.graph.vs), dtype=np.float64)
        weights[self._passage_nodes] = passage_scores * self.config.passage_node_weight
        self.trace['passage_score_seconds'] = time.monotonic() - t
        return weights

    def search(self, question):
        self.trace = {'ppr_calls': 0, 'new_entity_count': 0,
                      'index_retrieval_generation_calls': 0}
        t = time.monotonic()
        t0 = t
        emb = self.config.embedding_model.encode(question, normalize_embeddings=True,
                    show_progress_bar=False, batch_size=self.config.batch_size)
        self.trace['query_embed_seconds'] = time.monotonic() - t0
        t0 = time.monotonic()
        seeds = self.get_seed_entities(question) if self.entity_hash_ids else ([], [], [], [])
        self.trace['seed_match_seconds'] = time.monotonic() - t0
        t0 = time.monotonic()
        dense_idx, _ = self.dense_passage_retrieval(emb)
        self.trace['initial_dense_seconds'] = time.monotonic() - t0
        dense_ids = [self.passage_hash_ids[i] for i in dense_idx[:5]]
        if seeds[1]:
            ids, scores = self.graph_search_with_seed_entities(question, emb, *seeds)
        else:
            ids = [self.passage_hash_ids[i] for i in dense_idx]
            scores = []
        self.trace.update(seed_entities=seeds[1], seed_scores=[float(x) for x in seeds[3]],
                          dense_fallback=not bool(seeds[1]),
                          differs_from_dense_top5=ids[:5] != dense_ids,
                          dense_top5=dense_ids, retrieved_top5=ids[:5],
                          retrieval_seconds=time.monotonic()-t)
        return ids, scores


def model_bundle(models):
    import spacy
    spacy.require_gpu()
    embed = SentenceTransformer(str(Path(models)/'all-mpnet-base-v2'), device='cuda')
    n = sum(p.numel() for p in embed.parameters())
    assert n <= 1_000_000_000
    ner = BoundedNER('en_core_web_trf')
    return embed, ner, n


def make_rag(index, embed, ner):
    index = Path(index)
    cfg = LinearRAGConfig(dataset_name=index.name, working_dir=str(index.parent),
                          embedding_model=embed, llm_model=None, **PARAMS)
    return AuditedLinearRAG(cfg, ner)


def load_rag(index, embed, ner):
    index = Path(index)
    rag = make_rag(index, embed, ner)
    rag.graph = ig.Graph.Read_GraphML(str(index/'LinearRAG.graphml'))
    data = json.loads((index/'mention_adjacency.json').read_text())
    rag.entity_hash_id_to_sentence_hash_ids = data['e2s']
    rag.sentence_hash_id_to_entity_hash_ids = data['s2e']
    rag.prepare()
    return rag


def passages(rows):
    # Corpus-only deterministic word windows, including all text. Each prefix
    # follows the official loader, including its adjacent-passage edge rule.
    texts, metadata = [], []
    for row in rows:
        words = row.get('text', '').split()
        for start in range(0, max(1, len(words)), 220):
            text = f"{len(texts)}:{row.get('title', '')}\n" + ' '.join(words[start:start+240])
            texts.append(text)
            metadata.append({'doc_id': str(row['doc_id']), 'word_offset': start})
    return texts, metadata
