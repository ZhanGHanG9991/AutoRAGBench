"""LEGO-GraphRAG on a text-derived knowledge graph (query time).

The retrieval modules are the upstream ones from LEGO-GraphRAG/Instance, unchanged:
  subgraph extraction  SBE-PPR      personalized PageRank from the topic entities, top-N nodes (SE_PPR.py)
                       SAE-EEMs     + PreRetrievalModuleTriples (EEM ranks the subgraph's triples, keeps `window`)
                       SAE-LLMs     + PreRetrievalModuleLLMTriples (EEM top-`window`, then the LLM selects)
  path retrieval       SBR-SPR      RetrievalModuleDij (shortest paths, hop 4)
                       ISAR-EEMs    RetrievalModuleSemanticModel (EEM beam search, beam 8, hop 4)
                       OSAR-EEMs    SPR + PostRetrievalModuleSemanticModel (EEM keeps the top 32 paths)
                       OSAR-LLMs    SPR + PostRetrievalModuleLLM (EEM top 32, then the LLM filters)
  generation           PERSONA + REASONING_TEMPLATE with the first 32 paths (Generate.py)
ISAR-LLMs (an LLM call per hop and topic entity) does not fit the 4-calls-per-question budget and is left out.

What the text setting adds (Freebase QA datasets ship gold topic entities; our tasks do not):
  - topic entities: one generator call extracts the question's named entities (HippoRAG query-NER prompt); each
    is linked to the exact-match node, else to the nearest node by EEM cosine. No entity -> 3 nearest nodes.
  - QASPER (meta.doc_id): the graph is restricted to the nodes extracted from that paper.
  - the answer-format line of the task is appended to the generation prompt (as in every AutoRAGBench baseline).
  - ctx="paths+passages" additionally gives the generator the source passages of the triples on the top paths
    (up to `n_passages`, in path order); ctx="paths" is the original path-only generation.
"""
import json
import os
import pickle
import re
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # our config.py shadows Instance/config.py
for _p in (os.path.join(HERE, "lego_instance"), os.path.join(os.path.dirname(HERE), "Instance")):
    if os.path.isdir(_p):
        sys.path.insert(1, _p)
        break

import httpx  # noqa: E402
import igraph as ig  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import config  # noqa: E402
from post_retrieval import PostRetrievalModuleLLM, PostRetrievalModuleSemanticModel  # noqa: E402
from pre_retrieval import PreRetrievalModuleLLMTriples, PreRetrievalModuleTriples  # noqa: E402
from retrieval import RetrievalModuleDij, RetrievalModuleSemanticModel  # noqa: E402
from utils.PromptTemplate import PERSONA, REASONING_TEMPLATE  # noqa: E402
from utils.Query import Query  # noqa: E402
from utils.SemanticModel import EmbeddingModel  # noqa: E402

INSTANCES = {  # names follow the paper (Fig. 2 of the technical report)
    "SBE-PPR+SBR-SPR": dict(sae=None, pr="spr", post=None),
    "SBE-PPR+OSAR-EEMs": dict(sae=None, pr="spr", post="eem"),
    "SBE-PPR+OSAR-LLMs": dict(sae=None, pr="spr", post="llm"),
    "SBE-PPR+ISAR-EEMs": dict(sae=None, pr="isar_eem", post=None),
    "SAE-EEMs+SBR-SPR": dict(sae="eem", pr="spr", post=None),
    "SAE-EEMs+OSAR-EEMs": dict(sae="eem", pr="spr", post="eem"),
    "SAE-EEMs+OSAR-LLMs": dict(sae="eem", pr="spr", post="llm"),
    "SAE-EEMs+ISAR-EEMs": dict(sae="eem", pr="isar_eem", post=None),
    "SAE-LLMs+OSAR-EEMs": dict(sae="llm", pr="spr", post="eem"),
    "SAE-LLMs+OSAR-LLMs": dict(sae="llm", pr="spr", post="llm"),
}

NER_SYSTEM = "You're a very effective entity extraction system."
NER_USER = ("Please extract all named entities that are important for solving the questions below.\n"
            "Place the named entities in json format.\n\nQuestion: {q}")
NER_EX_Q = "Which magazine was started first Arthur's Magazine or First for Women?"
NER_EX_A = '{"named_entities": ["First for Women", "Arthur\'s Magazine"]}'


class BudgetExceeded(RuntimeError):
    pass


class GenLLM:
    """The fixed generator behind $LLM_BASE_URL, with the interface of LEGO's utils.LLM.LLM.
    Counts calls per thread (one question = one thread) so a question can never exceed the budget."""

    def __init__(self, base_url, api_key=None, max_calls=4):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.max_calls = max_calls
        self.local = threading.local()
        self.client = httpx.Client(timeout=55)

    def reset(self):
        self.local.calls, self.local.tokens, self.local.trace = 0, 0, []

    @property
    def calls(self):
        return getattr(self.local, "calls", 0)

    def chat(self, messages, max_tokens=512):
        if self.calls >= self.max_calls:
            raise BudgetExceeded(f"call {self.calls + 1} > {self.max_calls}")
        t = time.time()
        r = self.client.post(self.url, headers=self.headers, json={
            "model": config.reasoning_model, "messages": messages, "temperature": 0, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}})
        self.local.calls = self.calls + 1
        r.raise_for_status()
        j = r.json()
        u = j.get("usage", {})
        self.local.tokens = getattr(self.local, "tokens", 0) + u.get("total_tokens", 0)
        out = j["choices"][0]["message"].get("content") or ""
        getattr(self.local, "trace", []).append(out)
        return out, u.get("prompt_tokens", 0), u.get("completion_tokens", 0), time.time() - t

    # LEGO interface
    def invoke(self, sys_prompt, user_prompt, temperature=None, max_tokens=None):
        msgs = ([{"role": "system", "content": sys_prompt}] if sys_prompt else []) + \
               [{"role": "user", "content": user_prompt}]
        return self.chat(msgs, max_tokens or config.max_tokens)

    async def ainvoke(self, sys_prompt, user_prompt, temperature=None, max_tokens=None):
        return self.invoke(sys_prompt, user_prompt, temperature, max_tokens)


def key(s):  # same normalisation as build_graph.py
    return re.sub(r"\s+", " ", s.replace("->", "-")).strip().strip(".,;:!?\"'`()[]{} ").lower()


def clean_answer(text):  # baselines/ragcore.py
    text = re.sub(r"^(final answer|answer)\s*:\s*", "", text.strip(), flags=re.I)
    return text.split("\n")[0].strip().strip('"').strip()


class LegoText:
    def __init__(self, index_dir, task_info, llm, ppr_nodes=1000, damping=0.8, sae_window=256, path_num=32,
                 n_passages=8, llm_fallback=True):
        self.task, self.llm = task_info, llm
        self.ppr_nodes, self.damping, self.sae_window = ppr_nodes, damping, sae_window
        self.path_num, self.n_passages, self.llm_fallback = path_num, n_passages, llm_fallback
        self.G = ig.Graph.Read_Pickle(os.path.join(index_dir, "graph.pkl"))
        self.names = self.G.vs["name"]
        self.key2idx = {}
        for i, n in enumerate(self.names):
            self.key2idx.setdefault(key(n), i)
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.node_emb = torch.from_numpy(np.load(os.path.join(index_dir, "node_emb.npy"))).to(self.dev)
        if self.dev == "cpu":
            self.node_emb = self.node_emb.float()
        self.triple_pids = pickle.load(open(os.path.join(index_dir, "triple_pids.pkl"), "rb"))
        self.doc_nodes = pickle.load(open(os.path.join(index_dir, "doc_nodes.pkl"), "rb"))
        self.passages = [json.loads(l) for l in open(os.path.join(index_dir, "passages.jsonl"))]
        self.eem = EmbeddingModel(config.emb_model_dir)
        self._doc_graphs, self._modules, self._lock = {}, {}, threading.Lock()

    # ------------------------------------------------------------------ modules (upstream classes)
    def module(self, kind):
        with self._lock:
            if kind not in self._modules:
                self._modules[kind] = {
                    "sae:eem": lambda: PreRetrievalModuleTriples(window=self.sae_window, semantic_type="EMB"),
                    "sae:llm": lambda: PreRetrievalModuleLLMTriples(window=self.sae_window, rank_model=self.eem,
                                                                    llm=self.llm),
                    "pr:spr": lambda: RetrievalModuleDij(hop=4),
                    "pr:isar_eem": lambda: RetrievalModuleSemanticModel(hop=4, beam_width=8, semantic_type="EMB"),
                    "post:eem": lambda: PostRetrievalModuleSemanticModel(semantic_type="EMB", window=self.path_num),
                    "post:llm": lambda: PostRetrievalModuleLLM(llm_model=self.llm, rank_model=self.eem,
                                                               top_k=self.path_num),
                }[kind]()
            return self._modules[kind]

    # ------------------------------------------------------------------ topic entities
    def ner(self, question):
        out = self.llm.chat([{"role": "system", "content": NER_SYSTEM},
                             {"role": "user", "content": NER_USER.format(q=NER_EX_Q)},
                             {"role": "assistant", "content": NER_EX_A},
                             {"role": "user", "content": NER_USER.format(q=question)}], max_tokens=128)[0]
        m = re.search(r"\{.*\}", out, flags=re.S)
        try:
            ents = json.loads(m.group(0)).get("named_entities", []) if m else []
        except json.JSONDecodeError:
            ents = re.findall(r'"([^"]+)"', out)
        return [e for e in ents if isinstance(e, str) and e.strip() and e != "named_entities"][:8]

    def encode(self, texts):
        v = self.eem.model.encode(texts, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
        return v.to(self.node_emb.dtype)

    def link(self, ents, question, scope):
        """-> local vertex ids of the scope graph (<= 5): exact name match, else nearest node by EEM cosine."""
        H, allowed, g2l = scope
        emb = self.node_emb if allowed is None else self.node_emb[torch.as_tensor(allowed, device=self.dev)]
        seeds = []
        for e in ents:
            i = self.key2idx.get(key(e))
            if i is not None and (g2l is None or i in g2l):
                seeds.append(i if g2l is None else g2l[i])
            else:
                seeds.append(int((emb @ self.encode([e])[0]).argmax()))
        if not seeds:
            seeds = (emb @ self.encode([question])[0]).topk(min(3, emb.shape[0])).indices.tolist()
        return list(dict.fromkeys(seeds))[:5]

    def scope(self, doc_id):
        """(graph, global ids of its vertices, global->local) -- the whole KG, or one paper's nodes (QASPER)."""
        allowed = self.doc_nodes.get(doc_id) if doc_id else None
        if not allowed:
            return self.G, None, None
        with self._lock:
            if doc_id not in self._doc_graphs:
                self._doc_graphs[doc_id] = (self.G.subgraph(allowed), allowed, {g: l for l, g in enumerate(allowed)})
        return self._doc_graphs[doc_id]

    # ------------------------------------------------------------------ pipeline
    def retrieve(self, question, meta, inst):
        """Runs NER + the LEGO instance; returns paths (strings, ranked) and diagnostics."""
        t0 = time.time()
        ents = self.ner(question)
        sc = self.scope((meta or {}).get("doc_id"))
        H = sc[0]
        local = self.link(ents, question, sc)
        # SBE-PPR (SE_PPR.py: igraph personalized_pagerank, damping 0.8, top-N nodes -> induced subgraph)
        ppr = np.asarray(H.personalized_pagerank(damping=self.damping, reset_vertices=local, directed=True))
        top = list(np.argsort(-ppr)[:self.ppr_nodes])
        top = list(dict.fromkeys(local + [int(x) for x in top]))
        q = Query({"id": "q", "question": question, "answers": [], "entities": [H.vs[i]["name"] for i in local]})
        q.subgraph = H.subgraph(top)
        info = {"ner": ents, "seeds": q.entities, "subgraph_nodes": q.subgraph.vcount(),
                "subgraph_edges": q.subgraph.ecount()}
        if inst["sae"]:
            q = self.module(f"sae:{inst['sae']}").process(q)
            info["sae_edges"] = q.subgraph.ecount()
        q = self.module(f"pr:{inst['pr']}").process(q)
        info["pr_paths"] = len(q.reasoning_paths)
        if inst["post"]:
            post = self.module(f"post:{inst['post']}")
            before = q
            q = post.process(q)
            if inst["post"] == "llm" and not q.reasoning_paths and self.llm_fallback:
                q.reasoning_paths = post.filter(before)  # LLM kept nothing -> its EEM pre-ranking
                info["llm_fallback"] = True
        paths = q.reasoning_paths[:self.path_num]
        info["retrieve_s"] = round(time.time() - t0, 2)
        return paths, info

    def path_passages(self, paths):
        pids = []
        for p in paths:
            nodes = [p.entity] + p.entities_path
            for h, r, t in zip(nodes, p.relational_path, nodes[1:]):
                pids += self.triple_pids.get((h, r, t), [])
        return list(dict.fromkeys(pids))[:self.n_passages]

    def generate(self, question, paths, ctx):
        user = REASONING_TEMPLATE.format(paths="\n".join(str(p) for p in paths), question=question)
        pids = []
        if ctx == "paths+passages":
            pids = self.path_passages(paths)
            if pids:
                txt = "\n\n".join(f"[{n + 1}] {self.passages[i]['title']}\n{self.passages[i]['text']}"
                                  for n, i in enumerate(pids))
                user = f"Supporting passages (sources of the reasoning paths):\n{txt}\n\n" + user
        user = user.replace("## Output:", f"Answer format: {self.task['answer_format']}.\n"
                                          f"Reply with the answer only.\n\n## Output:")
        out = self.llm.invoke(PERSONA, user, max_tokens=128)[0]
        return clean_answer(out), list(dict.fromkeys(self.passages[i]["doc_id"] for i in pids))
