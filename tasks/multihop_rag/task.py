"""MultiHop-RAG (Tang & Yang, COLM 2024): multi-hop questions over one shared corpus of 609 English news
articles. Source: HF yixuantt/MultiHopRAG (MultiHopRAG.json: 2,556 queries; corpus.json: 609 articles).
Pool = the whole official corpus; each document starts with a "Source | Published | Author | Category" line,
since questions name outlets and dates. Question types: inference, comparison, temporal, null
("Insufficient information.").

Splits (there is no official split; the paper evaluates every query): the 2,556 queries reuse only 981 evidence
facts, so questions are linked through shared facts into connected components (303; the largest have 481 and
410 questions). test = whole components among those of <= 40 questions (best of 200 random orders), filling the question-type
mix first (inference questions sit mostly in the large components, so they end up under-represented) and then any
type up to n_test; at most 3 test questions share one identical evidence-fact set (near-paraphrases). train =
every question outside the test components. test and train therefore share no evidence fact.
Rule SPLIT = "random_dedup" (formal-v2, tasks.prepare --rules): test = a random draw stratified by question type
(the source mix; at most 3 test questions per identical evidence-fact set); train = every other question except
those whose evidence-fact set is identical to a test question's, and (TRAIN_DEDUP = "same_answer") those that
share an evidence fact with a test question that has the same answer. Single facts can be shared, like the
supporting articles of HotpotQA.
Metric: exact match against the reference answer (the official qa_evaluate.py counts any shared word as
correct, which a padded answer games)."""
import collections
import json

from tasks.common import Ids, Pool, _targets, exact_match, mix_of, new_rng, pick, sha256_file, write_split

SRC = "benchmarks/MultiHop-RAG/raw/yixuantt__MultiHopRAG"
# sampling rules (tasks.prepare --rules; defaults = formal-v1)
SPLIT = "components"   # "components": whole evidence components in test; "random_dedup": see the docstring
TRAIN_DEDUP = "same_set"  # random_dedup: "same_set", or "same_answer" (also drop shared-fact same-answer pairs)


def prepare(root, out, n_train, n_test, seed):
    rng = new_rng(seed)
    ids = Ids(rng)
    qs = json.load(open(f"{root}/{SRC}/MultiHopRAG.json"))
    arts = json.load(open(f"{root}/{SRC}/corpus.json"))
    pool = Pool(ids)
    for a in arts:
        head = f"Source: {a['source']} | Published: {a['published_at']} | Author: {a['author']} | Category: {a['category']}"
        pool.add(a["url"], a["title"], f"{head}\n\n{a['body']}")
    facts = lambda i: [e["fact"].strip() for e in qs[i]["evidence_list"]]
    typ = lambda i: qs[i]["question_type"]

    par = {}
    def find(x):
        while par.setdefault(x, x) != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for i in range(len(qs)):
        for f in facts(i)[1:]:
            par[find(f)] = find(facts(i)[0])
    comps = collections.defaultdict(list)
    for i in range(len(qs)):
        comps[find(facts(i)[0]) if facts(i) else f"null-{i}"].append(i)
    fset = lambda i: "|".join(sorted(facts(i)))

    def members(c):  # at most 3 questions per identical evidence-fact set
        per_set, out = collections.Counter(), []
        for i in comps[c]:
            if per_set[fset(i)] < 3:
                per_set[fset(i)] += 1
                out.append(i)
        return out

    def draw(order):
        """Whole components: first those that fit the remaining type quota, then any that fit in n_test."""
        need = _targets(n_test, mix_of(range(len(qs)), typ))
        test, used = [], set()
        for relaxed in (False, True):
            for c in order:
                if c in used or len(comps[c]) > 40:
                    continue
                m = members(c)
                ct = collections.Counter(typ(i) for i in m)
                if len(test) + len(m) > n_test or (not relaxed and any(ct[t] > need.get(t, 0) for t in ct)):
                    continue
                test += m
                used.add(c)
                for t, k in ct.items():
                    need[t] = need.get(t, 0) - k
                if len(test) == n_test:
                    return test, used, sum(max(k, 0) for k in need.values())
        return test, used, None

    if SPLIT == "random_dedup":
        return _random_dedup(out, pool, qs, arts, facts, typ, fset, comps, rng, ids, n_train, n_test, root, seed)
    # 200 random component orders; keep the complete draw whose type mix is closest to the source mix
    best = None
    for _ in range(200):
        order = sorted(comps)
        rng.shuffle(order)
        test, used, dev = draw(order)
        if dev is not None and (best is None or dev < best[2]):
            best = (test, used, dev)
    if best is None:
        raise RuntimeError("multihop_rag: no complete test draw from small components")
    test, test_comps, _ = best
    train = pick([i for c in comps if c not in test_comps for i in comps[c]], n_train, rng, typ)

    records = {}
    for name, drawn in (("train", train), ("test", test)):
        records[name] = [{
            "qid": ids.new(), "question": qs[i]["query"], "meta": {}, "answers": [qs[i]["answer"]],
            "question_type": typ(i),
            "supporting_doc_ids": sorted({pool.doc_id(e["url"]) for e in qs[i]["evidence_list"]}),
            "orig_index": i,
        } for i in drawn]
    train_facts = {f for i in train for f in facts(i)}
    train_arts = {e["url"] for i in train for e in qs[i]["evidence_list"]}
    shared = sum(f in train_facts for i in test for f in facts(i))
    assert shared == 0, f"{shared} test evidence facts also back train questions"
    return write_split(out, pool, records, rng, {
        "task": "multihop_rag", "source": SRC, "seed": seed,
        "source_sha256": {f: sha256_file(f"{root}/{SRC}/{f}") for f in ("MultiHopRAG.json", "corpus.json")},
        "published_range": [min(a["published_at"] for a in arts), max(a["published_at"] for a in arts)],
        "n_components": len(comps), "n_test_components": len(test_comps),
        "type_mix": {"source": mix_of(range(len(qs)), typ), "test": mix_of(test, typ), "train": mix_of(train, typ)},
        "test_facts_shared_with_train": shared,
        "test_q_with_evidence_article_in_train": sum(any(e["url"] in train_arts for e in qs[i]["evidence_list"]) for i in test)},
        public_extra=("question_type", "supporting_doc_ids"), gold_extra=("question_type", "orig_index"))


def _random_dedup(out, pool, qs, arts, facts, typ, fset, comps, rng, ids, n_train, n_test, root, seed):
    norm = lambda a: " ".join(a.lower().split())
    test = pick(range(len(qs)), n_test, rng, typ, keys_of=lambda i: [fset(i)] if facts(i) else [], cap=3)
    tset = set(test)
    test_sets = {fset(i) for i in test if facts(i)}
    fact_answers = collections.defaultdict(set)  # evidence fact -> answers of the test questions it backs
    for i in test:
        for f in facts(i):
            fact_answers[f].add(norm(qs[i]["answer"]))
    test_q = {norm(qs[i]["query"]) for i in test}

    def clean(i):
        if i in tset or norm(qs[i]["query"]) in test_q:
            return False
        if facts(i) and fset(i) in test_sets:
            return False
        if TRAIN_DEDUP == "same_answer" and any(norm(qs[i]["answer"]) in fact_answers.get(f, ()) for f in facts(i)):
            return False
        return True
    cands = [i for i in range(len(qs)) if clean(i)]
    train = pick(cands, n_train, rng, typ)
    records = {}
    for name, drawn in (("train", train), ("test", test)):
        records[name] = [{
            "qid": ids.new(), "question": qs[i]["query"], "meta": {}, "answers": [qs[i]["answer"]],
            "question_type": typ(i),
            "supporting_doc_ids": sorted({pool.doc_id(e["url"]) for e in qs[i]["evidence_list"]}),
            "orig_index": i,
        } for i in drawn]
    train_facts = {f for i in train for f in facts(i)}
    train_sets = {fset(i) for i in train if facts(i)}
    train_arts = {e["url"] for i in train for e in qs[i]["evidence_list"]}
    same_set = sum(bool(facts(i)) and fset(i) in train_sets for i in test)
    assert same_set == 0, f"{same_set} test questions have an identical evidence set in train"
    return write_split(out, pool, records, rng, {
        "task": "multihop_rag", "source": SRC, "seed": seed, "split": "random_dedup", "train_dedup": TRAIN_DEDUP,
        "source_sha256": {f: sha256_file(f"{root}/{SRC}/{f}") for f in ("MultiHopRAG.json", "corpus.json")},
        "published_range": [min(a["published_at"] for a in arts), max(a["published_at"] for a in arts)],
        "type_mix": {"source": mix_of(range(len(qs)), typ), "test": mix_of(test, typ), "train": mix_of(train, typ)},
        "n_train_candidates": len(cands), "n_removed_from_train": len(qs) - len(test) - len(cands),
        "test_q_identical_evidence_set_in_train": same_set,
        "test_q_sharing_an_evidence_fact_with_train": sum(any(f in train_facts for f in facts(i)) for i in test),
        "test_q_with_evidence_article_in_train": sum(any(e["url"] in train_arts for e in qs[i]["evidence_list"]) for i in test)},
        public_extra=("question_type", "supporting_doc_ids"), gold_extra=("question_type", "orig_index"))


def score_one(pred, gold):
    return {"acc": exact_match(pred, gold["answers"])}
