"""Shared helpers for task preparation and scoring.

Every task writes the same layout (see `write_split`):
  public/corpus/corpus.jsonl   {doc_id, title, text}
  public/train.jsonl           {qid, question, meta, answers, ...task extras}   (pilot data also has dev.jsonl)
  private/test_questions.jsonl {qid, question, meta}
  private/test_gold.jsonl      {qid, answers, ...task extras}
  private/split_manifest.json
tasks/prepare.py moves private/ into the hidden store ($HIDDEN_DIR/<data_version>/<task>/).

Formal data (tasks/*/task.py prepare(root, out, n_train, n_test, seed)): the test questions come from the
benchmark's official dev/test release and the train questions from its official train release (`pick`); the
corpus is the benchmark's full corpus, independent of which questions were drawn (`Pool` for small corpora,
`StreamCorpus` for million-document ones).
"""
import collections
import hashlib
import json
import random
import re
import string
import uuid
from pathlib import Path


# ---------------------------------------------------------------- ids, hashing

class Ids:
    """Seeded random UUIDs, so ids carry no information about the source data."""

    def __init__(self, rng):
        self.rng = rng

    def new(self):
        return str(uuid.UUID(int=self.rng.getrandbits(128), version=4))


def sha1(text):
    return hashlib.sha1(text.encode()).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------- corpus pool

class Pool:
    """Deduplicated document pool; key -> doc_id."""

    def __init__(self, ids):
        self.ids = ids
        self.docs = {}
        self.order = []

    def add(self, key, title, text, **extra):
        if key not in self.docs:
            d = {"doc_id": self.ids.new(), "title": title, "text": text, **extra}
            self.docs[key] = d
            self.order.append(key)
        return self.docs[key]["doc_id"]

    def __contains__(self, key):
        return key in self.docs

    def __len__(self):
        return len(self.docs)

    def doc_id(self, key):
        return self.docs[key]["doc_id"]

    def write(self, path, rng):
        keys = list(self.order)
        rng.shuffle(keys)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for k in keys:
                f.write(json.dumps(self.docs[k], ensure_ascii=False) + "\n")


class StreamCorpus:
    """Million-document corpus written while it is read (nothing but the wanted ids is kept in memory). Documents
    keep the source order: the corpus does not depend on the drawn questions, so the order carries no split
    information. Only the keys in `want` (supporting documents of drawn questions) are remembered: key -> doc_id.
    Usage: with StreamCorpus(path, ids, want) as c: c.add(key, title, text)."""

    def __init__(self, path, ids, want=()):
        self.path, self.ids, self.want = Path(path), ids, set(want)
        self.found = {}
        self.n = 0

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "w")
        return self

    def add(self, key, title, text):
        d = self.ids.new()
        if key in self.want:
            self.found[key] = d
        self.f.write(json.dumps({"doc_id": d, "title": title, "text": text}, ensure_ascii=False) + "\n")
        self.n += 1

    def __exit__(self, *exc):
        self.f.close()

    def __len__(self):
        return self.n

    def doc_id(self, key):
        return self.found[key]

    def write(self, path, rng):  # already written (write_split interface)
        assert Path(path) == self.path, (path, self.path)


# ---------------------------------------------------------------- splitting

def disjoint_split(item_keys, strata_of, sizes, rng, per_key=3):
    """Assign items to splits so that no evidence key is shared by two splits.

    item_keys: list of evidence-key lists (one per item); strata_of: item -> stratum (hop count, label, ...);
    sizes: [("test", 200), ("dev", 100), ("train", 500)], filled in that order. An item joins split S only if
    all its keys are unused or already owned by S, and each key serves at most `per_key` items (avoids
    clusters of near-duplicate questions). The stratum mix of each split follows the overall mix.
    Returns ({item: split}, [items not used by any split]).
    """
    order = list(range(len(item_keys)))
    rng.shuffle(order)
    counts = collections.Counter(strata_of(i) for i in order)
    frac = {s: c / len(order) for s, c in counts.items()}

    def targets(n):
        t = {s: int(n * f) for s, f in frac.items()}
        for s in sorted(frac, key=lambda s: n * frac[s] - t[s], reverse=True)[: n - sum(t.values())]:
            t[s] += 1
        return t

    owner, uses, split_of = {}, collections.Counter(), {}
    for name, n in sizes:
        need = targets(n)
        for i in order:
            if not any(need.values()):
                break
            if i in split_of or need.get(strata_of(i), 0) <= 0:
                continue
            keys = set(item_keys[i])
            if any(owner.get(k, name) != name or uses[k] >= per_key for k in keys):
                continue
            split_of[i] = name
            need[strata_of(i)] -= 1
            for k in keys:
                owner[k] = name
                uses[k] += 1
        if any(need.values()):
            raise RuntimeError(f"could not fill split {name}: missing {need}")
    rest = [i for i in order if i not in split_of]
    return split_of, rest


def _targets(n, frac):
    """Largest-remainder rounding of n * frac (stratum -> fraction) to integers summing to n."""
    t = {s: int(n * f) for s, f in frac.items()}
    for s in sorted(frac, key=lambda s: (n * frac[s] - t[s], str(s)), reverse=True)[: n - sum(t.values())]:
        t[s] += 1
    return t


def mix_of(items, strata_of):
    c = collections.Counter(strata_of(i) for i in items)
    return {s: k / sum(c.values()) for s, k in sorted(c.items(), key=lambda x: str(x[0]))}


def pick(cands, n, rng, strata_of=lambda i: "all", mix=None, keys_of=None, cap=None, strict=True):
    """Draw n items from `cands` in random order so that the stratum counts follow `mix` (stratum -> fraction;
    default: the natural mix of `cands`). With keys_of/cap, each key (e.g. a supporting paragraph or a paper)
    serves at most `cap` drawn items. n=None or n >= len(cands) without a cap returns every candidate.
    strict: raise if a stratum cannot be filled; otherwise return what was found."""
    order = list(cands)
    rng.shuffle(order)
    if n is None or (n >= len(order) and cap is None):
        return order
    need = _targets(n, mix or mix_of(order, strata_of))
    # with a key cap: scarcest stratum first (fewest candidates per needed item), so that the common strata do not
    # use up the keys the rare ones depend on; random order within a stratum
    if cap is not None:
        avail = collections.Counter(strata_of(i) for i in order)
        order.sort(key=lambda i: avail[strata_of(i)] / max(need.get(strata_of(i), 0), 1e-9))
    uses, out = collections.Counter(), []
    for i in order:
        s = strata_of(i)
        if need.get(s, 0) <= 0:
            continue
        keys = set(keys_of(i)) if keys_of else ()
        if cap is not None and any(uses[k] >= cap for k in keys):
            continue
        out.append(i)
        need[s] -= 1
        for k in keys:
            uses[k] += 1
        if len(out) == n:
            break
    missing = {s: k for s, k in need.items() if k > 0}
    if missing and strict:
        raise RuntimeError(f"pick: could not fill strata {missing} (n={n}, candidates={len(order)})")
    return out


def norm_question(s):
    return " ".join(normalize_answer(s).split())


def drop_duplicates_of(train_items, test_items, question_of):
    """Train items whose normalised question text equals a test question are removed (returns kept, n_dropped)."""
    test_q = {norm_question(question_of(i)) for i in test_items}
    kept = [i for i in train_items if norm_question(question_of(i)) not in test_q]
    return kept, len(train_items) - len(kept)


def is_holdout(qid):
    """Fixed ~10% holdout of train (by qid hash) that baselines / reference systems use for tuning in place of a
    dev split. Agents get the whole train file and decide themselves whether to hold anything out."""
    return int(hashlib.sha1(qid.encode()).hexdigest(), 16) % 10 == 0


# ---------------------------------------------------------------- writing

def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_split(out, pool, records, rng, manifest, gold_extra=(), public_extra=()):
    """records: {"train": [...], ["dev": [...],] "test": [...]}, each item a dict with at least
    qid, question, meta, answers (+ extras). Writes public/ (every split but test) and private/ files."""
    out = Path(out)
    pool.write(out / "public" / "corpus" / "corpus.jsonl", rng)
    pub_keys = ["qid", "question", "meta", "answers", *public_extra]
    for name in [k for k in records if k != "test"]:
        rows = records[name]
        rng.shuffle(rows)
        write_jsonl(out / "public" / f"{name}.jsonl", [{k: r[k] for k in pub_keys if k in r} for r in rows])
    test = records["test"]
    rng.shuffle(test)
    write_jsonl(out / "private" / "test_questions.jsonl", [{k: r[k] for k in ("qid", "question", "meta")} for r in test])
    write_jsonl(out / "private" / "test_gold.jsonl",
                [{k: r[k] for k in ("qid", "answers", *public_extra, *gold_extra) if k in r} for r in test])
    manifest = dict(manifest)
    manifest.update({"n_corpus": len(pool), "n": {k: len(v) for k, v in records.items()}})
    with open(out / "private" / "split_manifest.json", "w") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
    return manifest


# ---------------------------------------------------------------- answer normalisation / metrics

def normalize_answer(s):
    """SQuAD / MuSiQue / HotpotQA normalisation."""
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred, golds):
    p = normalize_answer(pred)
    return float(any(p == normalize_answer(g) for g in golds))


def token_f1(pred, golds):
    best = 0.0
    p = normalize_answer(pred).split()
    for g in golds:
        gt = normalize_answer(g).split()
        if not p or not gt:
            best = max(best, float(p == gt))
            continue
        common = collections.Counter(p) & collections.Counter(gt)
        same = sum(common.values())
        if same == 0:
            continue
        prec, rec = same / len(p), same / len(gt)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def mean_latency(preds, cap):
    """Speed metric: mean seconds per question from request to answer over ALL questions (evaluator.py
    predictions). A measured latency counts up to `cap` (the per-question limit); a question that got no answer in
    time (timeout, also when cut by the evaluation deadline) or was never asked (evaluation deadline, dead service,
    failed delivery) counts as `cap`. Returns (mean, n_capped)."""
    vals, capped = [], 0
    for p in preds:
        t = p.get("latency")
        if t is None or t >= cap or p.get("error") == "timeout":
            capped += 1
            t = cap
        vals.append(t)
    return (sum(vals) / len(vals) if vals else None), capped


def contains_any(pred, golds):
    """PopQA-style accuracy: any gold string occurs in the prediction."""
    p = normalize_answer(pred)
    return float(any(normalize_answer(g) and normalize_answer(g) in p for g in golds))


def first_label(pred, labels):
    """Map free text to the label whose keyword occurs first, e.g. yes/no/maybe."""
    t = (pred or "").lower()
    best, pos = None, len(t) + 1
    for lab, keys in labels.items():
        for k in keys:
            m = re.search(r"\b" + re.escape(k) + r"\b", t)
            if m and m.start() < pos:
                best, pos = lab, m.start()
    return best


def new_rng(seed):
    return random.Random(seed)
