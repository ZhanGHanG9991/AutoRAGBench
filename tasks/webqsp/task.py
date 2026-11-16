"""WebQSP: questions over Freebase (1-2 relation hops from the topic entity). Source: rmanluo/RoG-webqsp
(train + validation + test, 4,700 questions with 2-hop subgraphs). test = official test, train = official train +
validation. Corpus = union of every answerable question's
subgraph as one knowledge graph (see tasks/kgqa.py)."""
from tasks.common import exact_match, token_f1
from tasks.kgqa import prepare_kgqa

SOURCE = "benchmarks/WebQSP/raw/rmanluo__RoG-webqsp/data/*.parquet"


def prepare(root, out, n_train, n_test, seed):
    return prepare_kgqa("webqsp", root, out, n_train, n_test, seed, SOURCE, test_splits=("test",),
                        train_splits=("train", "validation"))


def score_one(pred, gold):
    return {"hits1": exact_match(pred, gold["answers"]), "f1": token_f1(pred, gold["answers"])}
