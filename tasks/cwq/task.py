"""CWQ (ComplexWebQuestions): compositional questions over Freebase (compositions, conjunctions, comparatives,
superlatives; up to 4 relation hops). Source: rmanluo/RoG-cwq validation + test (7,050 questions with
subgraphs). test = official test, train = official validation (RoG's CWQ train is not used; validation and test
are drawn from the same distribution). Corpus = union of every answerable question's subgraph (see tasks/kgqa.py)."""
from tasks.common import exact_match, token_f1
from tasks.kgqa import prepare_kgqa

SOURCE = "benchmarks/CWQ/raw/rmanluo__RoG-cwq/data/*.parquet"
# sampling rules (tasks.prepare --rules; defaults = formal-v1). FILE_SPLITS = the RoG splits read (their subgraphs
# form the corpus); formal-v2 reads train + validation + test and draws train from the official train only.
TRAIN_SPLITS = ("validation",)
FILE_SPLITS = ("validation", "test")


def prepare(root, out, n_train, n_test, seed):
    return prepare_kgqa("cwq", root, out, n_train, n_test, seed, SOURCE, test_splits=("test",),
                        train_splits=tuple(TRAIN_SPLITS), file_splits=tuple(FILE_SPLITS))


def score_one(pred, gold):
    return {"hits1": exact_match(pred, gold["answers"]), "f1": token_f1(pred, gold["answers"])}
