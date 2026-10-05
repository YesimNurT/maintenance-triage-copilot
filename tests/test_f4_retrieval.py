"""Case signatures, indexes and search on synthetic data; Pinecone is faked."""

import numpy as np
import pandas as pd
import pytest

from mtc.config import Settings
from mtc.retrieval.index import CaseHit, LocalIndex, PineconeIndex, make_index
from mtc.retrieval.search import index_cases, ranked_labels, retrieval_accuracy, vote
from mtc.retrieval.signatures import SIGNATURE_COLUMNS, build_cases, signature_matrix


def _scores(n: int = 120, signal: float = 4.0, seed: int = 0) -> pd.DataFrame:
    """Two labels that push different channels when ``signal`` is non-zero."""
    rng = np.random.default_rng(seed)
    label = np.where(np.arange(n) % 3 == 0, "baffle", "gasket")
    z = rng.normal(0, 1, (n, len(SIGNATURE_COLUMNS)))
    z[label == "gasket", 0] += signal
    z[label == "baffle", 5] += signal
    table = pd.DataFrame(z, columns=SIGNATURE_COLUMNS, index=pd.RangeIndex(1, n + 1))
    table["label"] = label
    table["split"] = np.where(np.arange(n) % 4 == 0, "val", "train")
    table["before_after"] = "before"
    table["binary_task"] = True
    table["date_diff"] = -1
    return table


def test_signature_matrix_clips_and_fills():
    scores = _scores(3)
    scores.iloc[0, 0] = np.nan
    scores.iloc[1, 0] = 100.0

    matrix = signature_matrix(scores)

    assert matrix.dtype == np.float32
    assert matrix[0, 0] == 0.0
    assert matrix[1, 0] == 8.0


def test_build_cases_uses_only_training_flights_of_the_phase():
    scores = _scores()
    scores.loc[scores.index[:10], "before_after"] = "after"

    cases = build_cases(scores)

    assert set(scores.loc[cases.index, "split"]) == {"train"}
    assert set(scores.loc[cases.index, "before_after"]) == {"before"}
    assert cases["case_id"].iloc[0] == f"C{cases.index[0]}"


def test_local_index_orders_by_cosine_and_ignores_zero_vectors():
    index = LocalIndex()
    vectors = np.array([[1.0, 0.0], [0.7, 0.7], [0.0, 0.0], [-1.0, 0.0]])
    index.upsert(["a", "b", "zero", "d"], vectors, [{"label": x} for x in "wxyz"])

    hits = index.query(np.array([2.0, 0.0]), k=4)

    assert [h.case_id for h in hits] == ["a", "b", "zero", "d"]
    assert hits[0].similarity == pytest.approx(1.0)
    assert hits[2].similarity == 0.0
    assert hits[0].label == "w"
    assert LocalIndex().query(np.array([1.0, 0.0]), k=3) == []


class FakePineconeHandle:
    def __init__(self):
        self.batches: list[list[dict]] = []

    def upsert(self, vectors):
        self.batches.append(vectors)

    def query(self, vector, top_k, include_metadata):
        assert include_metadata and top_k == 2
        return {"matches": [{"id": "C1", "score": 0.9, "metadata": {"label": "gasket"}},
                            {"id": "C2", "score": 0.4, "metadata": None}]}


def test_pinecone_index_batches_upserts_and_parses_matches():
    handle = FakePineconeHandle()
    index = PineconeIndex(handle)
    ids = [f"C{i}" for i in range(250)]

    index.upsert(ids, np.ones((250, 3), dtype=np.float32), [{"label": "x"}] * 250)
    hits = index.query(np.array([1.0, 0.0, 0.0]), k=2)

    assert [len(b) for b in handle.batches] == [100, 100, 50]
    first = handle.batches[0][0]
    assert first == {"id": "C0", "values": [1.0, 1.0, 1.0], "metadata": {"label": "x"}}
    assert hits == [CaseHit("C1", "gasket", 0.9), CaseHit("C2", "", 0.4)]


def test_make_index_is_local_without_api_key():
    assert isinstance(make_index(Settings(pinecone_api_key=None), dimension=10), LocalIndex)


def test_vote_and_ranked_labels():
    hits = [CaseHit("C1", "a", 0.9), CaseHit("C2", "b", 0.8), CaseHit("C3", "b", 0.1),
            CaseHit("C4", "c", 0.95)]

    votes = vote(hits)

    assert [v["label"] for v in votes] == ["b", "c", "a"]
    assert votes[0] == {"label": "b", "cases": 2, "share": 0.5}
    assert ranked_labels(hits, fallback=["a", "d", "b"]) == ["b", "c", "a", "d"]


def test_retrieval_finds_labels_when_signatures_differ_and_not_otherwise():
    with_signal, without = _scores(signal=4.0), _scores(signal=0.0)

    def accuracy(scores):
        queries = scores[scores["split"] == "val"]
        return retrieval_accuracy(build_cases(scores), queries, k_cases=5, n_boot=100, seed=0)

    good, poor = accuracy(with_signal), accuracy(without)

    assert good["top1_accuracy"] > 0.9
    assert good["beats_frequency_baseline"] is True
    assert poor["beats_frequency_baseline"] is False
    assert good["n_cases"] + good["n_queries"] == 120
    assert good["frequency_top3"] == 1.0


def test_index_cases_stores_labels():
    index = index_cases(LocalIndex(), build_cases(_scores()))

    assert len(index.ids) == 90
    assert set(m["label"] for m in index.metadata) == {"gasket", "baffle"}
