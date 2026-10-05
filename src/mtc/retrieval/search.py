"""Similar-case search and its evaluation (F4)."""

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from mtc.retrieval.index import CaseHit, CaseIndex, LocalIndex
from mtc.retrieval.signatures import signature_matrix


def index_cases(index: CaseIndex, cases: pd.DataFrame) -> CaseIndex:
    """Add every case with its label as metadata."""
    metadata = [{"label": label} for label in cases["label"]]
    index.upsert(cases["case_id"].tolist(), signature_matrix(cases), metadata)
    return index


def vote(hits: list[CaseHit]) -> list[dict[str, Any]]:
    """Labels among the hits by number of cases; ties go to the higher total similarity."""
    count = Counter(hit.label for hit in hits)
    strength = Counter()
    for hit in hits:
        strength[hit.label] += hit.similarity
    ranked = sorted(count, key=lambda label: (-count[label], -strength[label], label))
    return [
        {"label": label, "cases": count[label], "share": count[label] / len(hits)}
        for label in ranked
    ]


def ranked_labels(hits: list[CaseHit], fallback: list[str]) -> list[str]:
    """Voted labels first, then the remaining labels in ``fallback`` order."""
    voted = [v["label"] for v in vote(hits)]
    return voted + [label for label in fallback if label not in voted]


def retrieval_accuracy(
    cases: pd.DataFrame, queries: pd.DataFrame, k_cases: int, n_boot: int, seed: int
) -> dict[str, Any]:
    """Top-1 / top-3 label accuracy of nearest-case voting, with the frequency baseline."""
    index = index_cases(LocalIndex(), cases)
    frequency = cases["label"].value_counts().index.tolist()
    truth = queries["label"].to_numpy()
    vectors = signature_matrix(queries)
    rankings = [ranked_labels(index.query(v, k_cases), frequency) for v in vectors]
    top1 = np.array([r[0] == t for r, t in zip(rankings, truth, strict=True)])
    top3 = np.array([t in r[:3] for r, t in zip(rankings, truth, strict=True)])

    rng = np.random.default_rng(seed)
    draws = [top1[rng.integers(0, len(top1), len(top1))].mean() for _ in range(n_boot)]
    low, high = np.quantile(draws, [0.025, 0.975])
    baseline_top1 = float((truth == frequency[0]).mean())
    return {
        "n_cases": int(len(cases)),
        "n_queries": int(len(queries)),
        "top1_accuracy": float(top1.mean()),
        "top1_ci95": [float(low), float(high)],
        "top3_accuracy": float(top3.mean()),
        "frequency_top1": baseline_top1,
        "frequency_top3": float(np.isin(truth, frequency[:3]).mean()),
        "beats_frequency_baseline": bool(low > baseline_top1),
    }
