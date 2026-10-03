"""Gate 1: is there any before/after signal in simple per-flight summaries?

A linear model on standardised features is deliberately weak: if it beats chance and a
length-only model, a sequence model has something to work with. Imputation and scaling
are fitted inside the pipeline, so statistics come from the training split only.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


def make_model(seed: int) -> Pipeline:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=seed),
    )


def evaluate(y: np.ndarray, score: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    predicted = (score >= threshold).astype(int)
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "f1": float(f1_score(y, predicted)),
        "auc": float(roc_auc_score(y, score)),
    }


def bootstrap_auc_ci(
    y: np.ndarray, score: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap interval of the AUC over flights."""
    rng = np.random.default_rng(seed)
    y, score = np.asarray(y), np.asarray(score)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() != y[idx].max():
            aucs.append(roc_auc_score(y[idx], score[idx]))
    low, high = np.quantile(aucs, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def auc_by_group(
    y: np.ndarray, score: np.ndarray, groups: pd.Series, min_per_class: int = 20
) -> dict[str, dict[str, Any]]:
    """AUC inside each group with at least ``min_per_class`` flights of each class."""
    frame = pd.DataFrame({"y": np.asarray(y), "score": np.asarray(score), "g": groups.to_numpy()})
    result = {}
    for name, part in frame.groupby("g"):
        counts = part["y"].value_counts()
        if len(counts) == 2 and counts.min() >= min_per_class:
            auc = roc_auc_score(part["y"], part["score"])
            result[str(name)] = {"n": int(len(part)), "auc": float(auc)}
    return dict(sorted(result.items(), key=lambda item: -item[1]["auc"]))


def gate_passes(ci_low: float, length_only_auc: float, min_auc: float = 0.55) -> bool:
    """Gate 1 rule fixed in docs/DECISIONS.md before any result was seen."""
    return ci_low > min_auc and ci_low > length_only_auc
