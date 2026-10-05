"""Issue-class baselines on per-flight features (F2.2).

Predicts which of the MVP issue classes a flight belongs to. Run on "before" flights and,
as a control, on "after" flights: the repair removes the symptom, so class signal that
is still there after the repair comes from aircraft or period identity, not from the fault.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    top_k_accuracy_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_logistic(seed: int) -> ClassifierMixin:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed),
    )


def make_gbm(seed: int) -> ClassifierMixin:
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=500,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        class_weight="balanced",
        early_stopping=True,
        validation_fraction=0.15,
        random_state=seed,
    )


MODELS = {"logistic": make_logistic, "gbm": make_gbm}


def frequency_baseline(y_train: pd.Series, y_val: pd.Series, k: int = 3) -> dict[str, float]:
    """Accuracy of always answering the most frequent training class(es)."""
    ranked = y_train.value_counts().index
    return {
        "top1_accuracy": float((y_val == ranked[0]).mean()),
        f"top{k}_accuracy": float(y_val.isin(ranked[:k]).mean()),
    }


def multiclass_metrics(
    y: np.ndarray, proba: np.ndarray, classes: np.ndarray, k: int = 3
) -> dict[str, float]:
    """Top-1 / balanced / top-k accuracy and macro one-vs-rest AUC."""
    predicted = classes[proba.argmax(axis=1)]
    return {
        "n": int(len(y)),
        "top1_accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        f"top{k}_accuracy": float(top_k_accuracy_score(y, proba, k=k, labels=classes)),
        "macro_auc": float(roc_auc_score(y, proba, multi_class="ovr", labels=classes)),
    }


def bootstrap_macro_auc_ci(
    y: np.ndarray, proba: np.ndarray, classes: np.ndarray, n_boot: int, seed: int
) -> tuple[float, float]:
    """Percentile bootstrap interval of the macro AUC; draws missing a class are skipped."""
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == len(classes):
            aucs.append(roc_auc_score(y[idx], proba[idx], multi_class="ovr", labels=classes))
    low, high = np.quantile(aucs, [0.025, 0.975])
    return float(low), float(high)


def run_issue_experiment(
    train: pd.DataFrame, val: pd.DataFrame, columns: list[str], seed: int, n_boot: int
) -> dict[str, Any]:
    """Fit every model on ``train`` and score ``val``; both need a ``label`` column."""
    y_val = val["label"].to_numpy()
    result: dict[str, Any] = {
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "val_class_counts": {str(k): int(v) for k, v in val["label"].value_counts().items()},
        "frequency_baseline": frequency_baseline(train["label"], val["label"]),
    }
    for name, factory in MODELS.items():
        model = factory(seed).fit(train[columns], train["label"])
        proba = model.predict_proba(val[columns])
        metrics = multiclass_metrics(y_val, proba, model.classes_)
        metrics["macro_auc_ci95"] = bootstrap_macro_auc_ci(
            y_val, proba, model.classes_, n_boot, seed
        )
        result[name] = metrics
    return result
