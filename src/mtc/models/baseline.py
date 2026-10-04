"""Tabular baselines on per-flight features, compared across feature sets (F2.1).

The comparison answers one question: is there before/after signal outside the oil
channels? Gate 1 showed that oil pressure carries most of the linear signal.
"""

from collections.abc import Callable
from typing import Any

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier

from mtc.data.features import feature_groups
from mtc.models.signal_check import bootstrap_auc_ci, evaluate, make_model

ModelFactory = Callable[[int], ClassifierMixin]


def make_gbm(seed: int) -> HistGradientBoostingClassifier:
    """Gradient boosting: non-linear, handles missing values, early-stops on a train hold-out."""
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=500,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=seed,
    )


MODELS: dict[str, ModelFactory] = {"logistic": make_model, "gbm": make_gbm}


def feature_sets(columns: list[str]) -> dict[str, list[str]]:
    """Engine features as ``all``, ``without_oil`` and ``oil_only``."""
    oil = feature_groups(columns)["oil"]
    return {
        "all": list(columns),
        "without_oil": [c for c in columns if c not in oil],
        "oil_only": oil,
    }


def compare_feature_sets(
    train: pd.DataFrame,
    val: pd.DataFrame,
    sets: dict[str, list[str]],
    models: dict[str, ModelFactory],
    seed: int,
    n_boot: int,
) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], Any]]:
    """Fit every model on every feature set; metrics on ``val`` and the val scores.

    Both frames need a ``y`` column. Returns ``metrics[set][model]`` and
    ``scores[(set, model)]``.
    """
    y = val["y"].to_numpy()
    metrics: dict[str, dict[str, Any]] = {}
    scores = {}
    for set_name, columns in sets.items():
        metrics[set_name] = {"n_features": len(columns)}
        for model_name, factory in models.items():
            model = factory(seed).fit(train[columns], train["y"])
            score = model.predict_proba(val[columns])[:, 1]
            result = evaluate(y, score)
            result["auc_ci95"] = bootstrap_auc_ci(y, score, n_boot, seed)
            metrics[set_name][model_name] = result
            scores[(set_name, model_name)] = score
    return metrics, scores
