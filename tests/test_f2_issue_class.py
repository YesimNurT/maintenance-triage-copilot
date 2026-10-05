"""Issue-class baselines on synthetic tables; no raw data is touched."""

import numpy as np
import pandas as pd
import pytest

from mtc.models.issue_class import (
    MODELS,
    bootstrap_macro_auc_ci,
    frequency_baseline,
    multiclass_metrics,
    run_issue_experiment,
)

CLASSES = np.array(["a", "b", "c", "d"])


def _frames(signal: float, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    label = rng.choice(CLASSES, size=800, p=[0.5, 0.25, 0.15, 0.10])
    code = pd.Series(label).map({"a": 0, "b": 1, "c": 2, "d": 3}).to_numpy()
    frame = pd.DataFrame(
        {
            "x1": signal * code + rng.normal(0, 1, 800),
            "x2": signal * (code % 2) + rng.normal(0, 1, 800),
            "label": label,
        }
    )
    return frame.iloc[:560], frame.iloc[560:]


def test_frequency_baseline():
    train = pd.Series(["a"] * 5 + ["b"] * 3 + ["c"] * 2 + ["d"])
    val = pd.Series(["a", "b", "c", "d"])

    assert frequency_baseline(train, val) == {"top1_accuracy": 0.25, "top3_accuracy": 0.75}


def test_multiclass_metrics_on_perfect_and_uniform_scores():
    y = np.array(["a", "b", "c", "d"] * 5)
    perfect = np.eye(4)[np.tile(np.arange(4), 5)]
    uniform = np.full((20, 4), 0.25)

    assert multiclass_metrics(y, perfect, CLASSES)["macro_auc"] == 1.0
    assert multiclass_metrics(y, perfect, CLASSES)["top1_accuracy"] == 1.0
    assert multiclass_metrics(y, uniform, CLASSES)["macro_auc"] == 0.5
    assert multiclass_metrics(y, uniform, CLASSES)["top3_accuracy"] == pytest.approx(0.75)


def test_bootstrap_interval_contains_the_estimate():
    y = np.array(["a", "b", "c", "d"] * 25)
    rng = np.random.default_rng(0)
    proba = np.eye(4)[np.tile(np.arange(4), 25)] * 0.5 + rng.random((100, 4)) * 0.5
    proba /= proba.sum(axis=1, keepdims=True)

    low, high = bootstrap_macro_auc_ci(y, proba, CLASSES, n_boot=200, seed=0)

    assert low <= multiclass_metrics(y, proba, CLASSES)["macro_auc"] <= high


def test_experiment_finds_signal_when_present_and_not_otherwise():
    with_signal = run_issue_experiment(*_frames(2.0), ["x1", "x2"], seed=0, n_boot=50)
    without = run_issue_experiment(*_frames(0.0), ["x1", "x2"], seed=0, n_boot=50)

    for model in MODELS:
        assert with_signal[model]["macro_auc"] > 0.85
        assert without[model]["macro_auc"] < 0.6
    assert with_signal["n_train"] == 560
    assert sum(with_signal["val_class_counts"].values()) == 240
    assert with_signal["frequency_baseline"]["top1_accuracy"] == pytest.approx(0.5, abs=0.1)
