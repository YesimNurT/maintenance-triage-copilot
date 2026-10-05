"""F2.1 trend and baseline helpers on synthetic tables; no raw data is touched."""

import numpy as np
import pandas as pd
import pytest

from mtc.data.features import binary_task_table
from mtc.data.trend import median_by_day, step_at_maintenance
from mtc.models.baseline import MODELS, compare_feature_sets, feature_sets, make_gbm


def _trend_table() -> pd.DataFrame:
    days = np.repeat(np.arange(-3, 4), 30)
    return pd.DataFrame(
        {
            "date_diff": days,
            "oil": np.where(days > 0, 70.0, 68.0),
            "label": np.where(np.arange(len(days)) % 2 == 0, "gasket", "baffle"),
        }
    )


def test_median_by_day_drops_thin_days():
    table = pd.concat([_trend_table(), pd.DataFrame({"date_diff": [10], "oil": [99.0]})])

    by_day = median_by_day(table, "oil", max_abs_days=30, min_flights=20)

    assert by_day.index.tolist() == [-3, -2, -1, 0, 1, 2, 3]
    assert by_day.loc[-1, "median"] == 68.0
    assert by_day.loc[1, "median"] == 70.0
    assert by_day.loc[1, "n"] == 30


def test_step_at_maintenance_per_group_excludes_day_zero():
    table = _trend_table()
    table.loc[table["date_diff"] == 0, "oil"] = 500.0

    steps = step_at_maintenance(table, "oil", "label", window_days=2, min_flights=20)

    assert set(steps.index) == {"gasket", "baffle"}
    assert steps["step"].tolist() == [2.0, 2.0]
    assert steps["n_before"].tolist() == [30, 30]


def test_step_at_maintenance_drops_small_groups():
    table = _trend_table()
    table.loc[:5, "label"] = "rare"

    steps = step_at_maintenance(table, "oil", "label", min_flights=20)

    assert "rare" not in steps.index


def _task_frames(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, 600)
    frame = pd.DataFrame(
        {
            "cruise_E1 OilP_mean": y + rng.normal(0, 0.4, 600),
            "cruise_E1 CHT1_mean": rng.normal(0, 1, 600),
            "cruise_E1 EGT3_dev": rng.normal(0, 1, 600),
            "y": y,
        }
    )
    return frame.iloc[:400], frame.iloc[400:]


def test_feature_sets_split_on_oil():
    sets = feature_sets(["cruise_E1 OilP_mean", "cruise_E1 CHT1_mean", "cruise_E1 EGT3_dev"])

    assert sets["oil_only"] == ["cruise_E1 OilP_mean"]
    assert sets["without_oil"] == ["cruise_E1 CHT1_mean", "cruise_E1 EGT3_dev"]
    assert len(sets["all"]) == 3


def test_compare_feature_sets_finds_signal_only_where_it_is():
    train, val = _task_frames()
    sets = feature_sets([c for c in train.columns if c != "y"])

    metrics, scores = compare_feature_sets(train, val, sets, MODELS, seed=0, n_boot=100)

    for model in MODELS:
        assert metrics["oil_only"][model]["auc"] > 0.9
        assert metrics["without_oil"][model]["auc"] < 0.65
        low, high = metrics["all"][model]["auc_ci95"]
        assert low <= metrics["all"][model]["auc"] <= high
    assert scores[("all", "gbm")].shape == (200,)


def test_make_gbm_accepts_missing_values():
    train, _ = _task_frames()
    x = train.drop(columns="y").copy()
    x.iloc[::7, 0] = np.nan

    model = make_gbm(seed=0).fit(x, train["y"])

    assert model.predict_proba(x).shape == (400, 2)


def test_binary_task_table_filters_and_labels():
    features = pd.DataFrame(
        {"n_seconds": [900.0, 100.0, 900.0, 900.0], "missing_share": [0.0, 0.0, 0.5, 0.0]},
        index=pd.Index([1, 2, 3, 4], name="Master Index"),
    )
    splits = pd.DataFrame(
        {
            "Master Index": [1, 2, 3, 4],
            "split": ["train"] * 4,
            "before_after": ["before", "before", "after", "same"],
            "date_diff": [-1, -1, 1, 0],
            "label": ["gasket"] * 4,
            "in_benchmark": [True] * 4,
            "binary_task": [True, True, True, False],
        }
    )

    table = binary_task_table(features, splits, min_seconds=600, max_missing=0.2)

    assert table.index.tolist() == [1]
    assert table["y"].tolist() == [1]
    assert table.loc[1, "split"] == "train"
    assert pytest.approx(table.loc[1, "n_seconds"]) == 900.0
