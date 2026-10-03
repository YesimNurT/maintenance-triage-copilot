"""Features and Gate 1 model on synthetic flights; no raw data is touched."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mtc.data.features import (
    LENGTH_FEATURES,
    build_features,
    engine_feature_columns,
    flight_features,
)
from mtc.data.loading import CONTEXT_CHANNELS, ENGINE_CHANNELS
from mtc.data.quality import VALID_RANGES
from mtc.data.splits import pick_sample
from mtc.models.signal_check import (
    auc_by_group,
    bootstrap_auc_ci,
    evaluate,
    gate_passes,
    make_model,
)


def synthetic_flight(flight_id: int, seconds: int = 300, egt3_offset: float = 0.0) -> pd.DataFrame:
    """Ground - climb - cruise - descent - ground profile with plausible engine values."""
    n = seconds // 5
    ias = np.r_[np.zeros(n), np.full(n, 75.0), np.full(n, 105.0), np.full(n, 80.0), np.zeros(n)]
    vspd = np.r_[np.zeros(n), np.full(n, 700.0), np.zeros(n), np.full(n, -600.0), np.zeros(n)]
    rows = pd.DataFrame({"Master Index": flight_id, "timestep": np.arange(len(ias))})
    rows["IAS"], rows["VSpd"] = ias, vspd
    rows["AltMSL"], rows["OAT"], rows["NormAc"] = 3000.0, 60.0, 1.0
    for i in range(1, 5):
        rows[f"E1 CHT{i}"] = 350.0 + i
        rows[f"E1 EGT{i}"] = 1300.0 + 10 * i + (egt3_offset if i == 3 else 0.0)
    rows["E1 RPM"], rows["E1 OilT"], rows["E1 OilP"], rows["E1 FFlow"] = 2400.0, 190.0, 60.0, 8.0
    return rows[["Master Index", "timestep", *ENGINE_CHANNELS, *CONTEXT_CHANNELS]]


def test_flight_features_capture_single_cylinder_deviation():
    flight = synthetic_flight(1, egt3_offset=40.0).drop(columns=["Master Index", "timestep"])

    features = flight_features(flight)

    assert features["n_seconds"] == 300
    assert features["airborne_seconds"] == pytest.approx(180, abs=10)
    assert features["missing_share"] == 0.0
    assert features["cruise_E1 RPM_mean"] == 2400.0
    assert features["cruise_E1 EGT3_dev"] > features["cruise_E1 EGT1_dev"]
    assert features["cruise_EGT_spread"] == pytest.approx(60.0)


def test_build_features_masks_implausible_values(tmp_path: Path):
    flights = pd.concat([synthetic_flight(1), synthetic_flight(2)])
    flights.loc[flights["Master Index"] == 2, "E1 RPM"] = 9999.0
    flights.to_parquet(tmp_path / "p0.parquet")

    table = build_features(tmp_path, [1, 2], VALID_RANGES, batch_size=1)

    assert table.index.tolist() == [1, 2]
    assert table.loc[1, "cruise_E1 RPM_mean"] == 2400.0
    assert np.isnan(table.loc[2, "cruise_E1 RPM_mean"])


def test_engine_feature_columns_exclude_length_and_quality():
    table = pd.DataFrame(columns=[*LENGTH_FEATURES, "missing_share", "cruise_E1 RPM_mean"])
    assert engine_feature_columns(table) == ["cruise_E1 RPM_mean"]


def test_model_and_metrics_on_separable_data():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(100), np.ones(100)].astype(int)
    x = pd.DataFrame({"a": y + rng.normal(0, 0.3, 200), "b": rng.normal(0, 1, 200)})
    x.loc[0, "b"] = np.nan

    score = make_model(seed=0).fit(x, y).predict_proba(x)[:, 1]
    metrics = evaluate(y, score)
    low, high = bootstrap_auc_ci(y, score, n_boot=200, seed=0)

    assert metrics["auc"] > 0.95
    assert metrics["n"] == 200
    assert low <= metrics["auc"] <= high


def test_auc_by_group_skips_small_groups():
    y = np.array([0, 1] * 30 + [0, 1])
    score = np.array([0.2, 0.8] * 30 + [0.9, 0.1])
    groups = pd.Series(["big"] * 60 + ["small"] * 2)

    result = auc_by_group(y, score, groups, min_per_class=20)

    assert result == {"big": {"n": 60, "auc": 1.0}}


def test_gate_rule():
    assert gate_passes(ci_low=0.60, length_only_auc=0.52)
    assert not gate_passes(ci_low=0.54, length_only_auc=0.50)
    assert not gate_passes(ci_low=0.60, length_only_auc=0.62)


def test_pick_sample_takes_each_cell():
    splits = pd.DataFrame(
        {
            "Master Index": range(12),
            "split": ["train", "val", "test"] * 4,
            "before_after": ["before"] * 6 + ["after"] * 6,
            "binary_task": [True] * 11 + [False],
        }
    )

    sample = pick_sample(splits, per_group=1, seed=0)

    assert len(sample) == 6
    assert sample.groupby(["split", "before_after"]).size().eq(1).all()
