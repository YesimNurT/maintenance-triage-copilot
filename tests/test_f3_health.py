"""Normal-behaviour model and health score on synthetic tables; no raw data is touched."""

import numpy as np
import pandas as pd
import pytest

from mtc.models.health_score import SCORE_CHANNELS, health_score, precision_at_k, symptom_profile
from mtc.models.normal_behavior import (
    TARGET_CHANNELS,
    fit_normal_model,
    target_column,
)


def _table(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Healthy flights: every target is a linear function of RPM plus small noise."""
    rng = np.random.default_rng(seed)
    table = pd.DataFrame(
        {
            "cruise_E1 RPM_mean": rng.normal(2300, 100, n),
            "cruise_E1 FFlow_mean": rng.normal(8, 1, n),
            "cruise_share": rng.uniform(0.3, 0.7, n),
        }
    )
    for i, channel in enumerate(TARGET_CHANNELS):
        table[target_column(channel)] = (
            100 + i + 0.1 * table["cruise_E1 RPM_mean"] + rng.normal(0, 2, n)
        )
    return table


def test_model_uses_only_available_conditions_and_predicts_healthy_flights():
    model = fit_normal_model(_table())

    z = model.residual_z(_table(seed=1))

    assert model.conditions == ["cruise_E1 RPM_mean", "cruise_E1 FFlow_mean", "cruise_share"]
    assert z.columns.tolist() == TARGET_CHANNELS
    assert z.std().between(0.7, 1.4).all()
    assert z.mean().abs().max() < 0.3


def test_shifted_channel_shows_up_in_residual_score_and_profile():
    model = fit_normal_model(_table())
    flights = _table(n=50, seed=2)
    flights[target_column("E1 OilP")] -= 12.0

    z = model.residual_z(flights)

    assert z["E1 OilP"].median() < -4
    assert health_score(z, SCORE_CHANNELS["all"]).median() > 1.5
    assert health_score(z, SCORE_CHANNELS["without_oil"]).median() < 1.3
    profile = symptom_profile(z.iloc[0])
    assert profile[0]["channel"] == "E1 OilP"
    assert profile[0]["direction"] == "low"


def test_missing_actual_gives_missing_residual_not_a_crash():
    model = fit_normal_model(_table())
    flights = _table(n=5, seed=3)
    flights.loc[0, target_column("E1 CHT1")] = np.nan

    z = model.residual_z(flights)

    assert np.isnan(z.loc[0, "E1 CHT1"])
    assert np.isfinite(health_score(z, SCORE_CHANNELS["all"]).iloc[0])


def test_health_score_clips_extreme_residuals():
    z = pd.DataFrame({c: [0.0] for c in TARGET_CHANNELS})
    z.loc[0, "E1 CHT1"] = 1000.0

    assert health_score(z, TARGET_CHANNELS).iloc[0] == pytest.approx(8 / np.sqrt(10))


def test_symptom_profile_orders_by_size_and_respects_threshold():
    row = pd.Series({"a": 1.0, "b": -3.5, "c": 2.5, "d": np.nan})

    assert symptom_profile(row, threshold=2.0) == [
        {"channel": "b", "z": -3.5, "direction": "low"},
        {"channel": "c", "z": 2.5, "direction": "high"},
    ]


def test_precision_at_k():
    y = np.array([1, 0, 1, 0])
    score = np.array([0.9, 0.8, 0.1, 0.2])

    assert precision_at_k(y, score, [1, 2, 10]) == {"p@1": 1.0, "p@2": 0.5}
