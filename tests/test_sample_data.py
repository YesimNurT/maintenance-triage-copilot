"""Checks on the real extract in data/sample (written by scripts/make_sample.py)."""

from pathlib import Path

import pandas as pd
import pytest

from mtc.data.events import FLIGHT_ID
from mtc.data.features import flight_features
from mtc.data.loading import CONTEXT_CHANNELS, ENGINE_CHANNELS
from mtc.data.quality import VALID_RANGES, mask_out_of_range

SAMPLE = Path(__file__).parents[1] / "data" / "sample"

pytestmark = pytest.mark.skipif(
    not (SAMPLE / "flights.parquet").exists(), reason="data/sample not built"
)


def test_sample_has_every_split_and_all_channels():
    header = pd.read_csv(SAMPLE / "header.csv")
    flights = pd.read_parquet(SAMPLE / "flights.parquet")

    assert set(header["split"]) == {"train", "val", "test"}
    assert set(header["before_after"]) == {"before", "after"}
    assert set(flights[FLIGHT_ID]) == set(header[FLIGHT_ID])
    assert set(ENGINE_CHANNELS + CONTEXT_CHANNELS) <= set(flights.columns)


def test_features_on_real_flights_are_physically_plausible():
    flights = mask_out_of_range(pd.read_parquet(SAMPLE / "flights.parquet"), VALID_RANGES)

    for _, flight in flights.groupby(FLIGHT_ID):
        features = flight_features(flight.reset_index(drop=True))
        assert features["cruise_share"] > 0.1
        assert 1500 < features["cruise_E1 RPM_mean"] < 2800
        assert 200 < features["cruise_E1 CHT1_mean"] < 450
