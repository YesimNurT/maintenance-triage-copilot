"""Loading, quality and phase rules on synthetic flights; no raw data is touched."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mtc.data.loading import batched, load_flights
from mtc.data.phases import PhaseRules, label_phases
from mtc.data.quality import flight_quality, mask_out_of_range, out_of_range_share


def test_load_flights_filters_and_joins_files(tmp_path: Path):
    part = pd.DataFrame(
        {"Master Index": [1, 1, 2], "timestep": [1, 0, 0], "E1 RPM": [2400.0, 2300.0, 0.0]}
    )
    part.to_parquet(tmp_path / "p0.parquet")
    pd.DataFrame({"Master Index": [1], "timestep": [2], "E1 RPM": [2500.0]}).to_parquet(
        tmp_path / "p1.parquet"
    )
    (tmp_path / "_metadata").write_bytes(b"ignored")

    rows = load_flights(tmp_path, [1], ["E1 RPM"])

    assert rows["timestep"].tolist() == [0, 1, 2]
    assert rows["E1 RPM"].tolist() == [2300.0, 2400.0, 2500.0]


def test_batched():
    assert list(batched([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


def test_out_of_range_share_and_mask():
    df = pd.DataFrame({"E1 RPM": [2400.0, 3248.0, None, 0.0], "other": [1, 2, 3, 4]})
    ranges = {"E1 RPM": (0.0, 3000.0), "missing": (0.0, 1.0)}

    assert out_of_range_share(df, ranges) == {"E1 RPM": pytest.approx(1 / 3)}
    clean = mask_out_of_range(df, ranges)
    assert clean["E1 RPM"].isna().tolist() == [False, True, True, False]
    assert clean["other"].tolist() == [1, 2, 3, 4]


def test_flight_quality_flags_short_and_gappy_flights():
    df = pd.DataFrame(
        {
            "Master Index": [1] * 4 + [2] * 2 + [3] * 4,
            "a": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, None, None, 1.0, 1.0],
        }
    )

    quality = flight_quality(df, ["a"], min_seconds=3, max_missing=0.2)

    assert quality["n_seconds"].tolist() == [4, 2, 4]
    assert quality["missing_share"].tolist() == [0.0, 0.0, 0.5]
    assert quality["ok"].tolist() == [True, False, False]


def test_label_phases_follows_a_simple_profile():
    ias = [0.0] * 30 + [80.0] * 90 + [100.0] * 90 + [80.0] * 90 + [0.0] * 30
    vspd = [0.0] * 30 + [700.0] * 90 + [0.0] * 90 + [-600.0] * 90 + [0.0] * 30
    flight = pd.DataFrame({"IAS": ias, "VSpd": vspd})

    phase = label_phases(flight, PhaseRules(smooth_seconds=5))

    assert phase.iloc[10] == "ground"
    assert phase.iloc[60] == "climb"
    assert phase.iloc[160] == "cruise"
    assert phase.iloc[250] == "descent"
    assert phase.iloc[-5] == "ground"


def test_label_phases_ignores_single_spikes_and_marks_missing():
    vspd = np.zeros(60)
    vspd[30] = 5000.0
    flight = pd.DataFrame({"IAS": [100.0] * 59 + [None], "VSpd": vspd})
    flight.loc[50:, "IAS"] = None

    phase = label_phases(flight, PhaseRules(smooth_seconds=5))

    assert phase.iloc[30] == "cruise"
    assert phase.iloc[55] == "unknown"
