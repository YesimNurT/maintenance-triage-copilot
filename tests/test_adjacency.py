"""Adjacency checks on synthetic flights; no raw data is touched."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mtc.config import Settings
from mtc.data.adjacency import flight_means, fold_contiguity, neighbour_similarity

SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_adjacency.py"


def test_flight_means_combines_flights_split_across_files(tmp_path: Path):
    first = pd.DataFrame({"Master Index": [1, 1, 2], "OAT": [10.0, 20.0, 5.0]})
    second = pd.DataFrame({"Master Index": [1, 3], "OAT": [30.0, None]})
    first.to_parquet(tmp_path / "p0.parquet")
    second.to_parquet(tmp_path / "p1.parquet")

    means = flight_means(tmp_path, ["OAT"])

    assert means.index.tolist() == [1, 2, 3]
    assert means.loc[1, "OAT"] == pytest.approx(20.0)
    assert means.loc[2, "OAT"] == pytest.approx(5.0)
    assert np.isnan(means.loc[3, "OAT"])


def _flights(values: np.ndarray) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    index = pd.RangeIndex(1, len(values) + 1, name="Master Index")
    features = pd.DataFrame({"OAT": values}, index=index)
    labels = pd.Series("gasket", index=index)
    groups = pd.Series(np.arange(len(values)) // 4, index=index)
    return features, labels, groups


def test_neighbours_are_closer_than_random_when_order_carries_structure():
    features, labels, groups = _flights(np.repeat(np.arange(50.0), 4))

    stats = neighbour_similarity(features, labels, groups, lags=[1, 20], seed=0)["OAT"]

    assert stats["lag_1"] == 0.0
    assert stats["lag_1_within_group"] == 0.0
    assert stats["lag_1_across_groups"] == 1.0
    assert stats["lag_20"] == 5.0
    assert stats["random"] > stats["lag_20"]


def test_neighbours_match_random_when_order_is_shuffled():
    values = np.random.default_rng(1).permutation(np.repeat(np.arange(50.0), 4))
    features, labels, groups = _flights(values)

    stats = neighbour_similarity(features, labels, groups, lags=[1], seed=0)["OAT"]

    assert stats["lag_1"] == pytest.approx(stats["random"], rel=0.35)


def test_neighbour_pairs_never_cross_labels():
    features, labels, groups = _flights(np.array([0.0, 0.0, 100.0, 100.0]))
    labels.iloc[2:] = "baffle"

    stats = neighbour_similarity(features, labels, groups, lags=[1], seed=0)["OAT"]

    assert stats["lag_1"] == 0.0


def test_fold_contiguity_blocks_versus_alternating():
    blocks = pd.DataFrame({"Master Index": range(6), "fold": [0, 0, 0, 1, 1, 1]})
    alternating = pd.DataFrame({"Master Index": range(6), "fold": [0, 1, 0, 1, 0, 1]})

    assert fold_contiguity(blocks)["n_runs"] == 2
    assert fold_contiguity(blocks)["same_fold_neighbour_share"] == pytest.approx(0.8)
    assert fold_contiguity(alternating)["same_fold_neighbour_share"] == 0.0
    assert fold_contiguity(alternating)["share_if_random"] == 0.5


def test_audit_script_writes_report(tmp_path: Path, capsys):
    raw = tmp_path / "raw"
    (raw / "all_flights" / "one_parq").mkdir(parents=True)
    (raw / "2days").mkdir()
    header = pd.DataFrame(
        {
            "Master Index": [1, 2, 3, 4],
            "before_after": ["before", "after", "before", "after"],
            "label": ["gasket"] * 4,
        }
    )
    header.to_csv(raw / "all_flights" / "flight_header.csv", index=False)
    header.assign(fold=[0, 0, 1, 1]).to_csv(raw / "2days" / "flight_header.csv", index=False)
    sensors = pd.DataFrame({"Master Index": [1, 2, 3, 4], "OAT": [1.0, 1.5, 9.0, 9.5]})
    sensors.assign(volt1=28.0, volt2=28.0).to_parquet(
        raw / "all_flights" / "one_parq" / "p0.parquet"
    )
    spec = importlib.util.spec_from_file_location("audit_adjacency", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    out = module.run(Settings(ngafid_raw_dir=raw, results_dir=tmp_path / "results"))

    report = json.loads(out.read_text())
    assert report["n_flights_with_features"] == 4
    assert report["neighbour_similarity"]["OAT"]["lag_1_within_group"] == 0.5
    assert report["benchmark_folds"]["n_runs"] == 2
    assert "benchmark folds" in capsys.readouterr().out
