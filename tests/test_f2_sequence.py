"""Sequence baseline on tiny synthetic data, CPU only. Skipped when torch is missing."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import DataLoader  # noqa: E402

from mtc.config import Settings  # noqa: E402
from mtc.data.quality import VALID_RANGES  # noqa: E402
from mtc.models.inception import InceptionTime  # noqa: E402
from mtc.models.sequence_data import (  # noqa: E402
    CHANNEL_SETS,
    SEQUENCE_CHANNELS,
    channel_indices,
    fit_channel_stats,
    flight_window,
    normalise,
    write_sequences,
)
from mtc.models.train import SequenceDataset, pick_device, predict, train_model  # noqa: E402
from tests.test_f1_pipeline import _raw_dataset, _script  # noqa: E402
from tests.test_features_signal import synthetic_flight  # noqa: E402


def test_flight_window_pads_short_and_trims_long_flights():
    flight = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})

    padded = flight_window(flight, ["a", "b"], length=5)
    trimmed = flight_window(flight, ["a"], length=2)

    assert padded.shape == (5, 2)
    assert np.isnan(padded[:2]).all()
    assert padded[2:, 0].tolist() == [1.0, 2.0, 3.0]
    assert trimmed[:, 0].tolist() == [2.0, 3.0]


def test_write_sequences_follows_id_order_and_masks(tmp_path: Path):
    flights = pd.concat([synthetic_flight(1), synthetic_flight(2)])
    flights.loc[flights["Master Index"] == 2, "E1 RPM"] = 9999.0
    (tmp_path / "parq").mkdir()
    flights.to_parquet(tmp_path / "parq" / "p0.parquet")

    out = write_sequences(
        tmp_path / "parq", [2, 1], VALID_RANGES, tmp_path / "seq.npy", length=100
    )

    rpm = SEQUENCE_CHANNELS.index("E1 RPM")
    assert out.shape == (2, 100, len(SEQUENCE_CHANNELS))
    assert np.isnan(out[0, :, rpm]).all()
    assert (out[1, :, rpm] == 2400.0).all()
    assert np.load(tmp_path / "seq.npy").shape == out.shape


def test_channel_stats_use_only_the_given_rows():
    sequences = np.zeros((3, 4, 2), dtype=np.float32)
    sequences[0, :, 0], sequences[1, :, 0], sequences[2, :, 0] = 1.0, 3.0, 1000.0
    sequences[0, 0, 0] = np.nan

    mean, std = fit_channel_stats(sequences, rows=[0, 1])

    assert mean[0] == pytest.approx(15 / 7)
    assert std[1] == 1.0  # constant channel: std falls back to 1
    assert normalise(sequences[0], mean, std)[0, 0] == 0.0


def test_channel_sets():
    assert len(CHANNEL_SETS["all"]) == 17
    assert len(CHANNEL_SETS["without_oil"]) == 15
    assert [SEQUENCE_CHANNELS[i] for i in channel_indices("oil_only")] == ["E1 OilT", "E1 OilP"]


@pytest.mark.parametrize("in_channels", [1, 2, 15])
def test_inception_output_shape_for_any_length(in_channels: int):
    model = InceptionTime(in_channels, filters=4, depth=4)

    assert model(torch.zeros(3, in_channels, 64)).shape == (3,)
    assert model(torch.zeros(2, in_channels, 200)).shape == (2,)


def test_training_learns_a_level_shift():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    y = np.tile([0, 1], 60)
    sequences = rng.normal(0, 1, (120, 64, 2)).astype(np.float32)
    sequences[y == 1, :, 0] += 1.5
    mean, std = fit_channel_stats(sequences, range(80))

    def loader(rows, shuffle):
        dataset = SequenceDataset(sequences, list(rows), y[list(rows)], mean, std, [0, 1])
        return DataLoader(dataset, batch_size=16, shuffle=shuffle)

    model = InceptionTime(2, filters=4, depth=2)
    history = train_model(
        model, loader(range(80), True), loader(range(80, 120), False), y[80:],
        epochs=8, learning_rate=1e-2, patience=8, device=pick_device("cpu"), log=lambda _: None,
    )
    scores = predict(model, loader(range(80, 120), False), pick_device("cpu"))

    assert max(h["holdout_auc"] for h in history) > 0.95
    assert scores.shape == (40,)
    assert ((scores >= 0) & (scores <= 1)).all()


def test_sequence_scripts_run_end_to_end(tmp_path: Path):
    raw = tmp_path / "raw"
    _raw_dataset(raw, n=200)
    settings = Settings(
        ngafid_raw_dir=raw,
        data_dir=tmp_path / "data",
        results_dir=tmp_path / "results",
        min_flight_seconds=100,
        sequence_length=64,
        seq_epochs=2,
        seq_depth=2,
        seq_filters=4,
        seq_batch_size=16,
        device="cpu",
    )
    _script("make_splits").run(settings)
    _script("build_features").run(settings)

    sequences = _script("build_sequences").run(settings)
    out = _script("f2_sequence_baseline").run(settings, "without_oil")

    index = pd.read_parquet(settings.processed_dir / "sequences_index.parquet")
    assert set(index["split"]) == {"train", "val"}
    assert np.load(sequences).shape == (len(index), 64, 17)
    report = json.loads(out.read_text())
    assert len(report["channels"]) == 15
    assert report["n_fit"] + report["n_holdout"] == (index["split"] == "train").sum()
    assert 0.0 <= report["val"]["auc"] <= 1.0
    assert importlib.util.find_spec("torch") is not None
