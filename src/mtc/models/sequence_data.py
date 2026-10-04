"""Fixed-length 1 Hz windows per flight for the sequence baseline (F2.3).

Every flight contributes its last ``length`` seconds (the benchmark rule of Yang et al.
2022); shorter flights are padded at the front with NaN. Normalisation statistics come
from training flights only, and NaN becomes 0 after normalisation, i.e. the training mean.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from mtc.data.events import FLIGHT_ID
from mtc.data.loading import CONTEXT_CHANNELS, ENGINE_CHANNELS, batched, load_flights
from mtc.data.quality import mask_out_of_range

SEQUENCE_CHANNELS = ENGINE_CHANNELS + CONTEXT_CHANNELS
OIL_CHANNELS = ["E1 OilT", "E1 OilP"]
CHANNEL_SETS: dict[str, list[str]] = {
    "all": SEQUENCE_CHANNELS,
    "without_oil": [c for c in SEQUENCE_CHANNELS if c not in OIL_CHANNELS],
    "oil_only": OIL_CHANNELS,
}


def flight_window(flight: pd.DataFrame, channels: Sequence[str], length: int) -> np.ndarray:
    """Last ``length`` rows of one time-ordered flight as ``[length, channels]`` float32."""
    values = flight[list(channels)].to_numpy(dtype=np.float32)[-length:]
    window = np.full((length, len(channels)), np.nan, dtype=np.float32)
    window[length - len(values) :] = values
    return window


def write_sequences(
    dataset_dir: Path,
    flight_ids: Sequence[int],
    ranges: dict[str, tuple[float, float]],
    out_path: Path,
    length: int,
    batch_size: int = 250,
) -> np.ndarray:
    """Write ``[n_flights, length, channels]`` to ``out_path`` (.npy) batch by batch.

    Rows follow the order of ``flight_ids``. A memory map keeps RAM use to one batch.
    """
    ids = list(flight_ids)
    row = {flight_id: i for i, flight_id in enumerate(ids)}
    shape = (len(ids), length, len(SEQUENCE_CHANNELS))
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=shape)
    out[:] = np.nan
    for batch in batched(ids, batch_size):
        data = mask_out_of_range(load_flights(dataset_dir, batch, SEQUENCE_CHANNELS), ranges)
        for flight_id, flight in data.groupby(FLIGHT_ID):
            out[row[flight_id]] = flight_window(flight, SEQUENCE_CHANNELS, length)
    out.flush()
    return out


def fit_channel_stats(
    sequences: np.ndarray, rows: Sequence[int], chunk: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and std over the given (training) rows, ignoring NaN."""
    n_channels = sequences.shape[2]
    count, total, total_sq = np.zeros(n_channels), np.zeros(n_channels), np.zeros(n_channels)
    rows = np.sort(np.asarray(rows))
    for start in range(0, len(rows), chunk):
        block = np.asarray(sequences[rows[start : start + chunk]], dtype=np.float64)
        block = block.reshape(-1, n_channels)
        count += np.sum(~np.isnan(block), axis=0)
        total += np.nansum(block, axis=0)
        total_sq += np.nansum(block**2, axis=0)
    mean = total / np.maximum(count, 1)
    std = np.sqrt(np.maximum(total_sq / np.maximum(count, 1) - mean**2, 0.0))
    return mean.astype(np.float32), np.where(std > 0, std, 1.0).astype(np.float32)


def normalise(window: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Standardise with training statistics; missing values become 0."""
    return np.nan_to_num((window - mean) / std, nan=0.0).astype(np.float32)


def channel_indices(selection: str) -> list[int]:
    """Positions of a named channel set inside ``SEQUENCE_CHANNELS``."""
    return [SEQUENCE_CHANNELS.index(c) for c in CHANNEL_SETS[selection]]
