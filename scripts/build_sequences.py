"""Write 1 Hz windows of the usable train/val binary-task flights (needs build_features).

    uv run python scripts/build_sequences.py

Writes data/processed/sequences.npy (float32, [flights, seconds, channels]) and
sequences_index.parquet (one row per flight, same order). Test flights are left out.
"""

import time
from pathlib import Path

import pandas as pd

from mtc.config import Settings, get_settings
from mtc.data.features import binary_task_table
from mtc.data.quality import VALID_RANGES
from mtc.models.sequence_data import SEQUENCE_CHANNELS, write_sequences


def run(settings: Settings) -> Path:
    features = pd.read_parquet(settings.processed_dir / "flight_features.parquet")
    splits = pd.read_parquet(settings.processed_dir / "splits.parquet")
    table = binary_task_table(
        features, splits, settings.min_flight_seconds, settings.max_missing_share
    )
    index = table.loc[table["split"] != "test", ["split", "label", "y"]].sort_index()
    out = settings.processed_dir / "sequences.npy"
    size_gb = len(index) * settings.sequence_length * len(SEQUENCE_CHANNELS) * 4 / 1e9
    print(f"writing {len(index)} flights x {settings.sequence_length} s x "
          f"{len(SEQUENCE_CHANNELS)} channels ({size_gb:.1f} GB) to {out}")
    start = time.perf_counter()
    write_sequences(
        settings.ngafid_raw_dir / "all_flights" / "one_parq",
        index.index.tolist(),
        VALID_RANGES,
        out,
        settings.sequence_length,
    )
    index.reset_index().to_parquet(settings.processed_dir / "sequences_index.parquet", index=False)
    print(f"done in {time.perf_counter() - start:.0f} s")
    return out


if __name__ == "__main__":
    run(get_settings())
