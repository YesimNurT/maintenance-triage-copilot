"""Write a small extract to data/sample/ for tests and examples (needs make_splits first).

    uv run python scripts/make_sample.py

NGAFID maintenance dataset, Yang et al. 2022, Zenodo 6624956, CC BY 4.0.
"""

from pathlib import Path

import pandas as pd

from mtc.config import Settings, get_settings
from mtc.data.events import FLIGHT_ID
from mtc.data.loading import CONTEXT_CHANNELS, ENGINE_CHANNELS, load_flights
from mtc.data.splits import pick_sample

PER_GROUP = 2


def run(settings: Settings) -> Path:
    splits = pd.read_parquet(settings.processed_dir / "splits.parquet")
    header = pick_sample(splits, PER_GROUP, settings.random_seed)
    rows = load_flights(
        settings.ngafid_raw_dir / "all_flights" / "one_parq",
        header[FLIGHT_ID].tolist(),
        ENGINE_CHANNELS + CONTEXT_CHANNELS,
    )
    channels = ENGINE_CHANNELS + CONTEXT_CHANNELS
    rows[channels] = rows[channels].astype("float32")

    out = settings.sample_dir
    out.mkdir(parents=True, exist_ok=True)
    header.to_csv(out / "header.csv", index=False)
    rows.to_parquet(out / "flights.parquet", index=False)
    size_mb = (out / "flights.parquet").stat().st_size / 1e6
    print(f"written {len(header)} flights, {len(rows)} rows, {size_mb:.1f} MB to {out}")
    return out


if __name__ == "__main__":
    run(get_settings())
