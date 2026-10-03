"""Per-flight summary features for every binary-task flight (needs make_splits first).

    uv run python scripts/build_features.py

Writes data/processed/flight_features.parquet. Reads only the engine and context
channels of the selected flights, in batches.
"""

import time
from pathlib import Path

import pandas as pd

from mtc.config import Settings, get_settings
from mtc.data.events import FLIGHT_ID
from mtc.data.features import build_features
from mtc.data.quality import VALID_RANGES


def run(settings: Settings) -> Path:
    splits = pd.read_parquet(settings.processed_dir / "splits.parquet")
    ids = splits.loc[splits["binary_task"], FLIGHT_ID].tolist()
    print(f"building features for {len(ids)} flights")
    start = time.perf_counter()
    table = build_features(settings.ngafid_raw_dir / "all_flights" / "one_parq", ids, VALID_RANGES)
    out = settings.processed_dir / "flight_features.parquet"
    table.to_parquet(out)
    print(f"written {out}: {table.shape[0]} flights x {table.shape[1]} features "
          f"in {time.perf_counter() - start:.0f} s")
    return out


if __name__ == "__main__":
    run(get_settings())
