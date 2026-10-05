"""Per-flight summary features (needs make_splits first).

    uv run python scripts/build_features.py         # binary-task flights
    uv run python scripts/build_features.py --all   # every flight
    uv run python scripts/build_features.py --all --rebuild   # recompute after a feature change

Writes data/processed/flight_features.parquet. Flights already in that file are kept and
not recomputed, so ``--all`` only adds the missing ones. Reads only the engine and context
channels of the selected flights, in batches.
"""

import argparse
import time
from pathlib import Path

import pandas as pd

from mtc.config import Settings, get_settings
from mtc.data.events import FLIGHT_ID
from mtc.data.features import build_features
from mtc.data.quality import VALID_RANGES


def run(settings: Settings, all_flights: bool = False, rebuild: bool = False) -> Path:
    splits = pd.read_parquet(settings.processed_dir / "splits.parquet")
    wanted = splits if all_flights else splits[splits["binary_task"]]
    out = settings.processed_dir / "flight_features.parquet"
    done = pd.read_parquet(out) if out.exists() and not rebuild else pd.DataFrame()
    ids = sorted(set(wanted[FLIGHT_ID]) - set(done.index))
    print(f"building features for {len(ids)} flights ({len(done)} already done)")
    start = time.perf_counter()
    if ids:
        dataset_dir = settings.ngafid_raw_dir / "all_flights" / "one_parq"
        new = build_features(dataset_dir, ids, VALID_RANGES)
        done = pd.concat([done, new]).sort_index().rename_axis(FLIGHT_ID)
        done.to_parquet(out)
    print(f"written {out}: {done.shape[0]} flights x {done.shape[1]} features "
          f"in {time.perf_counter() - start:.0f} s")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--all", action="store_true", help="every flight, not only the binary task")
    parser.add_argument("--rebuild", action="store_true", help="recompute existing flights too")
    args = parser.parse_args()
    run(get_settings(), all_flights=args.all, rebuild=args.rebuild)
