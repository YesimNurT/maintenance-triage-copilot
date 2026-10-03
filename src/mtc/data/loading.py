"""Load the 1 Hz sensor rows of selected flights from the NGAFID parquet dataset."""

from collections.abc import Iterator, Sequence
from pathlib import Path

import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds

from mtc.data.events import FLIGHT_ID

ENGINE_CHANNELS = [
    "E1 CHT1", "E1 CHT2", "E1 CHT3", "E1 CHT4",
    "E1 EGT1", "E1 EGT2", "E1 EGT3", "E1 EGT4",
    "E1 RPM", "E1 OilT", "E1 OilP", "E1 FFlow",
]
# operating conditions the engine responds to; inputs of the normal-behaviour model
CONTEXT_CHANNELS = ["IAS", "VSpd", "AltMSL", "OAT", "NormAc"]
TIME = "timestep"


def load_flights(
    dataset_dir: Path, flight_ids: Sequence[int], columns: Sequence[str]
) -> pd.DataFrame:
    """Rows of ``flight_ids``, sorted by flight and time.

    Uses a parquet filter on the flight id, so only matching row groups are read and a
    flight that spans several files comes back whole.
    """
    dataset = ds.dataset(dataset_dir, format="parquet")
    table = dataset.to_table(
        columns=[FLIGHT_ID, TIME, *columns],
        filter=pc.field(FLIGHT_ID).isin(list(flight_ids)),
    )
    return table.to_pandas().sort_values([FLIGHT_ID, TIME], ignore_index=True)


def batched(items: Sequence[int], size: int) -> Iterator[list[int]]:
    """Consecutive chunks of ``items`` with at most ``size`` elements."""
    for start in range(0, len(items), size):
        yield list(items[start : start + size])
