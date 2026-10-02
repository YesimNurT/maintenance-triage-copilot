"""Group neighbouring flights of the NGAFID flight header.

The public data has no event or aircraft id (removed for privacy, Yang et al. 2022).
Rows ordered by ``Master Index`` form short runs that look like before -> same -> after
sequences, so a new group starts when the label changes or the phase goes backwards.

These groups are NOT the paper's maintenance events: the audit of 2026-10-02 found 9,387
groups against 2,111 reported events, and 685 groups repeat a ``number_flights_before``
value. Treat them as local flight groups only.
"""

from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

FLIGHT_ID = "Master Index"
PHASE_ORDER = {"before": 0, "same": 1, "after": 2}


def assign_event_ids(header: pd.DataFrame) -> pd.Series:
    """Event id per flight, aligned to ``header.index``.

    Rows are ordered by flight id; an event boundary is a change of label or a step back
    in phase (for example after -> before).
    """
    ordered = header.sort_values(FLIGHT_ID)
    phase = ordered["before_after"].map(PHASE_ORDER)
    if phase.isna().any():
        unknown = sorted(ordered.loc[phase.isna(), "before_after"].unique())
        raise ValueError(f"unknown before_after values: {unknown}")
    new_event = (ordered["label"] != ordered["label"].shift()) | (phase < phase.shift())
    event_id = new_event.cumsum() - 1
    return event_id.astype("int64").rename("event_id").reindex(header.index)


def summarise_events(header: pd.DataFrame, event_id: pd.Series) -> dict[str, Any]:
    """Counts used to judge whether the rebuilt events are plausible."""
    df = header.assign(event_id=event_id)
    groups = df.groupby("event_id")

    pattern = groups["before_after"].agg(
        lambda s: "+".join(p for p in PHASE_ORDER if p in set(s))
    )
    before = df[df["before_after"] == "before"]
    numbered = before[before["number_flights_before"] > 0]
    repeated = numbered.duplicated(["event_id", "number_flights_before"], keep=False)
    labels_per_event = groups["label"].first()

    return {
        "n_events": int(groups.ngroups),
        "flights_per_event": _quantiles(groups.size()),
        "before_flights_per_event": _quantiles(before.groupby("event_id").size()),
        "phase_patterns": _counts(pattern),
        "events_per_label": _counts(labels_per_event),
        "events_with_repeated_flight_number": int(numbered.loc[repeated, "event_id"].nunique()),
    }


def flight_level_values(dataset_dir: Path, column: str) -> pd.DataFrame:
    """Distinct (flight id, ``column``) pairs of a parquet dataset.

    Reads two columns, one file at a time, so memory stays small.
    """
    parts = [_distinct_pairs(path, column) for path in sorted(dataset_dir.glob("*.parquet"))]
    return pd.concat(parts).drop_duplicates(ignore_index=True)


def _distinct_pairs(path: Path, column: str) -> pd.DataFrame:
    # the flight id is either a stored column or only the pandas index of the file
    stored = FLIGHT_ID in pq.read_schema(path).names
    table = pq.read_table(path, columns=[FLIGHT_ID, column] if stored else [column]).to_pandas()
    if FLIGHT_ID not in table.columns:
        table = table.rename_axis(FLIGHT_ID).reset_index()
    return table[[FLIGHT_ID, column]].drop_duplicates()


def _quantiles(sizes: pd.Series) -> dict[str, float]:
    stats = sizes.quantile([0.0, 0.25, 0.5, 0.75, 1.0])
    names = ["min", "p25", "median", "p75", "max"]
    return {name: float(value) for name, value in zip(names, stats, strict=True)}


def _counts(values: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in values.value_counts().items()}
