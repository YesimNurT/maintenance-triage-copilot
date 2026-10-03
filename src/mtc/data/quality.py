"""Plausibility checks for sensor values and whole flights.

Ranges are provisional (docs/DECISIONS.md, 2026-10-03) and are checked against the
channel quantiles from ``scripts/audit_channels.py``. Units follow the Garmin G1000
export: °F, psi, gal/h, rpm, kt, ft/min, ft, g.
"""

import pandas as pd

from mtc.data.events import FLIGHT_ID

VALID_RANGES: dict[str, tuple[float, float]] = {
    **{f"E1 CHT{i}": (-40.0, 600.0) for i in range(1, 5)},
    **{f"E1 EGT{i}": (-40.0, 1800.0) for i in range(1, 5)},
    "E1 RPM": (0.0, 3000.0),
    "E1 OilT": (-40.0, 300.0),
    "E1 OilP": (0.0, 120.0),
    "E1 FFlow": (0.0, 30.0),
    "IAS": (0.0, 200.0),
    "VSpd": (-4000.0, 4000.0),
    "AltMSL": (-1000.0, 18000.0),
    "OAT": (-60.0, 130.0),
    "NormAc": (-3.0, 5.0),
}


def out_of_range_share(df: pd.DataFrame, ranges: dict[str, tuple[float, float]]) -> dict:
    """Share of present values outside the range, per channel found in ``df``."""
    shares = {}
    for column, (low, high) in ranges.items():
        if column in df:
            present = df[column].dropna()
            outside = (present < low) | (present > high)
            shares[column] = float(outside.mean()) if len(present) else 0.0
    return shares


def mask_out_of_range(df: pd.DataFrame, ranges: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Copy of ``df`` with implausible values set to NaN."""
    clean = df.copy()
    for column, (low, high) in ranges.items():
        if column in clean:
            clean[column] = clean[column].where(clean[column].between(low, high))
    return clean


def flight_quality(
    df: pd.DataFrame, channels: list[str], min_seconds: int, max_missing: float = 0.2
) -> pd.DataFrame:
    """Per flight: number of rows, missing share over ``channels`` and an ``ok`` flag."""
    grouped = df.groupby(FLIGHT_ID)
    quality = pd.DataFrame(
        {
            "n_seconds": grouped.size(),
            "missing_share": grouped[channels].apply(lambda g: g.isna().to_numpy().mean()),
        }
    )
    quality["ok"] = (quality["n_seconds"] >= min_seconds) & (
        quality["missing_share"] <= max_missing
    )
    return quality
