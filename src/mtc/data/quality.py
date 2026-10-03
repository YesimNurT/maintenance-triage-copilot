"""Plausibility checks for sensor values and whole flights.

Ranges were checked against the channel quantiles of 300 random flights
(``scripts/audit_channels.py``, docs/DECISIONS.md 2026-10-03): at most 0.2% of the values
of any channel fall outside. Units: CHT, EGT and OilT in °F, OAT in °C, OilP in psi,
FFlow in gal/h, IAS in kt, VSpd in ft/min, AltMSL in ft, NormAc in g around 0. Lower
bounds of OilP and IAS leave room for sensor noise around zero.
"""

import pandas as pd

from mtc.data.events import FLIGHT_ID

VALID_RANGES: dict[str, tuple[float, float]] = {
    **{f"E1 CHT{i}": (-40.0, 600.0) for i in range(1, 5)},
    **{f"E1 EGT{i}": (-40.0, 1800.0) for i in range(1, 5)},
    "E1 RPM": (0.0, 3000.0),
    "E1 OilT": (-40.0, 300.0),
    "E1 OilP": (-5.0, 120.0),
    "E1 FFlow": (0.0, 30.0),
    "IAS": (-5.0, 200.0),
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
