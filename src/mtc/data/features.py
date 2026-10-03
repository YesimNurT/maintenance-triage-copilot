"""Per-flight summary features for the Gate 1 signal check.

Engine channels are summarised per flight phase (mean and spread), plus cylinder
deviations from the mean of the four cylinders, which is where a single-cylinder
symptom such as "EGT3 up" would show. Length features are kept separate so a model on
length alone can be used as a confound check.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from mtc.data.events import FLIGHT_ID
from mtc.data.loading import CONTEXT_CHANNELS, ENGINE_CHANNELS, batched, load_flights
from mtc.data.phases import label_phases
from mtc.data.quality import mask_out_of_range

SUMMARY_PHASES = ("climb", "cruise")
LENGTH_FEATURES = ["n_seconds", "airborne_seconds"]
QUALITY_FEATURES = ["missing_share"]
CHT = [f"E1 CHT{i}" for i in range(1, 5)]
EGT = [f"E1 EGT{i}" for i in range(1, 5)]


def flight_features(flight: pd.DataFrame) -> dict[str, float]:
    """Features of one cleaned flight ordered by time (engine + context channels)."""
    phase = label_phases(flight)
    features: dict[str, float] = {
        "n_seconds": float(len(flight)),
        "airborne_seconds": float(phase.isin(["climb", "cruise", "descent"]).sum()),
        "missing_share": float(flight[ENGINE_CHANNELS].isna().to_numpy().mean()),
    }
    for name in SUMMARY_PHASES:
        part = flight[phase == name]
        features[f"{name}_share"] = len(part) / max(len(flight), 1)
        for column in ENGINE_CHANNELS:
            features[f"{name}_{column}_mean"] = _mean(part[column])
            features[f"{name}_{column}_std"] = _std(part[column])
        for group, columns in (("CHT", CHT), ("EGT", EGT)):
            values = part[columns]
            features[f"{name}_{group}_spread"] = _mean(values.max(axis=1) - values.min(axis=1))
            deviation = values.sub(values.mean(axis=1), axis=0)
            for column in columns:
                features[f"{name}_{column}_dev"] = _mean(deviation[column])
    return features


def build_features(
    dataset_dir: Path,
    flight_ids: Sequence[int],
    ranges: dict[str, tuple[float, float]],
    batch_size: int = 500,
) -> pd.DataFrame:
    """Feature table indexed by flight id; implausible values are masked first."""
    rows = {}
    for batch in batched(list(flight_ids), batch_size):
        data = mask_out_of_range(
            load_flights(dataset_dir, batch, ENGINE_CHANNELS + CONTEXT_CHANNELS), ranges
        )
        for flight_id, flight in data.groupby(FLIGHT_ID):
            rows[flight_id] = flight_features(flight.reset_index(drop=True))
    table = pd.DataFrame.from_dict(rows, orient="index")
    return table.rename_axis(FLIGHT_ID).sort_index()


def engine_feature_columns(table: pd.DataFrame) -> list[str]:
    """Columns that describe engine behaviour (everything but length and quality)."""
    excluded = set(LENGTH_FEATURES) | set(QUALITY_FEATURES)
    return [c for c in table.columns if c not in excluded]


def feature_groups(columns: Sequence[str]) -> dict[str, list[str]]:
    """Engine features by what they measure, for the Gate 1 ablation.

    ``cylinder_relative`` (deviation of one cylinder from the other three, spread) is the
    group a single-cylinder symptom would show in; ``rpm_fuel_flow`` and ``phase_shares``
    mostly reflect how the aircraft was flown.
    """
    groups: dict[str, list[str]] = {
        "phase_shares": [], "rpm_fuel_flow": [], "oil": [],
        "cylinder_relative": [], "cht_egt_level": [],
    }
    for column in columns:
        if column.endswith("_share"):
            groups["phase_shares"].append(column)
        elif "RPM" in column or "FFlow" in column:
            groups["rpm_fuel_flow"].append(column)
        elif "Oil" in column:
            groups["oil"].append(column)
        elif column.endswith(("_dev", "_spread")):
            groups["cylinder_relative"].append(column)
        else:
            groups["cht_egt_level"].append(column)
    return groups


def _mean(values: pd.Series) -> float:
    return float(values.mean()) if values.notna().any() else np.nan


def _std(values: pd.Series) -> float:
    return float(values.std()) if values.notna().sum() > 1 else np.nan
