"""Check whether flights that are neighbours in ``Master Index`` belong together.

The public data has no aircraft or event id, so a leakage-safe split has to rely on row
order. That is only defensible if neighbouring flights are measurably more alike than
random flights of the same issue label. Slow-changing channels (outside air temperature,
bus voltage) act as fingerprints of "same day" and "same aircraft".
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mtc.data.events import FLIGHT_ID, read_flight_columns


def flight_means(dataset_dir: Path, columns: list[str]) -> pd.DataFrame:
    """Per-flight mean of ``columns``, indexed by flight id.

    Sums and counts are accumulated file by file, so a flight that spans two parquet files
    is still averaged correctly and memory stays small.
    """
    sums, counts = [], []
    for path in sorted(dataset_dir.glob("*.parquet")):
        grouped = read_flight_columns(path, columns).groupby(FLIGHT_ID)
        sums.append(grouped.sum(min_count=1))
        counts.append(grouped.count())
    total = pd.concat(sums).groupby(level=0).sum(min_count=1)
    n = pd.concat(counts).groupby(level=0).sum()
    return (total / n.where(n > 0)).sort_index()


def neighbour_similarity(
    features: pd.DataFrame,
    labels: pd.Series,
    groups: pd.Series,
    lags: list[int],
    seed: int,
) -> dict[str, dict[str, float]]:
    """Median absolute difference between flights ``lag`` rows apart, per feature.

    ``features``, ``labels`` and ``groups`` share a flight-id index. Only pairs with the
    same label are compared. ``random`` is the same statistic after shuffling flights
    inside each label; ``lag_1_within_group`` / ``lag_1_across_groups`` split the direct
    neighbours by whether they fall in the same local flight group.
    """
    features = features.sort_index()
    labels = labels.reindex(features.index)
    groups = groups.reindex(features.index)
    rng = np.random.default_rng(seed)
    shuffled = features.groupby(labels, group_keys=False).transform(
        lambda s: rng.permutation(s.to_numpy())
    )

    result: dict[str, dict[str, float]] = {}
    for column in features.columns:
        x = features[column]
        stats: dict[str, float] = {}
        for lag in lags:
            diff = (x - x.shift(lag)).abs()[labels == labels.shift(lag)]
            stats[f"lag_{lag}"] = float(diff.median())
        direct = (x - x.shift(1)).abs()[labels == labels.shift(1)]
        same_group = (groups == groups.shift(1)).reindex(direct.index)
        stats["lag_1_within_group"] = float(direct[same_group].median())
        stats["lag_1_across_groups"] = float(direct[~same_group].median())
        stats["random"] = float((x - shuffled[column]).abs().median())
        result[column] = stats
    return result


def fold_contiguity(header: pd.DataFrame, fold_column: str = "fold") -> dict[str, Any]:
    """How the benchmark folds sit along the flight-id order.

    ``same_fold_neighbour_share`` near ``1 / n_folds`` means folds were assigned flight by
    flight at random; near 1 means contiguous blocks.
    """
    fold = header.sort_values(FLIGHT_ID)[fold_column]
    same = (fold == fold.shift()).iloc[1:]
    n_folds = int(fold.nunique())
    return {
        "n_flights": int(len(fold)),
        "n_folds": n_folds,
        "n_runs": int((~same).sum()) + 1,
        "same_fold_neighbour_share": float(same.mean()),
        "share_if_random": 1.0 / n_folds,
    }
