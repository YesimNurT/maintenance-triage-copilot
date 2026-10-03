"""Flight-level train / val / test split (docs/DECISIONS.md, 2026-10-03).

The data has no aircraft or event id, so the unit is the flight. Flights of the paper's
benchmark subset keep their published fold so results stay comparable; every other flight
gets a seeded random fold, stratified by label and before/after.
"""

import numpy as np
import pandas as pd

from mtc.data.events import FLIGHT_ID

BENCHMARK_PHASE = {1: "before", 0: "after"}


def check_benchmark_subset(header: pd.DataFrame, benchmark: pd.DataFrame) -> None:
    """Fail if a benchmark flight is missing from the full header or disagrees with it."""
    merged = benchmark.merge(header, on=FLIGHT_ID, how="left", suffixes=("_b", ""))
    missing = merged["label"].isna()
    if missing.any():
        raise ValueError(f"{int(missing.sum())} benchmark flights are not in the full header")
    if (merged["label_b"] != merged["label"]).any():
        raise ValueError("benchmark labels differ from the full header")
    if (merged["before_after_b"].map(BENCHMARK_PHASE) != merged["before_after"]).any():
        raise ValueError("benchmark before/after differs from the full header")


def assign_folds(
    header: pd.DataFrame, benchmark: pd.DataFrame, n_folds: int, seed: int
) -> pd.Series:
    """Fold per flight, aligned to ``header.index``."""
    fold = header[FLIGHT_ID].map(benchmark.set_index(FLIGHT_ID)["fold"])
    rng = np.random.default_rng(seed)
    rest = header[fold.isna()]
    for _, index in sorted(rest.groupby(["label", "before_after"]).groups.items()):
        shuffled = rng.permutation(np.asarray(index))
        start = int(rng.integers(n_folds))
        fold.loc[shuffled] = (np.arange(len(shuffled)) + start) % n_folds
    return fold.astype("int64").rename("fold")


def split_from_fold(fold: pd.Series, val_fold: int, test_fold: int) -> pd.Series:
    """Map folds to "train" / "val" / "test"."""
    if val_fold == test_fold:
        raise ValueError("val_fold and test_fold must differ")
    split = pd.Series("train", index=fold.index, name="split")
    split[fold == val_fold] = "val"
    split[fold == test_fold] = "test"
    return split


def binary_task_mask(header: pd.DataFrame, window_days: int) -> pd.Series:
    """Flights of the binary task: before / after within ``window_days``, day 0 excluded."""
    days = header["date_diff"].abs()
    in_window = (days >= 1) & (days <= window_days)
    return in_window & header["before_after"].isin(["before", "after"])


def pick_sample(splits: pd.DataFrame, per_group: int, seed: int) -> pd.DataFrame:
    """A few binary-task flights from every split x before/after cell."""
    task = splits[splits["binary_task"]]
    shuffled = task.sample(frac=1.0, random_state=seed)
    picked = shuffled.groupby(["split", "before_after"]).head(per_group)
    return picked.sort_values(FLIGHT_ID, ignore_index=True)
