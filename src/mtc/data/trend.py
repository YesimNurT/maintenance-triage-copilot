"""How a per-flight feature moves around the maintenance date.

Used to tell a servicing artefact from part degradation: servicing shows as a step at
day 0 that fades afterwards, degradation as a drift before day 0.
"""

import pandas as pd


def median_by_day(
    table: pd.DataFrame, column: str, max_abs_days: int = 30, min_flights: int = 20
) -> pd.DataFrame:
    """Median of ``column`` and flight count per ``date_diff``, for well-populated days."""
    window = table[table["date_diff"].abs() <= max_abs_days]
    stats = window.groupby("date_diff")[column].agg(median="median", n="count")
    return stats[stats["n"] >= min_flights]


def step_at_maintenance(
    table: pd.DataFrame, column: str, group: str, window_days: int = 2, min_flights: int = 20
) -> pd.DataFrame:
    """Median just before and just after maintenance per ``group``, and their difference.

    Day 0 is left out: those flights can be on either side of the repair.
    """
    days = table["date_diff"]
    before = table[(days < 0) & (days >= -window_days)].groupby(group)[column]
    after = table[(days > 0) & (days <= window_days)].groupby(group)[column]
    stats = pd.DataFrame(
        {
            "before_median": before.median(),
            "after_median": after.median(),
            "n_before": before.count(),
            "n_after": after.count(),
        }
    ).dropna()
    stats = stats[(stats["n_before"] >= min_flights) & (stats["n_after"] >= min_flights)]
    stats["step"] = stats["after_median"] - stats["before_median"]
    return stats.sort_values("step", ascending=False)
