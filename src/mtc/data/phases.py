"""Rule-based flight phases from airspeed and vertical speed.

Thresholds are provisional (docs/DECISIONS.md, 2026-10-03). Signals are smoothed with a
rolling median first, so single-second spikes do not flip the phase.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

PHASES = ("ground", "climb", "cruise", "descent", "unknown")


@dataclass(frozen=True)
class PhaseRules:
    airborne_ias_kt: float = 50.0
    climb_vspd_fpm: float = 300.0
    descent_vspd_fpm: float = -300.0
    smooth_seconds: int = 15


def label_phases(flight: pd.DataFrame, rules: PhaseRules | None = None) -> pd.Series:
    """Phase per row of one flight ordered by time; needs ``IAS`` and ``VSpd``."""
    rules = rules or PhaseRules()

    def smooth(column: str) -> pd.Series:
        return flight[column].rolling(rules.smooth_seconds, center=True, min_periods=1).median()

    ias, vspd = smooth("IAS"), smooth("VSpd")
    phase = np.select(
        [
            ias.isna() | vspd.isna(),
            ias < rules.airborne_ias_kt,
            vspd > rules.climb_vspd_fpm,
            vspd < rules.descent_vspd_fpm,
        ],
        ["unknown", "ground", "climb", "descent"],
        default="cruise",
    )
    return pd.Series(phase, index=flight.index, name="phase")
