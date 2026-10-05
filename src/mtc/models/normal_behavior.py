"""Normal-behaviour model on per-flight summaries (F3, kept thin on purpose).

For each engine target (cruise CHT, EGT, oil temperature and pressure) a ridge regression
predicts the value from how the flight was flown. It is fitted on post-maintenance
flights of the training split only, so "expected" means "expected for a healthy engine
under these operating conditions". The residual, divided by the spread of the healthy
residuals, is the evidence the rest of the system works with.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from mtc.data.features import CONTEXT_FEATURES

TARGET_CHANNELS = [
    "E1 CHT1", "E1 CHT2", "E1 CHT3", "E1 CHT4",
    "E1 EGT1", "E1 EGT2", "E1 EGT3", "E1 EGT4",
    "E1 OilT", "E1 OilP",
]
OIL_TARGETS = ["E1 OilT", "E1 OilP"]
CONDITION_FEATURES = [
    "cruise_E1 RPM_mean", "cruise_E1 RPM_std", "cruise_E1 FFlow_mean", "cruise_E1 FFlow_std",
    "climb_E1 RPM_mean", "climb_E1 FFlow_mean", "cruise_share", "climb_share",
    "airborne_seconds", *CONTEXT_FEATURES,
]


def target_column(channel: str) -> str:
    return f"cruise_{channel}_mean"


@dataclass
class NormalBehaviourModel:
    conditions: list[str]
    models: dict[str, Pipeline]
    residual_scale: dict[str, float]

    def expected(self, table: pd.DataFrame) -> pd.DataFrame:
        """Expected cruise value per target channel, indexed like ``table``."""
        x = table[self.conditions]
        return pd.DataFrame(
            {channel: model.predict(x) for channel, model in self.models.items()},
            index=table.index,
        )

    def residual_z(self, table: pd.DataFrame) -> pd.DataFrame:
        """(actual - expected) / healthy residual spread; NaN where the actual is missing."""
        expected = self.expected(table)
        actual = pd.DataFrame(
            {channel: table[target_column(channel)] for channel in self.models}, index=table.index
        )
        return (actual - expected) / pd.Series(self.residual_scale)


def fit_normal_model(healthy: pd.DataFrame, alpha: float = 1.0) -> NormalBehaviourModel:
    """Fit one ridge model per target on healthy flights.

    Condition features that are missing from the table are skipped, so the model also
    works on feature tables built before the context channels were added.
    """
    conditions = [c for c in CONDITION_FEATURES if c in healthy.columns]
    models, scale = {}, {}
    for channel in TARGET_CHANNELS:
        rows = healthy[healthy[target_column(channel)].notna()]
        model = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha)
        )
        model.fit(rows[conditions], rows[target_column(channel)])
        residual = rows[target_column(channel)] - model.predict(rows[conditions])
        # robust spread: a few broken sensors must not widen what counts as normal
        mad = float(np.median(np.abs(residual - np.median(residual))))
        models[channel], scale[channel] = model, 1.4826 * mad if mad > 0 else 1.0
    return NormalBehaviourModel(conditions, models, scale)
