"""Residual signatures of flights and the case table built from training flights (F4)."""

import numpy as np
import pandas as pd

from mtc.models.health_score import Z_CLIP
from mtc.models.normal_behavior import TARGET_CHANNELS

SIGNATURE_COLUMNS = [f"z_{channel}" for channel in TARGET_CHANNELS]


def signature_matrix(scores: pd.DataFrame) -> np.ndarray:
    """``[flights, channels]`` float32: clipped residuals, missing values as 0."""
    z = scores[SIGNATURE_COLUMNS].clip(-Z_CLIP, Z_CLIP).fillna(0.0)
    return z.to_numpy(dtype=np.float32)


def case_id(flight_id: int) -> str:
    return f"C{int(flight_id)}"


def build_cases(scores: pd.DataFrame, phase: str = "before") -> pd.DataFrame:
    """Training flights of the binary task in ``phase`` as retrievable cases.

    ``scores`` is the table written by ``scripts/f3_health_score.py``. Only the training
    split is used, so a query from val or test can never retrieve itself or a neighbour
    from its own split.
    """
    rows = scores[
        (scores["split"] == "train") & scores["binary_task"] & (scores["before_after"] == phase)
    ]
    cases = rows[["label", "date_diff", *SIGNATURE_COLUMNS]].copy()
    cases.insert(0, "case_id", [case_id(i) for i in cases.index])
    return cases
