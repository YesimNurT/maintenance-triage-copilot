"""Health score and symptom profile from normal-behaviour residuals (F3)."""

import numpy as np
import pandas as pd

from mtc.models.normal_behavior import OIL_TARGETS, TARGET_CHANNELS

SCORE_CHANNELS: dict[str, list[str]] = {
    "all": TARGET_CHANNELS,
    "without_oil": [c for c in TARGET_CHANNELS if c not in OIL_TARGETS],
}
Z_CLIP = 8.0


def health_score(residual_z: pd.DataFrame, channels: list[str]) -> pd.Series:
    """Root mean square of the clipped residuals: 1 is typical for a healthy flight.

    Clipping keeps one broken sensor from dominating; missing channels are ignored.
    """
    z = residual_z[channels].clip(-Z_CLIP, Z_CLIP)
    return np.sqrt((z**2).mean(axis=1)).rename("health_score")


def symptom_profile(z_row: pd.Series, threshold: float = 2.0, top: int = 3) -> list[dict]:
    """Channels whose residual exceeds ``threshold``, largest first, with direction."""
    strong = z_row.dropna()
    strong = strong[strong.abs() >= threshold]
    ordered = strong.reindex(strong.abs().sort_values(ascending=False).index).head(top)
    return [
        {"channel": channel, "z": round(float(z), 2), "direction": "high" if z > 0 else "low"}
        for channel, z in ordered.items()
    ]


def precision_at_k(y: np.ndarray, score: np.ndarray, ks: list[int]) -> dict[str, float]:
    """Share of positives among the ``k`` highest scores, for each ``k``."""
    order = np.argsort(-np.asarray(score), kind="stable")
    ranked = np.asarray(y)[order]
    return {f"p@{k}": float(ranked[:k].mean()) for k in ks if k <= len(ranked)}
