"""F3: normal-behaviour model, residuals and health score; Gate 2 on the val split.

    uv run python scripts/f3_health_score.py

Writes data/processed/health_scores.parquet (residuals and scores of every usable
flight) and results/f3/health_score.json. Evaluation uses val only.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from mtc.config import Settings, get_settings
from mtc.data.features import usable_flights
from mtc.models.health_score import SCORE_CHANNELS, health_score, precision_at_k
from mtc.models.normal_behavior import fit_normal_model
from mtc.models.signal_check import bootstrap_auc_ci, evaluate

N_BOOT = 1000
MIN_AUC = 0.55  # docs/DECISIONS.md, 2026-10-05
KS = [50, 100, 200]


def run(settings: Settings) -> Path:
    features = pd.read_parquet(settings.processed_dir / "flight_features.parquet")
    splits = pd.read_parquet(settings.processed_dir / "splits.parquet")
    data = usable_flights(
        features, splits, settings.min_flight_seconds, settings.max_missing_share
    )
    healthy = data[(data["split"] == "train") & (data["before_after"] == "after")]
    model = fit_normal_model(healthy)
    z = model.residual_z(data)

    scores = pd.DataFrame(
        {f"score_{name}": health_score(z, ch) for name, ch in SCORE_CHANNELS.items()}
    )
    out_table = z.add_prefix("z_").join(scores).join(
        data[["split", "before_after", "label", "date_diff", "binary_task"]]
    )
    out_table.to_parquet(settings.processed_dir / "health_scores.parquet")

    val = out_table[(out_table["split"] == "val") & out_table["binary_task"]]
    y = (val["before_after"] == "before").to_numpy().astype(int)
    report: dict = {
        "n_healthy_train": int(len(healthy)),
        "conditions": model.conditions,
        "residual_scale": {k: round(v, 3) for k, v in model.residual_scale.items()},
        "n_val": int(len(val)),
        "val_before_share": float(y.mean()),
        "scores": {},
    }
    for name in SCORE_CHANNELS:
        score = val[f"score_{name}"].fillna(0.0).to_numpy()
        metrics = evaluate(y, score, threshold=np.median(score))
        metrics["auc_ci95"] = bootstrap_auc_ci(y, score, N_BOOT, settings.random_seed)
        metrics.update(precision_at_k(y, score, KS))
        metrics["ranking_value"] = bool(metrics["auc_ci95"][0] > MIN_AUC)
        report["scores"][name] = metrics
    # diagnostic only: which single residual separates before from after, and in which direction
    report["signed_residual_auc"] = {
        column.removeprefix("z_"): float(roc_auc_score(y, val[column].fillna(0.0)))
        for column in val.columns
        if column.startswith("z_")
    }
    healthy_val = val[val["before_after"] == "after"]
    report["healthy_val_score_quantiles"] = {
        name: {
            str(q): float(healthy_val[f"score_{name}"].quantile(q)) for q in (0.5, 0.9, 0.95, 0.99)
        }
        for name in SCORE_CHANNELS
    }

    print(f"healthy train flights {len(healthy)}, conditions {len(model.conditions)}, "
          f"val {len(val)} (before share {y.mean():.3f})")
    for name, m in report["scores"].items():
        low, high = m["auc_ci95"]
        at_k = "  ".join(f"{k} {m[k]:.3f}" for k in m if k.startswith("p@"))
        print(f"{name:12s} auc {m['auc']:.3f} [{low:.3f}, {high:.3f}]  {at_k}  "
              f"ranking_value {m['ranking_value']}")

    signed = "  ".join(f"{k} {v:.2f}" for k, v in report["signed_residual_auc"].items())
    print(f"signed residual auc (diagnostic): {signed}")

    out = settings.results_dir / "f3" / "health_score.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"written {out}")
    return out


if __name__ == "__main__":
    run(get_settings())
