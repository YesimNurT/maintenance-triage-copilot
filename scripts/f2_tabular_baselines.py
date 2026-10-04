"""F2.1: tabular baselines with and without oil features, on the val split.

    uv run python scripts/f2_tabular_baselines.py

The test split is never read here.
"""

import json
from pathlib import Path

import pandas as pd

from mtc.config import Settings, get_settings
from mtc.data.features import binary_task_table, engine_feature_columns
from mtc.models.baseline import MODELS, compare_feature_sets, feature_sets
from mtc.models.signal_check import auc_by_group

N_BOOT = 1000
MIN_AUC = 0.55  # docs/DECISIONS.md, 2026-10-04


def run(settings: Settings) -> Path:
    features = pd.read_parquet(settings.processed_dir / "flight_features.parquet")
    splits = pd.read_parquet(settings.processed_dir / "splits.parquet")
    data = binary_task_table(
        features, splits, settings.min_flight_seconds, settings.max_missing_share
    )
    train, val = data[data["split"] == "train"], data[data["split"] == "val"]

    sets = feature_sets(engine_feature_columns(features))
    metrics, scores = compare_feature_sets(train, val, sets, MODELS, settings.random_seed, N_BOOT)
    best_low = max(metrics["without_oil"][name]["auc_ci95"][0] for name in MODELS)
    report = {
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "majority_accuracy": float(max(val["y"].mean(), 1 - val["y"].mean())),
        "metrics": metrics,
        "without_oil_auc_by_label": auc_by_group(
            val["y"].to_numpy(), scores[("without_oil", "gbm")], val["label"]
        ),
        "cht_egt_signal_present": bool(best_low > MIN_AUC),
    }

    print(f"train {len(train)}, val {len(val)}, majority {report['majority_accuracy']:.3f}")
    for set_name, result in metrics.items():
        for model_name in MODELS:
            m = result[model_name]
            low, high = m["auc_ci95"]
            print(f"{set_name:12s} {model_name:9s} auc {m['auc']:.3f} [{low:.3f}, {high:.3f}] "
                  f"acc {m['accuracy']:.3f}")
    print("\n== without oil, gbm: val AUC per label")
    for label, value in report["without_oil_auc_by_label"].items():
        print(f"{value['auc']:.3f}  n={value['n']:4d}  {label}")
    print(f"\ncht_egt_signal_present: {report['cht_egt_signal_present']}")

    out = settings.results_dir / "f2" / "tabular_baselines.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"written {out}")
    return out


if __name__ == "__main__":
    run(get_settings())
