"""F2.2: issue-class baselines on the MVP classes, with an after-repair control.

    uv run python scripts/f2_issue_class.py

Uses train and val flights of the binary task only; the test split is never read.
"""

import json
from pathlib import Path

import pandas as pd

from mtc.config import Settings, get_settings
from mtc.data.features import binary_task_table, engine_feature_columns
from mtc.models.baseline import feature_sets
from mtc.models.issue_class import MODELS, run_issue_experiment

N_BOOT = 1000
MIN_AUC = 0.55  # docs/DECISIONS.md, 2026-10-05
SETS = ("all", "without_oil")


def run(settings: Settings) -> Path:
    features = pd.read_parquet(settings.processed_dir / "flight_features.parquet")
    splits = pd.read_parquet(settings.processed_dir / "splits.parquet")
    data = binary_task_table(
        features, splits, settings.min_flight_seconds, settings.max_missing_share
    )
    data = data[data["label"].isin(settings.mvp_classes)]
    sets = feature_sets(engine_feature_columns(features))

    report: dict = {"classes": settings.mvp_classes}
    for phase in ("before", "after"):
        part = data[data["before_after"] == phase]
        train, val = part[part["split"] == "train"], part[part["split"] == "val"]
        report[phase] = {
            name: run_issue_experiment(train, val, sets[name], settings.random_seed, N_BOOT)
            for name in SETS
        }

    best = max(MODELS, key=lambda m: report["before"]["all"][m]["macro_auc"])
    low = report["before"]["all"][best]["macro_auc_ci95"][0]
    control = report["after"]["all"][best]["macro_auc"]
    report["decision"] = {
        "model": best,
        "class_signal_present": bool(low > MIN_AUC),
        "symptom_specific": bool(low > MIN_AUC and low > control),
    }

    for phase in ("before", "after"):
        first = report[phase]["all"]
        base = first["frequency_baseline"]
        print(f"\n== {phase} flights: train {first['n_train']}, val {first['n_val']}, "
              f"most-frequent top1 {base['top1_accuracy']:.3f} top3 {base['top3_accuracy']:.3f}")
        for name in SETS:
            for model in MODELS:
                m = report[phase][name][model]
                ci_low, ci_high = m["macro_auc_ci95"]
                print(f"{name:12s} {model:9s} macro auc {m['macro_auc']:.3f} "
                      f"[{ci_low:.3f}, {ci_high:.3f}]  top1 {m['top1_accuracy']:.3f}  "
                      f"balanced {m['balanced_accuracy']:.3f}  top3 {m['top3_accuracy']:.3f}")
    print(f"\ndecision: {report['decision']}")

    out = settings.results_dir / "f2" / "issue_class.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"written {out}")
    return out


if __name__ == "__main__":
    run(get_settings())
