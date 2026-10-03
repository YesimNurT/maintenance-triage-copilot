"""Gate 1: before/after signal check on the val split (needs build_features first).

    uv run python scripts/gate1_signal.py

The test split is never read here.
"""

import json
from pathlib import Path

import pandas as pd

from mtc.config import Settings, get_settings
from mtc.data.events import FLIGHT_ID
from mtc.data.features import LENGTH_FEATURES, engine_feature_columns
from mtc.models.signal_check import (
    auc_by_group,
    bootstrap_auc_ci,
    evaluate,
    gate_passes,
    make_model,
)

N_BOOT = 1000
MAX_MISSING = 0.2


def run(settings: Settings) -> Path:
    features = pd.read_parquet(settings.processed_dir / "flight_features.parquet")
    splits = pd.read_parquet(settings.processed_dir / "splits.parquet").set_index(FLIGHT_ID)
    data = features.join(splits[["split", "before_after", "label", "in_benchmark"]])
    usable = (data["n_seconds"] >= settings.min_flight_seconds) & (
        data["missing_share"] <= MAX_MISSING
    )
    data = data[usable & data["split"].isin(["train", "val"])]
    data = data.assign(y=(data["before_after"] == "before").astype(int))
    train, val = data[data["split"] == "train"], data[data["split"] == "val"]

    engine = engine_feature_columns(features)
    report: dict = {
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "dropped_by_quality": int((~usable).sum()),
        "val_before_share": float(val["y"].mean()),
        "majority_accuracy": float(max(val["y"].mean(), 1 - val["y"].mean())),
    }
    scores = {}
    for name, columns in (("engine", engine), ("length_only", LENGTH_FEATURES)):
        model = make_model(settings.random_seed).fit(train[columns], train["y"])
        scores[name] = model.predict_proba(val[columns])[:, 1]
        report[name] = evaluate(val["y"].to_numpy(), scores[name])

    y, score = val["y"].to_numpy(), scores["engine"]
    ci = bootstrap_auc_ci(y, score, N_BOOT, settings.random_seed)
    report["engine"]["auc_ci95"] = ci
    bench = val["in_benchmark"].to_numpy()
    report["engine_benchmark_flights"] = evaluate(y[bench], score[bench])
    report["engine_auc_by_label"] = auc_by_group(y, score, val["label"])
    report["gate1_passed"] = gate_passes(ci[0], report["length_only"]["auc"])

    print(json.dumps(report, indent=2))
    out = settings.results_dir / "f1" / "gate1_signal.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"written {out}")
    return out


if __name__ == "__main__":
    run(get_settings())
