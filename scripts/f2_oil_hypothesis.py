"""F2.1: does oil pressure step at maintenance (servicing) or drift before it (wear)?

    uv run python scripts/f2_oil_hypothesis.py

Needs `build_features.py --all`. Uses train and val flights only.
"""

import json
from pathlib import Path

import pandas as pd

from mtc.config import Settings, get_settings
from mtc.data.events import FLIGHT_ID
from mtc.data.trend import median_by_day, step_at_maintenance

COLUMNS = ["cruise_E1 OilP_mean", "cruise_E1 OilT_mean", "cruise_E1 CHT1_mean", "cruise_EGT_spread"]


def run(settings: Settings) -> Path:
    features = pd.read_parquet(settings.processed_dir / "flight_features.parquet")
    splits = pd.read_parquet(settings.processed_dir / "splits.parquet").set_index(FLIGHT_ID)
    table = features.join(splits[["split", "date_diff", "before_after", "label"]])
    usable = (table["n_seconds"] >= settings.min_flight_seconds) & (
        table["missing_share"] <= settings.max_missing_share
    )
    table = table[usable & (table["split"] != "test")]

    report: dict = {"n_flights": int(len(table)), "by_day": {}, "step_by_label": {}}
    print(f"== {len(table)} usable train/val flights")
    for column in COLUMNS:
        by_day = median_by_day(table, column)
        report["by_day"][column] = by_day.round(3).to_dict("index")
        print(f"\n== {column}: median by days from maintenance")
        print(by_day.round(2).T.to_string())
    steps = step_at_maintenance(table, COLUMNS[0], "label", settings.binary_window_days)
    report["step_by_label"] = steps.round(3).to_dict("index")
    print(f"\n== {COLUMNS[0]}: step at maintenance per label")
    print(steps.round(2).to_string())
    phase = table.groupby("before_after")[COLUMNS[0]].agg(["median", "count"])
    report["by_phase"] = phase.round(3).to_dict("index")
    print(f"\n== {COLUMNS[0]} by before / same / after")
    print(phase.round(2).to_string())

    out = settings.results_dir / "f2" / "oil_hypothesis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwritten {out}")
    return out


if __name__ == "__main__":
    run(get_settings())
