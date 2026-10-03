"""Assign every flight a fold and split; write data/processed/splits.parquet.

    uv run python scripts/make_splits.py
"""

import json
from pathlib import Path

import pandas as pd

from mtc.config import Settings, get_settings
from mtc.data.events import FLIGHT_ID
from mtc.data.splits import assign_folds, binary_task_mask, check_benchmark_subset, split_from_fold


def run(settings: Settings) -> Path:
    raw = settings.ngafid_raw_dir
    header = pd.read_csv(raw / "all_flights" / "flight_header.csv")
    benchmark = pd.read_csv(raw / "2days" / "flight_header.csv")
    check_benchmark_subset(header, benchmark)

    header["fold"] = assign_folds(header, benchmark, settings.n_folds, settings.random_seed)
    header["split"] = split_from_fold(header["fold"], settings.val_fold, settings.test_fold)
    header["in_benchmark"] = header[FLIGHT_ID].isin(benchmark[FLIGHT_ID])
    header["binary_task"] = binary_task_mask(header, settings.binary_window_days)

    out = settings.processed_dir / "splits.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    header.to_parquet(out, index=False)

    task = header[header["binary_task"]]
    summary = {
        "flights_per_split": header["split"].value_counts().to_dict(),
        "binary_task_per_split": pd.crosstab(task["split"], task["before_after"]).to_dict(),
        "binary_task_flights_in_benchmark": int(task["in_benchmark"].sum()),
        "benchmark_flights": int(len(benchmark)),
    }
    print(json.dumps(summary, indent=2))
    report = settings.results_dir / "f1" / "splits_summary.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, indent=2))
    print(f"written {out} and {report}")
    return out


if __name__ == "__main__":
    run(get_settings())
