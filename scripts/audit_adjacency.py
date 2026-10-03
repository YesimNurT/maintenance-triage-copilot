"""Measure whether neighbouring flights belong together, and how the benchmark folds lie.

    uv run python scripts/audit_adjacency.py
"""

import json
from pathlib import Path

import pandas as pd

from mtc.config import Settings, get_settings
from mtc.data.adjacency import flight_means, fold_contiguity, neighbour_similarity
from mtc.data.events import FLIGHT_ID, assign_event_ids

FINGERPRINT_COLUMNS = ["OAT", "volt1", "volt2"]
LAGS = [1, 2, 5, 10, 50, 200]


def run(settings: Settings) -> Path:
    raw = settings.ngafid_raw_dir
    header = pd.read_csv(raw / "all_flights" / "flight_header.csv").set_index(FLIGHT_ID)
    groups = assign_event_ids(header.reset_index()).set_axis(header.index)

    features = flight_means(raw / "all_flights" / "one_parq", FINGERPRINT_COLUMNS)
    similarity = neighbour_similarity(
        features, header["label"], groups, LAGS, settings.random_seed
    )
    folds = fold_contiguity(pd.read_csv(raw / "2days" / "flight_header.csv"))
    report = {
        "n_flights_with_features": int(len(features)),
        "n_missing_feature_values": {c: int(features[c].isna().sum()) for c in features},
        "neighbour_similarity": similarity,
        "benchmark_folds": folds,
    }

    print(f"== flights with sensor means: {len(features)}; missing: "
          f"{report['n_missing_feature_values']}")
    print("\n== median absolute difference between flights (same label only)")
    print(pd.DataFrame(similarity).round(3).to_string())
    print(f"\n== benchmark folds along Master Index: {folds}")

    out = settings.results_dir / "f1" / "adjacency_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwritten {out}")
    return out


if __name__ == "__main__":
    run(get_settings())
