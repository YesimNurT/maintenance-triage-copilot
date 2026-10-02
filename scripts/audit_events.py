"""Rebuild maintenance events from the header and check that they are plausible.

    uv run python scripts/audit_events.py            # header only, fast
    uv run python scripts/audit_events.py --cluster  # also scan the parquet `cluster` column
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from mtc.config import Settings, get_settings
from mtc.data.events import FLIGHT_ID, assign_event_ids, flight_level_values, summarise_events


def run(settings: Settings, with_cluster: bool = False) -> Path:
    root = settings.ngafid_raw_dir / "all_flights"
    header = pd.read_csv(root / "flight_header.csv").sort_values(FLIGHT_ID, ignore_index=True)
    header["event_id"] = assign_event_ids(header)
    report = summarise_events(header, header["event_id"])

    print("== first 40 rows with rebuilt event id")
    print(header.drop(columns=["hierarchy"]).head(40).to_string())

    before = header[header["before_after"] == "before"]
    by_number = before.groupby("number_flights_before")["date_diff"].agg(["count", "min", "max"])
    report["date_diff_by_flight_number"] = by_number.to_dict("index")
    print("\n== 'before' flights: date_diff by number_flights_before")
    print(by_number.to_string())

    print(f"\n== events: {report['n_events']} (paper: 2,111)")
    for key in ("flights_per_event", "before_flights_per_event", "phase_patterns"):
        print(f"- {key}: {report[key]}")
    print(f"- events_with_repeated_flight_number: {report['events_with_repeated_flight_number']}")
    print("\n== events per label")
    for label, n in report["events_per_label"].items():
        print(f"{n:5d}  {label}")

    if with_cluster:
        clusters = flight_level_values(root / "one_parq", "cluster")
        merged = clusters.merge(header[[FLIGHT_ID, "label"]], on=FLIGHT_ID, how="left")
        report["cluster"] = {
            "n_values": int(clusters["cluster"].nunique()),
            "max_values_per_flight": int(clusters.groupby(FLIGHT_ID)["cluster"].nunique().max()),
            "max_labels_per_cluster": int(merged.groupby("cluster")["label"].nunique().max()),
            "max_clusters_per_label": int(merged.groupby("label")["cluster"].nunique().max()),
            "flights_missing_in_header": int(merged["label"].isna().sum()),
        }
        print(f"\n== parquet `cluster` column: {report['cluster']}")

    out = settings.results_dir / "f1" / "event_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwritten {out}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cluster", action="store_true", help="scan the parquet cluster column")
    run(get_settings(), with_cluster=parser.parse_args().cluster)
