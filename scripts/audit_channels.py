"""Check channel ranges, flight quality and phase shares on a random sample of flights.

    uv run python scripts/audit_channels.py            # 300 flights
    uv run python scripts/audit_channels.py --n 1000
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from mtc.config import Settings, get_settings
from mtc.data.events import FLIGHT_ID
from mtc.data.loading import CONTEXT_CHANNELS, ENGINE_CHANNELS, load_flights
from mtc.data.phases import label_phases
from mtc.data.quality import VALID_RANGES, flight_quality, out_of_range_share

QUANTILES = [0.0, 0.001, 0.01, 0.5, 0.99, 0.999, 1.0]


def run(settings: Settings, n_flights: int = 300) -> Path:
    root = settings.ngafid_raw_dir / "all_flights"
    header = pd.read_csv(root / "flight_header.csv")
    ids = header[FLIGHT_ID].sample(n_flights, random_state=settings.random_seed).tolist()
    channels = ENGINE_CHANNELS + CONTEXT_CHANNELS
    rows = load_flights(root / "one_parq", ids, channels)

    quantiles = rows[channels].quantile(QUANTILES).T
    quantiles.columns = [f"q{q}" for q in QUANTILES]
    outside = out_of_range_share(rows, VALID_RANGES)
    quality = flight_quality(rows, ENGINE_CHANNELS, settings.min_flight_seconds)
    phases = pd.concat(
        [label_phases(f.reset_index(drop=True)) for _, f in rows.groupby(FLIGHT_ID)],
        ignore_index=True,
    )

    print(f"== {len(ids)} flights, {len(rows)} rows")
    print("\n== channel quantiles (raw) and share outside VALID_RANGES")
    print(quantiles.assign(outside=pd.Series(outside)).round(4).to_string())
    print(f"\n== flight quality: ok {int(quality['ok'].sum())} / {len(quality)}")
    print(quality[["n_seconds", "missing_share"]].describe().round(3).to_string())
    print("\n== phase share over all rows")
    print(phases.value_counts(normalize=True).round(3).to_string())

    report = {
        "n_flights": len(ids),
        "quantiles": quantiles.to_dict("index"),
        "outside_share": outside,
        "quality_ok": int(quality["ok"].sum()),
        "phase_share": phases.value_counts(normalize=True).to_dict(),
    }
    out = settings.results_dir / "f1" / "channel_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=float))
    print(f"\nwritten {out}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=300, help="number of flights to sample")
    run(get_settings(), parser.parse_args().n)
