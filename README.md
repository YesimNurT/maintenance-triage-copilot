# Maintenance Triage Copilot

Which aircraft should be inspected today, and what should the technician check first?

Work in progress. See `docs/BRIEF.md` for the goal, architecture and evaluation plan.

## Setup
```bash
cp .env.example .env        # fill in keys
uv sync --group dev          # core + dev tools
uv sync --group dev --extra dl --extra agent --extra api --extra app   # everything
uv run pytest
```

## Data
```bash
uv run python scripts/download_ngafid.py --list   # files in the Zenodo record
uv run python scripts/download_ngafid.py          # download into data/raw/ngafid (5.4 GB)
```

## F1 pipeline (run in this order)
```bash
uv run python scripts/make_splits.py      # folds and splits -> data/processed/splits.parquet
uv run python scripts/audit_channels.py   # channel ranges, quality, phases on 300 flights
uv run python scripts/make_sample.py      # small extract -> data/sample/
uv run python scripts/build_features.py   # per-flight features (long)
uv run python scripts/gate1_signal.py     # Gate 1 on the val split
```

Data: NGAFID maintenance dataset, Yang et al. 2022, Zenodo 6624956, CC BY 4.0.
