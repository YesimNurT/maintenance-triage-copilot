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
