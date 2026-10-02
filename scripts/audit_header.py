"""Profile the NGAFID header tables and write results/f1/header_audit.json.

    uv run python scripts/audit_header.py
"""

import json
from pathlib import Path

import pandas as pd

from mtc.config import Settings, get_settings
from mtc.data.header import find_tables, parquet_schema, profile_table


def run(settings: Settings) -> Path:
    raw = settings.ngafid_raw_dir
    report: dict = {"tables": {}, "parquet": {}}

    for path in find_tables(raw):
        df = pd.read_csv(path)
        profile = profile_table(df)
        report["tables"][str(path.relative_to(raw))] = profile
        print(f"\n== {path.relative_to(raw)}: {profile['n_rows']} rows, "
              f"{profile['n_duplicate_rows']} duplicate rows")
        print(df.head(3).to_string())
        for name, info in profile["columns"].items():
            detail = info.get("value_counts") or f"{info.get('min')} .. {info.get('max')}"
            print(f"- {name} [{info['dtype']}] unique={info['n_unique']} "
                  f"missing={info['n_missing']}: {detail}")

    for directory in sorted({p.parent for p in raw.rglob("*.parquet")}):
        schema = parquet_schema(directory)
        report["parquet"][str(directory.relative_to(raw))] = schema
        print(f"\n== {directory.relative_to(raw)}: {schema['n_files']} files, "
              f"{schema['size_gb']} GB")
        for name, dtype in schema["columns"].items():
            print(f"- {name} [{dtype}]")

    out = settings.results_dir / "f1" / "header_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwritten {out}")
    return out


if __name__ == "__main__":
    run(get_settings())
