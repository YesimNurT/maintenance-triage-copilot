"""Profile the NGAFID header tables without assuming their column names.

Used once in F1 to learn what the header tables contain (ids, labels, dates) before any
split or class decision is made. Only the small csv tables are read; for the 1 Hz sensor
data only the parquet schema is inspected, never the rows.
"""

from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


def find_tables(root: Path) -> list[Path]:
    """All csv files under ``root``, sorted so the report order is stable."""
    return sorted(root.rglob("*.csv"))


def profile_table(df: pd.DataFrame, max_categories: int = 40) -> dict[str, Any]:
    """Per-column dtype, missing count, unique count and range or value counts.

    Columns with at most ``max_categories`` distinct values get their full value counts
    (labels, classes); the others get min / max (ids, dates, lengths).
    """
    columns: dict[str, Any] = {}
    for name in df.columns:
        series = df[name]
        info: dict[str, Any] = {
            "dtype": str(series.dtype),
            "n_missing": int(series.isna().sum()),
            "n_unique": int(series.nunique(dropna=True)),
        }
        present = series.dropna()
        if info["n_unique"] <= max_categories:
            counts = present.value_counts()
            info["value_counts"] = {str(k): int(v) for k, v in counts.items()}
        elif not present.empty:
            info["min"] = str(present.min())
            info["max"] = str(present.max())
        columns[str(name)] = info
    return {
        "n_rows": int(len(df)),
        "n_duplicate_rows": int(df.duplicated().sum()),
        "columns": columns,
    }


def parquet_schema(dataset_dir: Path) -> dict[str, Any]:
    """Column names and types of a parquet dataset, read from metadata only."""
    files = sorted(dataset_dir.glob("*.parquet"))
    metadata = dataset_dir / "_common_metadata"
    schema = pq.read_schema(metadata if metadata.exists() else files[0])
    return {
        "n_files": len(files),
        "size_gb": round(sum(f.stat().st_size for f in files) / 1e9, 2),
        "columns": {field.name: str(field.type) for field in schema},
    }
