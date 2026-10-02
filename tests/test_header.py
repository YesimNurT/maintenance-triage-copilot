"""Header profiling on small hand-made tables; no raw data is touched."""

import importlib.util
import json
from pathlib import Path

import pandas as pd

from mtc.config import Settings
from mtc.data.header import find_tables, parquet_schema, profile_table

SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_header.py"


def _header() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "flight_id": [1, 2, 3, 4],
            "label": ["a", "a", "b", None],
            "length": [10.0, 20.0, 30.0, 40.0],
        }
    )


def test_find_tables_is_recursive_and_sorted(tmp_path: Path):
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "x.csv").write_text("a\n1\n")
    (tmp_path / "a.csv").write_text("a\n1\n")
    (tmp_path / "notes.txt").write_text("skip")

    assert find_tables(tmp_path) == [tmp_path / "a.csv", tmp_path / "b" / "x.csv"]


def test_profile_table_counts_and_ranges():
    profile = profile_table(_header(), max_categories=3)

    assert profile["n_rows"] == 4
    assert profile["n_duplicate_rows"] == 0
    label = profile["columns"]["label"]
    assert label["n_missing"] == 1
    assert label["value_counts"] == {"a": 2, "b": 1}
    flight_id = profile["columns"]["flight_id"]
    assert flight_id["n_unique"] == 4
    assert (flight_id["min"], flight_id["max"]) == ("1", "4")
    assert "value_counts" not in flight_id


def test_parquet_schema_reads_names_and_types(tmp_path: Path):
    pd.DataFrame({"id": [1, 2], "E1 RPM": [2300.0, 2310.0]}).to_parquet(tmp_path / "p0.parquet")

    schema = parquet_schema(tmp_path)

    assert schema["n_files"] == 1
    assert schema["columns"] == {"id": "int64", "E1 RPM": "double"}


def test_audit_script_writes_report(tmp_path: Path, capsys):
    raw = tmp_path / "raw"
    (raw / "one_parq").mkdir(parents=True)
    _header().to_csv(raw / "flight_header.csv", index=False)
    pd.DataFrame({"id": [1]}).to_parquet(raw / "one_parq" / "p0.parquet")
    spec = importlib.util.spec_from_file_location("audit_header", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    out = module.run(Settings(ngafid_raw_dir=raw, results_dir=tmp_path / "results"))

    report = json.loads(out.read_text())
    assert report["tables"]["flight_header.csv"]["n_rows"] == 4
    assert report["parquet"]["one_parq"]["columns"] == {"id": "int64"}
    assert "flight_header.csv: 4 rows" in capsys.readouterr().out
