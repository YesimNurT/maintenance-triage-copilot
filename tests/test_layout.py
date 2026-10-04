"""Raw-data layout discovery on temporary folders."""

import importlib.util
import tarfile
from pathlib import Path

import pandas as pd
import pytest

from mtc.config import Settings
from mtc.data.layout import extract_archives, find_raw_parts, link_raw_layout

SCRIPT = Path(__file__).parents[1] / "scripts" / "kaggle_prepare_raw.py"


def _odd_layout(root: Path) -> None:
    """Same files as NGAFID, under folder names the code does not expect."""
    (root / "ds" / "full" / "one_parq").mkdir(parents=True)
    (root / "ds" / "subset").mkdir()
    pd.DataFrame({"Master Index": [1], "label": ["x"]}).to_csv(
        root / "ds" / "full" / "flight_header.csv", index=False
    )
    pd.DataFrame({"Master Index": [1], "fold": [0]}).to_csv(
        root / "ds" / "subset" / "flight_header.csv", index=False
    )
    pd.DataFrame({"a": [1.0]}).to_parquet(root / "ds" / "full" / "one_parq" / "p0.parquet")


def test_link_raw_layout_builds_expected_tree(tmp_path: Path):
    _odd_layout(tmp_path / "input")

    target = link_raw_layout(tmp_path / "input", tmp_path / "raw")
    link_raw_layout(tmp_path / "input", tmp_path / "raw")  # safe to repeat

    assert "fold" in pd.read_csv(target / "2days" / "flight_header.csv").columns
    assert "label" in pd.read_csv(target / "all_flights" / "flight_header.csv").columns
    assert (target / "all_flights" / "one_parq" / "p0.parquet").exists()


def test_find_raw_parts_reports_what_is_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="one_parq"):
        find_raw_parts(tmp_path)


def test_script_extracts_archives_when_needed(tmp_path: Path):
    _odd_layout(tmp_path / "staging")
    (tmp_path / "input").mkdir()
    with tarfile.open(tmp_path / "input" / "data.tar.gz", "w:gz") as tar:
        tar.add(tmp_path / "staging" / "ds", arcname="ds")
    assert extract_archives(tmp_path / "input", tmp_path / "check") != []
    spec = importlib.util.spec_from_file_location("kaggle_prepare_raw", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    target = module.run(Settings(ngafid_raw_dir=tmp_path / "raw"), tmp_path / "input")

    assert (target / "all_flights" / "one_parq" / "p0.parquet").exists()
    assert (target / "2days" / "flight_header.csv").exists()
