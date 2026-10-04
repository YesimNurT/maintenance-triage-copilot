"""Find the NGAFID files under any folder and expose them in the layout the code expects.

Zenodo and Kaggle ship the same files under different folder names (or as archives).
``link_raw_layout`` builds ``<target>/all_flights/{flight_header.csv, one_parq}`` and
``<target>/2days/flight_header.csv`` as symlinks, so nothing is copied.
"""

import tarfile
from pathlib import Path

import pandas as pd

HEADER_NAME = "flight_header.csv"


def extract_archives(root: Path, dest: Path) -> list[Path]:
    """Extract every ``*.tar.gz`` under ``root`` into ``dest``; returns the archives found."""
    archives = sorted(root.rglob("*.tar.gz"))
    for archive in archives:
        with tarfile.open(archive) as tar:
            tar.extractall(dest, filter="data")
    return archives


def find_raw_parts(root: Path) -> dict[str, Path]:
    """Locate the full header, the benchmark header (has a ``fold`` column) and one_parq."""
    parts: dict[str, Path] = {}
    for header in sorted(root.rglob(HEADER_NAME)):
        columns = pd.read_csv(header, nrows=0).columns
        parts["benchmark_header" if "fold" in columns else "all_header"] = header
    for directory in sorted(p for p in root.rglob("one_parq") if p.is_dir()):
        if any(directory.glob("*.parquet")):
            parts["one_parq"] = directory
    missing = {"all_header", "benchmark_header", "one_parq"} - set(parts)
    if missing:
        raise FileNotFoundError(f"not found under {root}: {sorted(missing)}")
    return parts


def link_raw_layout(root: Path, target: Path) -> Path:
    """Symlink the files found under ``root`` into the expected layout below ``target``."""
    parts = find_raw_parts(root)
    links = {
        target / "all_flights" / HEADER_NAME: parts["all_header"],
        target / "all_flights" / "one_parq": parts["one_parq"],
        target / "2days" / HEADER_NAME: parts["benchmark_header"],
    }
    for link, source in links.items():
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink():
            link.unlink()
        if not link.exists():
            link.symlink_to(source.resolve(), target_is_directory=source.is_dir())
    return target
