"""Split rules on small hand-made headers; no raw data is touched."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from mtc.config import Settings
from mtc.data.splits import (
    assign_folds,
    binary_task_mask,
    check_benchmark_subset,
    split_from_fold,
)

SCRIPT = Path(__file__).parents[1] / "scripts" / "make_splits.py"


def _header(n: int = 200) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Master Index": range(1, n + 1),
            "before_after": ["before", "after"] * (n // 2),
            "date_diff": [-1, 1] * (n // 2),
            "label": ["gasket"] * (n // 2) + ["baffle"] * (n // 2),
        }
    )


def _benchmark(header: pd.DataFrame) -> pd.DataFrame:
    bench = header.iloc[:10].copy()
    bench["before_after"] = bench["before_after"].map({"before": 1, "after": 0})
    bench["fold"] = [0, 1, 2, 3, 4] * 2
    return bench


def test_benchmark_flights_keep_their_fold():
    header = _header()
    fold = assign_folds(header, _benchmark(header), n_folds=5, seed=0)
    assert fold.iloc[:10].tolist() == [0, 1, 2, 3, 4] * 2


def test_other_flights_are_balanced_and_reproducible():
    header = _header()
    fold = assign_folds(header, _benchmark(header), n_folds=5, seed=0)

    rest = header.iloc[10:].assign(fold=fold.iloc[10:])
    counts = rest.groupby(["label", "before_after"])["fold"].value_counts()
    assert counts.max() - counts.min() <= 1
    assert fold.equals(assign_folds(header, _benchmark(header), n_folds=5, seed=0))
    assert not fold.equals(assign_folds(header, _benchmark(header), n_folds=5, seed=1))


def test_splits_do_not_overlap_and_cover_every_flight():
    header = _header()
    fold = assign_folds(header, _benchmark(header), n_folds=5, seed=0)
    split = split_from_fold(fold, val_fold=3, test_fold=4)

    ids = {name: set(header.loc[split == name, "Master Index"]) for name in split.unique()}
    assert set(ids) == {"train", "val", "test"}
    assert not ids["train"] & ids["val"]
    assert not ids["train"] & ids["test"]
    assert not ids["val"] & ids["test"]
    assert sum(len(v) for v in ids.values()) == len(header)


def test_split_rejects_same_val_and_test_fold():
    with pytest.raises(ValueError, match="must differ"):
        split_from_fold(pd.Series([0, 1]), val_fold=1, test_fold=1)


def test_check_benchmark_subset_detects_mismatch():
    header = _header()
    check_benchmark_subset(header, _benchmark(header))

    wrong_label = _benchmark(header).assign(label="other")
    with pytest.raises(ValueError, match="labels differ"):
        check_benchmark_subset(header, wrong_label)
    with pytest.raises(ValueError, match="not in the full header"):
        check_benchmark_subset(header.iloc[5:], _benchmark(header))


def test_binary_task_mask_uses_window_and_excludes_day_zero():
    header = pd.DataFrame(
        {
            "before_after": ["before", "before", "same", "before", "after", "after"],
            "date_diff": [-1, -3, 0, 0, 2, 3],
        }
    )
    assert binary_task_mask(header, window_days=2).tolist() == [
        True, False, False, False, True, False,
    ]


def test_make_splits_script(tmp_path: Path):
    raw = tmp_path / "raw"
    (raw / "all_flights").mkdir(parents=True)
    (raw / "2days").mkdir()
    header = _header()
    header.to_csv(raw / "all_flights" / "flight_header.csv", index=False)
    _benchmark(header).to_csv(raw / "2days" / "flight_header.csv", index=False)
    spec = importlib.util.spec_from_file_location("make_splits", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    settings = Settings(
        ngafid_raw_dir=raw, data_dir=tmp_path / "data", results_dir=tmp_path / "results"
    )

    out = module.run(settings)

    splits = pd.read_parquet(out)
    assert len(splits) == 200
    assert splits["in_benchmark"].sum() == 10
    assert splits["binary_task"].all()
