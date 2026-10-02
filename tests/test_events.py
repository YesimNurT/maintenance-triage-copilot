"""Event reconstruction on small hand-made headers; no raw data is touched."""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from mtc.config import Settings
from mtc.data.events import assign_event_ids, flight_level_values, summarise_events

SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_events.py"


def _header(rows: list[tuple[int, str, str, int]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["Master Index", "before_after", "label", "date_diff"])
    numbers = {"before": 1}
    df["number_flights_before"] = df["before_after"].map(numbers).fillna(-1).astype(int)
    return df


TWO_LABELS = [
    (1, "before", "gasket", -1),
    (2, "same", "gasket", 0),
    (3, "after", "gasket", 1),
    (4, "before", "baffle", -1),
    (5, "after", "baffle", 2),
]


def test_label_change_starts_new_event():
    assert assign_event_ids(_header(TWO_LABELS)).tolist() == [0, 0, 0, 1, 1]


def test_phase_going_backwards_splits_same_label():
    rows = [
        (1, "before", "gasket", -1),
        (2, "after", "gasket", 1),
        (3, "before", "gasket", -2),
        (4, "same", "gasket", 0),
        (5, "before", "gasket", -1),
    ]
    assert assign_event_ids(_header(rows)).tolist() == [0, 0, 1, 1, 2]


def test_event_ids_follow_flight_id_not_row_order():
    header = _header(TWO_LABELS).iloc[::-1]

    event_id = assign_event_ids(header)

    assert event_id.index.equals(header.index)
    assert event_id.tolist() == [1, 1, 0, 0, 0]


def test_unknown_phase_is_rejected():
    with pytest.raises(ValueError, match="unknown before_after"):
        assign_event_ids(_header([(1, "during", "gasket", 0)]))


def test_summarise_events():
    header = _header(TWO_LABELS)

    summary = summarise_events(header, assign_event_ids(header))

    assert summary["n_events"] == 2
    assert summary["flights_per_event"]["max"] == 3.0
    assert summary["phase_patterns"] == {"before+same+after": 1, "before+after": 1}
    assert summary["events_per_label"] == {"gasket": 1, "baffle": 1}
    assert summary["events_with_repeated_flight_number"] == 0


def test_summarise_events_flags_repeated_flight_number():
    rows = [(1, "before", "gasket", -1), (2, "before", "gasket", -1)]
    header = _header(rows)

    summary = summarise_events(header, assign_event_ids(header))

    assert summary["events_with_repeated_flight_number"] == 1


def _write_parquet(directory: Path) -> None:
    directory.mkdir(parents=True)
    rows = pd.DataFrame({"Master Index": [1, 1, 2], "cluster": ["c_1", "c_1", "c_2"]})
    rows.to_parquet(directory / "p0.parquet")
    rows.iloc[:2].to_parquet(directory / "p1.parquet")


def test_flight_level_values_deduplicates_across_files(tmp_path: Path):
    _write_parquet(tmp_path / "one_parq")

    values = flight_level_values(tmp_path / "one_parq", "cluster")

    assert values.to_dict("records") == [
        {"Master Index": 1, "cluster": "c_1"},
        {"Master Index": 2, "cluster": "c_2"},
    ]


def test_flight_level_values_handles_flight_id_as_index(tmp_path: Path):
    rows = pd.DataFrame({"Master Index": [1, 2], "cluster": ["c_1", "c_1"]})
    rows.set_index("Master Index").to_parquet(tmp_path / "p0.parquet")

    values = flight_level_values(tmp_path, "cluster")

    assert values["Master Index"].tolist() == [1, 2]


def test_audit_script_writes_report(tmp_path: Path, capsys):
    root = tmp_path / "raw" / "all_flights"
    _write_parquet(root / "one_parq")
    _header(TWO_LABELS).assign(hierarchy=None).to_csv(root / "flight_header.csv", index=False)
    spec = importlib.util.spec_from_file_location("audit_events", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    settings = Settings(ngafid_raw_dir=tmp_path / "raw", results_dir=tmp_path / "results")

    out = module.run(settings, with_cluster=True)

    report = json.loads(out.read_text())
    assert report["n_events"] == 2
    assert report["cluster"]["n_values"] == 2
    assert "events: 2" in capsys.readouterr().out
