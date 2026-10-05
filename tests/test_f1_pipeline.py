"""End-to-end run of the F1 and F2.1 scripts on a synthetic raw dataset."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mtc.config import Settings
from tests.test_features_signal import synthetic_flight

SCRIPTS = Path(__file__).parents[1] / "scripts"
LABELS = ["gasket", "baffle", "rocker", "intake"]


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _raw_dataset(raw: Path, n: int = 400) -> None:
    ids = np.arange(1, n + 1)
    header = pd.DataFrame(
        {
            "Master Index": ids,
            "before_after": np.where(ids % 2 == 0, "before", "after"),
            "date_diff": np.where(ids % 2 == 0, -1, 1),
            "flight_length": 300.0,
            "label": np.array(LABELS)[(ids - 1) * len(LABELS) // n],
            "hierarchy": None,
            "number_flights_before": np.where(ids % 2 == 0, 0, -1),
        }
    )
    (raw / "all_flights" / "one_parq").mkdir(parents=True)
    (raw / "2days").mkdir()
    header.to_csv(raw / "all_flights" / "flight_header.csv", index=False)
    bench = header.iloc[:50].copy()
    bench["before_after"] = (bench["before_after"] == "before").astype(int)
    bench["fold"] = np.arange(50) % 5
    bench.to_csv(raw / "2days" / "flight_header.csv", index=False)
    rng = np.random.default_rng(0)
    flights = [
        synthetic_flight(i, egt3_offset=(30.0 if i % 2 == 0 else 0.0) + rng.normal(0, 10))
        for i in ids
    ]
    # like the real files: the flight id is stored as the pandas index
    pd.concat(flights).set_index("Master Index").to_parquet(
        raw / "all_flights" / "one_parq" / "p0.parquet"
    )


def test_f1_scripts_run_end_to_end(tmp_path: Path, capsys):
    raw = tmp_path / "raw"
    _raw_dataset(raw)
    settings = Settings(
        ngafid_raw_dir=raw,
        data_dir=tmp_path / "data",
        results_dir=tmp_path / "results",
        min_flight_seconds=100,
        mvp_classes=LABELS,
    )

    _script("make_splits").run(settings)
    _script("audit_channels").run(settings, n_flights=20)
    sample_dir = _script("make_sample").run(settings)
    _script("build_features").run(settings)
    out = _script("gate1_signal").run(settings)
    features = _script("build_features").run(settings, all_flights=True)
    trend = _script("f2_oil_hypothesis").run(settings)
    baselines = _script("f2_tabular_baselines").run(settings)
    issue = json.loads(_script("f2_issue_class").run(settings).read_text())
    health = json.loads(_script("f3_health_score").run(settings).read_text())
    retrieval = json.loads(_script("f4_case_retrieval").run(settings).read_text())

    assert len(pd.read_csv(sample_dir / "header.csv")) == 12
    report = json.loads(out.read_text())
    assert report["n_train"] > report["n_val"] > 0
    assert report["engine"]["auc"] > 0.9
    assert report["gate1_passed"] is True
    assert set(report["engine_auc_by_label"]) <= set(LABELS)
    assert report["auc_by_feature_group"]["cylinder_relative"]["auc"] > 0.9
    assert len(pd.read_parquet(features)) == 400
    assert json.loads(trend.read_text())["n_flights"] > 0
    f2 = json.loads(baselines.read_text())
    assert f2["metrics"]["without_oil"]["gbm"]["auc"] > 0.9
    assert f2["cht_egt_signal_present"] is True
    assert issue["before"]["all"]["n_val"] > 0
    assert set(issue["decision"]) == {"model", "class_signal_present", "symptom_specific"}
    assert set(health["scores"]) == {"all", "without_oil"}
    assert "cruise_OAT_mean" in health["conditions"]
    assert (settings.processed_dir / "health_scores.parquet").exists()
    assert retrieval["before"]["n_queries"] > 0
    assert (settings.processed_dir / "cases.parquet").exists()
    assert "phase share" in capsys.readouterr().out
