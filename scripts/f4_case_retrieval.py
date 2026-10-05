"""F4: build the case table and evaluate similar-case retrieval on the val split.

    uv run python scripts/f4_case_retrieval.py

Needs f3_health_score. Writes data/processed/cases.parquet (training "before" flights,
every label) and results/f4/retrieval.json (MVP classes, with an after-repair control).
"""

import json
from pathlib import Path

import pandas as pd

from mtc.config import Settings, get_settings
from mtc.retrieval.search import retrieval_accuracy
from mtc.retrieval.signatures import build_cases

K_CASES = 10
N_BOOT = 1000


def run(settings: Settings) -> Path:
    scores = pd.read_parquet(settings.processed_dir / "health_scores.parquet")
    build_cases(scores).to_parquet(settings.processed_dir / "cases.parquet")

    mvp = scores[scores["label"].isin(settings.mvp_classes)]
    report: dict = {"k_cases": K_CASES, "classes": settings.mvp_classes}
    for phase in ("before", "after"):
        cases = build_cases(mvp, phase)
        queries = mvp[
            (mvp["split"] == "val") & mvp["binary_task"] & (mvp["before_after"] == phase)
        ]
        report[phase] = retrieval_accuracy(
            cases, queries, K_CASES, N_BOOT, settings.random_seed
        )
        r = report[phase]
        low, high = r["top1_ci95"]
        print(f"{phase:6s} cases {r['n_cases']}, queries {r['n_queries']}: "
              f"top1 {r['top1_accuracy']:.3f} [{low:.3f}, {high:.3f}] "
              f"(frequency {r['frequency_top1']:.3f})  top3 {r['top3_accuracy']:.3f} "
              f"(frequency {r['frequency_top3']:.3f})  beats baseline "
              f"{r['beats_frequency_baseline']}")

    out = settings.results_dir / "f4" / "retrieval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"written {out}")
    return out


if __name__ == "__main__":
    run(get_settings())
