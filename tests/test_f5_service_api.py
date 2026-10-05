"""Service, evaluation and API on a small synthetic score table. No LLM, no Pinecone."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("langgraph")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from mtc.agent.evaluate import evaluate_agent  # noqa: E402
from mtc.agent.service import TriageService  # noqa: E402
from mtc.api.app import create_app  # noqa: E402
from mtc.config import Settings  # noqa: E402
from mtc.retrieval.index import LocalIndex  # noqa: E402
from mtc.retrieval.search import index_cases  # noqa: E402
from mtc.retrieval.signatures import SIGNATURE_COLUMNS, build_cases  # noqa: E402

OIL = "z_E1 OilP"


def _scores(seed: int = 0) -> pd.DataFrame:
    """Train cases with a clear low-oil signature; val has two such flights and two quiet ones."""
    rng = np.random.default_rng(seed)
    n = 44
    table = pd.DataFrame(rng.normal(0, 0.3, (n, len(SIGNATURE_COLUMNS))), columns=SIGNATURE_COLUMNS,
                         index=pd.RangeIndex(1, n + 1, name="Master Index"))
    table["split"] = ["train"] * 40 + ["val"] * 4
    val_phases = ["before", "before", "after", "after"]
    table["before_after"] = ["before"] * 30 + ["after"] * 10 + val_phases
    table["label"] = "oil cooler"
    table["binary_task"] = True
    table["date_diff"] = -1
    strong = list(range(1, 31)) + [41, 43]
    table.loc[strong, OIL] = -5.0
    table["score_all"] = np.sqrt((table[SIGNATURE_COLUMNS] ** 2).mean(axis=1))
    return table


@pytest.fixture
def service(tmp_path: Path) -> TriageService:
    scores = _scores()
    index = index_cases(LocalIndex(), build_cases(scores))
    healthy = scores.loc[31:40, "score_all"].to_numpy()
    return TriageService(Settings(results_dir=tmp_path), scores, index, healthy)


def test_ranking_puts_strong_deviations_first(service: TriageService):
    ranking = service.ranking(limit=3)

    assert {r["flight_id"] for r in ranking[:2]} == {41, 43}
    assert ranking[0]["rank"] == 1
    assert ranking[0]["symptoms"][0] == {"channel": "E1 OilP", "z": -5.0, "direction": "low"}


def test_note_for_strong_and_quiet_flights(service: TriageService):
    strong, quiet = service.note(41), service.note(42)

    assert strong["decision"]["passed"]
    assert strong["note"]["status"] == "note"
    assert strong["note"]["checks"][0]["citations"]
    assert quiet["note"]["status"] == "inconclusive"
    assert quiet["note"]["checks"] == []


def test_review_is_logged_and_approved_note_becomes_a_case(service: TriageService):
    before = len(service.index.ids)

    approved = service.review(41, "approve", "confirmed on inspection")
    rejected = service.review(42, "reject")

    assert approved["added_as_case"] and not rejected["added_as_case"]
    assert len(service.index.ids) == before + 1
    assert service.index.ids[-1] == "C41"
    lines = service.reviews_path.read_text().splitlines()
    assert [json.loads(line)["decision"] for line in lines] == ["approve", "reject"]
    with pytest.raises(ValueError, match="decision must be one of"):
        service.review(41, "maybe")


def test_review_stores_the_note_the_technician_saw(service: TriageService):
    shown = service.note(41)
    service.writer = None  # a second writer call would fail: the review must reuse the note

    record = service.review(41, "reject")

    assert record["note"] == shown["note"]


def test_evaluate_agent_reports_rates_per_phase(service: TriageService):
    flights = service.scores[service.scores["split"] == "val"]

    report = evaluate_agent(service, flights)

    assert report["n_flights"] == 4
    assert report["before"]["note_rate"] == 0.5
    assert report["after"]["inconclusive_rate"] == 0.5
    assert report["before"]["citation_validity"] == 1.0
    assert report["before"]["label_match_rate"] == 1.0
    assert report["after"]["stop_reasons"]["residuals within normal range"] == 1


def test_api_endpoints(service: TriageService):
    client = TestClient(create_app(service))

    assert client.get("/health").json() == {"status": "ok"}
    assert len(client.get("/ranking", params={"limit": 2}).json()) == 2
    assert client.get("/flights/41/evidence").json()["residuals"][0]["id"] == "R1"
    assert client.post("/flights/41/note").json()["note"]["status"] == "note"
    review = client.post("/flights/41/review", json={"decision": "approve"})
    assert review.json()["added_as_case"] is True
    assert client.get("/flights/9999/evidence").status_code == 404
    assert client.post("/flights/41/review", json={"decision": "maybe"}).status_code == 422
    assert client.get("/ranking", params={"limit": 0}).status_code == 422


def test_from_processed_loads_tables_and_uses_local_index(tmp_path: Path):
    settings = Settings(data_dir=tmp_path, results_dir=tmp_path, pinecone_api_key=None)
    settings.processed_dir.mkdir(parents=True)
    scores = _scores()
    scores.to_parquet(settings.processed_dir / "health_scores.parquet")
    build_cases(scores).to_parquet(settings.processed_dir / "cases.parquet")

    service = TriageService.from_processed(settings)

    assert isinstance(service.index, LocalIndex)
    assert len(service.index.ids) == 30
    assert len(service.healthy_scores) == 10
    assert service.note(41)["note"]["status"] == "note"
