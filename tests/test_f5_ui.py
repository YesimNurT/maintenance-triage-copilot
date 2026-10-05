"""API client and Streamlit page against the in-process API. No network."""

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("fastapi")
pytest.importorskip("streamlit")

from pathlib import Path  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from mtc.api import client as client_module  # noqa: E402
from mtc.api.app import create_app  # noqa: E402
from mtc.api.client import ApiClient  # noqa: E402
from tests.test_f5_service_api import service  # noqa: E402, F401

APP = Path(__file__).parents[1] / "app" / "streamlit_app.py"


def test_client_round_trip(service):  # noqa: F811
    client = ApiClient("http://testserver/", session=TestClient(create_app(service)))

    assert client.base_url == "http://testserver"
    assert client.ranking(2)[0]["rank"] == 1
    assert client.note(41)["note"]["status"] == "note"
    assert client.review(41, "approve", "ok")["added_as_case"] is True


def test_streamlit_page_shows_ranking_note_and_stores_a_review(service, monkeypatch):  # noqa: F811
    session = TestClient(create_app(service))
    original = ApiClient.__init__
    monkeypatch.setattr(
        client_module.ApiClient, "__init__",
        lambda self, base_url, _=None: original(self, base_url, session),
    )

    page = AppTest.from_file(str(APP)).run(timeout=30)

    assert not page.exception
    assert page.title[0].value == "Maintenance Triage Copilot"
    assert len(page.dataframe[0].value) == 4
    assert page.success[0].value.startswith("Inspection note")

    page.button[0].click().run(timeout=30)

    assert "Review stored: approve and added as a new case." in page.info[0].value
    assert service.reviews_path.exists()
