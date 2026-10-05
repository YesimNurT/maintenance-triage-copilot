"""FastAPI service: ranking, evidence, inspection note and technician review.

    uv run uvicorn mtc.api.app:app --reload
"""

from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from mtc.agent.service import TriageService
from mtc.config import get_settings


class ReviewRequest(BaseModel):
    decision: Literal["approve", "edit", "reject"]
    comment: str = ""


def create_app(service: TriageService | None = None) -> FastAPI:
    """Build the app. Without a service it is created from the processed tables on first use."""
    api = FastAPI(title="Maintenance Triage Copilot", version="0.1.0")
    state: dict[str, TriageService | None] = {"service": service}

    def get_service() -> TriageService:
        if state["service"] is None:
            try:
                state["service"] = TriageService.from_processed(get_settings())
            except FileNotFoundError as error:
                raise HTTPException(503, f"processed tables missing: {error.filename}") from error
        return state["service"]

    def demo_flight(call, flight_id: int, *args) -> Any:
        try:
            return call(flight_id, *args)
        except KeyError as error:
            raise HTTPException(404, f"flight {flight_id} is not a demo flight") from error

    @api.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @api.get("/ranking")
    def ranking(limit: int = Query(20, ge=1, le=200)) -> list[dict[str, Any]]:
        return get_service().ranking(limit)

    @api.get("/flights/{flight_id}/evidence")
    def evidence(flight_id: int) -> dict[str, Any]:
        return demo_flight(get_service().evidence, flight_id).model_dump()

    @api.post("/flights/{flight_id}/note")
    def note(flight_id: int) -> dict[str, Any]:
        return demo_flight(get_service().note, flight_id)

    @api.post("/flights/{flight_id}/review")
    def review(flight_id: int, request: ReviewRequest) -> dict[str, Any]:
        return demo_flight(get_service().review, flight_id, request.decision, request.comment)

    return api


app = create_app()
