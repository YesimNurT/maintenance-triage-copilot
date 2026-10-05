"""Data contracts of the agent: the evidence it receives and the note it may return.

The LLM only ever sees an ``Evidence`` object (scores, residual summaries, case ids),
never raw time series. Every claim in a note carries the ids of the evidence it rests on.
"""

from typing import Literal

from pydantic import BaseModel, Field

INCONCLUSIVE_TEXT = "Inconclusive: evidence is too weak for a recommendation. Continue monitoring."


class ResidualEvidence(BaseModel):
    id: str  # "R1", "R2", ...
    channel: str
    z: float
    direction: Literal["high", "low"]


class CaseEvidence(BaseModel):
    id: str  # "C1", "C2", ...
    case_id: str
    label: str
    similarity: float


class Vote(BaseModel):
    label: str
    cases: int
    share: float


class Evidence(BaseModel):
    flight_id: int
    health_score: float
    score_percentile: float  # among healthy training flights, 0-100
    residuals: list[ResidualEvidence]
    cases: list[CaseEvidence]
    votes: list[Vote]

    def ids(self) -> set[str]:
        return {r.id for r in self.residuals} | {c.id for c in self.cases}


class Claim(BaseModel):
    text: str
    citations: list[str] = Field(default_factory=list)


class InspectionNote(BaseModel):
    status: Literal["note", "inconclusive"]
    symptoms: list[Claim] = Field(default_factory=list)
    similar_cases: list[Claim] = Field(default_factory=list)
    checks: list[Claim] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"] | None = None
    reason: str | None = None

    def claims(self) -> list[Claim]:
        return [*self.symptoms, *self.similar_cases, *self.checks]


class GateDecision(BaseModel):
    passed: bool
    reasons: list[str]


def inconclusive(reason: str) -> InspectionNote:
    """The only note the system gives when evidence is weak or a draft is rejected."""
    return InspectionNote(status="inconclusive", reason=f"{INCONCLUSIVE_TEXT} ({reason})")
