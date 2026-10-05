"""Agent gate, citation check and graph. No real LLM: writers are plain functions."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("langgraph")

from mtc.agent.evidence import build_evidence  # noqa: E402
from mtc.agent.graph import build_graph, run_agent  # noqa: E402
from mtc.agent.nodes import (  # noqa: E402
    citation_problems,
    confidence,
    evidence_gate,
    finalise,
    template_writer,
)
from mtc.agent.schemas import (  # noqa: E402
    CaseEvidence,
    Claim,
    Evidence,
    InspectionNote,
    ResidualEvidence,
    Vote,
)
from mtc.config import Settings  # noqa: E402
from mtc.retrieval.index import LocalIndex  # noqa: E402
from mtc.retrieval.signatures import SIGNATURE_COLUMNS  # noqa: E402

SETTINGS = Settings()


def _evidence(z: float = -4.2, n_cases: int = 4, agreeing: int = 3) -> Evidence:
    """``agreeing`` cases share one label; every other case has a label of its own."""
    labels = ["oil cooler"] * agreeing + [f"other {i}" for i in range(n_cases - agreeing)]
    cases = [
        CaseEvidence(id=f"C{i}", case_id=f"C{100 + i}", label=label, similarity=0.8)
        for i, label in enumerate(labels, start=1)
    ]
    counts = pd.Series(labels).value_counts(sort=False).sort_values(ascending=False, kind="stable")
    votes = [Vote(label=k, cases=int(v), share=v / n_cases) for k, v in counts.items()]
    residuals = [ResidualEvidence(id="R1", channel="E1 OilP", z=z, direction="low")]
    return Evidence(flight_id=7, health_score=1.9, score_percentile=97.0,
                    residuals=residuals, cases=cases, votes=votes)


def test_gate_passes_strong_residual_with_agreeing_cases():
    assert evidence_gate(_evidence(), SETTINGS).passed


@pytest.mark.parametrize(
    ("evidence", "reason"),
    [
        (_evidence(z=-1.5), "residuals within normal range"),
        (_evidence(n_cases=2, agreeing=2), "too few similar cases"),
        (_evidence(n_cases=5, agreeing=2), "similar cases disagree"),
    ],
)
def test_gate_stops_weak_evidence_with_a_reason(evidence, reason):
    decision = evidence_gate(evidence, SETTINGS)

    assert not decision.passed
    assert any(reason in r for r in decision.reasons)


def test_template_note_is_fully_cited():
    evidence = _evidence()

    note = template_writer(evidence)

    assert citation_problems(note, evidence) == []
    assert note.checks[0].citations == ["C1", "C2", "C3"]
    assert confidence(evidence) == "medium"
    assert finalise(note, evidence).confidence == "medium"


@pytest.mark.parametrize(
    ("claim", "field", "problem"),
    [
        (Claim(text="Oil pressure is low."), "symptoms", "uncited claim"),
        (Claim(text="Oil pressure is low.", citations=["R9"]), "symptoms", "unknown citation"),
        (Claim(text="Oil pressure is low.", citations=["C1"]), "symptoms",
         "symptom without a residual citation"),
        (Claim(text="Replace the oil pump.", citations=["R1"]), "checks",
         "check without a case citation"),
        (Claim(text="Oil pressure dropped by 12.5 psi.", citations=["R1"]), "symptoms",
         "number not in evidence"),
    ],
)
def test_unsupported_claims_are_detected(claim, field, problem):
    evidence = _evidence()
    note = template_writer(evidence)
    setattr(note, field, [*getattr(note, field), claim])

    assert any(problem in p for p in citation_problems(note, evidence))
    assert finalise(note, evidence).status == "inconclusive"


def test_numbers_from_the_evidence_and_channel_names_are_allowed():
    evidence = _evidence()
    note = template_writer(evidence)
    note.symptoms.append(
        Claim(text="E1 OilP residual is 4.2 below expected; 3 of 4 cases agree (75%).",
              citations=["R1", "C1"])
    )

    assert citation_problems(note, evidence) == []


def test_graph_returns_inconclusive_without_calling_the_writer():
    calls = []

    def writer(evidence):
        calls.append(evidence.flight_id)
        return template_writer(evidence)

    state = run_agent(build_graph(SETTINGS, writer), _evidence(z=-1.0))

    assert calls == []
    assert state["note"].status == "inconclusive"
    assert "residuals within normal range" in state["note"].reason
    assert state["note"].checks == []


def test_graph_writes_and_stamps_confidence_when_evidence_holds():
    state = run_agent(build_graph(SETTINGS), _evidence(n_cases=8, agreeing=7))

    assert state["decision"].passed
    assert state["note"].status == "note"
    assert state["note"].confidence == "high"


def test_graph_rejects_a_writer_that_invents_content():
    def careless_writer(evidence):
        return InspectionNote(
            status="note",
            symptoms=[Claim(text="Cylinder 3 is running hot.", citations=["R1"])],
            checks=[Claim(text="Replace the injector nozzle.")],
        )

    state = run_agent(build_graph(SETTINGS, careless_writer), _evidence())

    assert state["note"].status == "inconclusive"
    assert "note rejected" in state["note"].reason
    assert state["note"].checks == []


def test_build_evidence_from_scores_and_index():
    row = pd.Series(0.0, index=SIGNATURE_COLUMNS)
    row["z_E1 OilP"], row["z_E1 CHT1"] = -4.5, 2.4
    row["score_all"] = 1.6
    index = LocalIndex()
    near = np.zeros((3, len(SIGNATURE_COLUMNS)), dtype=np.float32)
    near[:, SIGNATURE_COLUMNS.index("z_E1 OilP")] = -3.0
    far = -near[:1]
    index.upsert(["C11", "C12", "C13", "C14"], np.vstack([near, far]),
                 [{"label": "oil cooler"}] * 3 + [{"label": "baffle"}])

    evidence = build_evidence(42, row, index, np.array([0.5, 1.0, 1.5, 2.0]), k=4,
                              min_similarity=0.6)

    assert [r.id for r in evidence.residuals] == ["R1", "R2"]
    assert evidence.residuals[0].channel == "E1 OilP"
    assert evidence.residuals[0].direction == "low"
    assert [c.case_id for c in evidence.cases] == ["C11", "C12", "C13"]
    assert evidence.votes[0].label == "oil cooler"
    assert evidence.score_percentile == 75.0
    assert evidence.ids() == {"R1", "R2", "C1", "C2", "C3"}
