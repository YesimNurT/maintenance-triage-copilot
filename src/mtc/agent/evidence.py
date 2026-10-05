"""Build the structured evidence for one flight from residuals and retrieved cases."""

import numpy as np
import pandas as pd

from mtc.agent.schemas import CaseEvidence, Evidence, ResidualEvidence, Vote
from mtc.models.health_score import symptom_profile
from mtc.retrieval.index import CaseIndex
from mtc.retrieval.search import vote
from mtc.retrieval.signatures import SIGNATURE_COLUMNS, signature_matrix

EVIDENCE_Z = 2.0  # residuals below this are not listed at all
MAX_RESIDUALS = 3


def build_evidence(
    flight_id: int,
    row: pd.Series,
    index: CaseIndex,
    healthy_scores: np.ndarray,
    k: int,
    min_similarity: float,
) -> Evidence:
    """Evidence for one row of the health-score table.

    Only cases at least ``min_similarity`` close are kept, so the votes describe the
    cases that actually resemble this flight.
    """
    z = row[SIGNATURE_COLUMNS].astype(float)
    z.index = [c.removeprefix("z_") for c in z.index]
    profile = symptom_profile(z, EVIDENCE_Z, MAX_RESIDUALS)
    residuals = [ResidualEvidence(id=f"R{i}", **item) for i, item in enumerate(profile, start=1)]

    vector = signature_matrix(row.to_frame().T)[0]
    hits = [h for h in index.query(vector, k) if h.similarity >= min_similarity]
    cases = [
        CaseEvidence(id=f"C{i}", case_id=h.case_id, label=h.label,
                     similarity=round(h.similarity, 2))
        for i, h in enumerate(hits, start=1)
    ]
    score = float(row["score_all"]) if pd.notna(row["score_all"]) else 0.0
    percentile = 100.0 * float(np.mean(np.asarray(healthy_scores) <= score))
    return Evidence(
        flight_id=int(flight_id),
        health_score=round(score, 2),
        score_percentile=round(percentile, 1),
        residuals=residuals,
        cases=cases,
        votes=[Vote(**v) for v in vote(hits)],
    )
