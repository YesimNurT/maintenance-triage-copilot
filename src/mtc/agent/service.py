"""Application service shared by the API, the UI and the evaluation.

Holds the health-score table, the case index and the compiled agent, and answers the
three questions of the product: what to look at first, on what evidence, and what the
technician decided.
"""

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mtc.agent.evidence import build_evidence
from mtc.agent.graph import build_graph, run_agent
from mtc.agent.nodes import NoteWriter, template_writer
from mtc.agent.schemas import Evidence
from mtc.config import Settings
from mtc.models.health_score import symptom_profile
from mtc.retrieval.index import CaseIndex, LocalIndex, make_index
from mtc.retrieval.search import index_cases
from mtc.retrieval.signatures import SIGNATURE_COLUMNS, case_id, signature_matrix

REVIEW_DECISIONS = {"approve", "edit", "reject"}


@dataclass
class TriageService:
    settings: Settings
    scores: pd.DataFrame
    index: CaseIndex
    healthy_scores: np.ndarray
    writer: NoteWriter = template_writer
    graph: Any = field(init=False)
    _shown: dict[int, dict[str, Any]] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.graph = build_graph(self.settings, lambda evidence: self.writer(evidence))

    @classmethod
    def from_processed(cls, settings: Settings, writer: NoteWriter = template_writer,
                       local_index: bool = False) -> "TriageService":
        """Load the tables written by the F3 and F4 scripts.

        Without a Pinecone key (or with ``local_index``) the cases are indexed in memory;
        with a key the Pinecone index is expected to be filled already.
        """
        scores = pd.read_parquet(settings.processed_dir / "health_scores.parquet")
        if local_index or not settings.pinecone_api_key:
            cases = pd.read_parquet(settings.processed_dir / "cases.parquet")
            index: CaseIndex = index_cases(LocalIndex(), cases)
        else:
            index = make_index(settings, len(SIGNATURE_COLUMNS))
        healthy = scores[(scores["split"] == "train") & (scores["before_after"] == "after")]
        return cls(settings, scores, index, healthy["score_all"].dropna().to_numpy(), writer)

    def _demo_flights(self) -> pd.DataFrame:
        scores = self.scores
        return scores[(scores["split"] == self.settings.demo_split) & scores["binary_task"]]

    def ranking(self, limit: int) -> list[dict[str, Any]]:
        """Demo flights by descending health score, with their strongest residuals."""
        top = self._demo_flights().sort_values("score_all", ascending=False).head(limit)
        rows = []
        for rank, (flight_id, row) in enumerate(top.iterrows(), start=1):
            z = row[SIGNATURE_COLUMNS].astype(float)
            z.index = [c.removeprefix("z_") for c in z.index]
            rows.append({
                "rank": rank,
                "flight_id": int(flight_id),
                "health_score": round(float(row["score_all"]), 2),
                "symptoms": symptom_profile(z),
            })
        return rows

    def evidence(self, flight_id: int) -> Evidence:
        """Evidence for one demo flight; ``KeyError`` if it is not a demo flight."""
        row = self._demo_flights().loc[flight_id]
        return build_evidence(
            flight_id, row, self.index, self.healthy_scores,
            self.settings.retrieval_k, self.settings.gate_min_similarity,
        )

    def note(self, flight_id: int) -> dict[str, Any]:
        """Run the agent for one flight: evidence, gate decision and final note."""
        state = run_agent(self.graph, self.evidence(flight_id))
        result = {key: state[key].model_dump() for key in ("evidence", "decision", "note")}
        self._shown[int(flight_id)] = result
        return result

    def review(self, flight_id: int, decision: str, comment: str = "") -> dict[str, Any]:
        """Store the technician's decision; an approved note becomes a new case."""
        if decision not in REVIEW_DECISIONS:
            raise ValueError(f"decision must be one of {sorted(REVIEW_DECISIONS)}")
        # review the note that was shown; an LLM would not write the same one twice
        result = self._shown.get(int(flight_id)) or self.note(flight_id)
        record = {
            "flight_id": int(flight_id),
            "decision": decision,
            "comment": comment,
            "note": result["note"],
            "reviewed_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "added_as_case": False,
        }
        if decision == "approve" and result["note"]["status"] == "note":
            label = result["evidence"]["votes"][0]["label"]
            row = self._demo_flights().loc[[flight_id]]
            self.index.upsert([case_id(flight_id)], signature_matrix(row), [{"label": label}])
            record["added_as_case"] = True
        path = self.reviews_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        return record

    @property
    def reviews_path(self) -> Path:
        return self.settings.results_dir / "reviews.jsonl"
