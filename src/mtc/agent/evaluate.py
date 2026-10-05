"""Reliability metrics of the agent on labelled flights (F5 / F6)."""

from collections import Counter
from typing import Any

import pandas as pd

from mtc.agent.nodes import citation_problems
from mtc.agent.schemas import Evidence, InspectionNote
from mtc.agent.service import TriageService


def evaluate_agent(service: TriageService, flights: pd.DataFrame) -> dict[str, Any]:
    """Run the agent on ``flights`` (rows of the score table) and summarise the outcomes.

    Reports how often a note is produced for flights before and after maintenance, why
    the others stop, whether produced notes are fully cited, and how often the suggested
    issue matches the logged one.
    """
    rows = []
    for flight_id, row in flights.iterrows():
        result = service.note(flight_id)
        note = InspectionNote(**result["note"])
        evidence = Evidence(**result["evidence"])
        produced = note.status == "note"
        rows.append({
            "phase": row["before_after"],
            "note": produced,
            "gate_passed": result["decision"]["passed"],
            "reasons": [r.split(" (")[0] for r in result["decision"]["reasons"]],
            "cited": produced and not citation_problems(note, evidence),
            "label_match": produced and evidence.votes[0].label == row["label"],
        })
    table = pd.DataFrame(rows)
    report: dict[str, Any] = {"n_flights": int(len(table))}
    for phase, part in table.groupby("phase"):
        notes = part[part["note"]]
        report[str(phase)] = {
            "n": int(len(part)),
            "note_rate": float(part["note"].mean()),
            "inconclusive_rate": float(1 - part["note"].mean()),
            "rejected_drafts": int((part["gate_passed"] & ~part["note"]).sum()),
            "citation_validity": float(notes["cited"].mean()) if len(notes) else None,
            "label_match_rate": float(notes["label_match"].mean()) if len(notes) else None,
            "stop_reasons": dict(Counter(r for reasons in part["reasons"] for r in reasons)),
        }
    return report
