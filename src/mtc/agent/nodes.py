"""Steps of the agent: evidence gate, note writing, citation check.

The gate and the checks are plain code, not LLM calls: the model may phrase a note, but
whether a note is allowed and whether it is supported is decided deterministically.
"""

import re
from collections.abc import Callable

from mtc.agent.schemas import Evidence, GateDecision, InspectionNote, inconclusive
from mtc.config import Settings

NoteWriter = Callable[[Evidence], InspectionNote]
NUMBER = re.compile(r"(?<![A-Za-z\d])\d+(?:\.\d+)?")


def evidence_gate(evidence: Evidence, settings: Settings) -> GateDecision:
    """Pass only with a strong residual and enough agreeing similar cases."""
    reasons = []
    strongest = max((abs(r.z) for r in evidence.residuals), default=0.0)
    if strongest < settings.gate_min_z:
        reasons.append(
            f"residuals within normal range (largest {strongest:.1f}, needs {settings.gate_min_z})"
        )
    if len(evidence.cases) < settings.gate_min_cases:
        reasons.append(
            f"too few similar cases ({len(evidence.cases)}, needs {settings.gate_min_cases})"
        )
    elif evidence.votes[0].share < settings.gate_min_vote_share:
        reasons.append(
            f"similar cases disagree (top share {evidence.votes[0].share:.2f}, "
            f"needs {settings.gate_min_vote_share})"
        )
    return GateDecision(passed=not reasons, reasons=reasons)


def allowed_numbers(evidence: Evidence) -> set[str]:
    """Every number a note may quote, in the spellings a writer would use."""
    values: list[float] = [evidence.health_score, evidence.score_percentile, len(evidence.cases)]
    for residual in evidence.residuals:
        values += [residual.z, abs(residual.z)]
    for case in evidence.cases:
        values.append(case.similarity)
    for item in evidence.votes:
        values += [item.cases, item.share, item.share * 100]
    spellings = set()
    for value in values:
        spellings |= {f"{value:.0f}", f"{value:.1f}", f"{value:.2f}", f"{value:g}"}
    return spellings


def citation_problems(note: InspectionNote, evidence: Evidence) -> list[str]:
    """Reasons to reject a drafted note; an empty list means every claim is supported."""
    problems = []
    known, numbers = evidence.ids(), allowed_numbers(evidence)
    if not note.symptoms:
        problems.append("no symptom claim")
    for claim in note.claims():
        if not claim.citations:
            problems.append(f"uncited claim: {claim.text!r}")
        unknown = sorted(set(claim.citations) - known)
        if unknown:
            problems.append(f"unknown citation {unknown} in {claim.text!r}")
        invented = sorted(set(NUMBER.findall(claim.text)) - numbers)
        if invented:
            problems.append(f"number not in evidence {invented} in {claim.text!r}")
    for claim in note.symptoms:
        if not any(c.startswith("R") for c in claim.citations):
            problems.append(f"symptom without a residual citation: {claim.text!r}")
    for claim in note.checks:
        if not any(c.startswith("C") for c in claim.citations):
            problems.append(f"check without a case citation: {claim.text!r}")
    return problems


def confidence(evidence: Evidence) -> str:
    """Set by rule from case agreement, never by the model."""
    top = evidence.votes[0]
    if top.share >= 0.75 and top.cases >= 5:
        return "high"
    return "medium" if top.share >= 0.6 else "low"


def template_writer(evidence: Evidence) -> InspectionNote:
    """Deterministic note without an LLM: the offline fallback and the test double."""
    symptoms = [
        {"text": f"{r.channel} is {r.direction}er than expected in cruise (z = {r.z:.1f}).",
         "citations": [r.id]}
        for r in evidence.residuals
    ]
    top = evidence.votes[0]
    cited = [c.id for c in evidence.cases if c.label == top.label]
    similar = [{
        "text": f"{top.cases} of {len(evidence.cases)} similar past cases were closed as "
                f"'{top.label}'.",
        "citations": cited,
    }]
    checks = [{"text": f"Inspect for: {top.label}.", "citations": cited}]
    return InspectionNote(status="note", symptoms=symptoms, similar_cases=similar, checks=checks)


def finalise(note: InspectionNote, evidence: Evidence) -> InspectionNote:
    """Reject an unsupported draft, otherwise stamp the rule-based confidence."""
    if note.status == "inconclusive":
        return inconclusive("the writer found the evidence insufficient")
    problems = citation_problems(note, evidence)
    if problems:
        return inconclusive("note rejected: " + "; ".join(problems))
    return note.model_copy(update={"confidence": confidence(evidence), "reason": None})
