"""F5: run the agent over the val flights and report its reliability.

    uv run python scripts/f5_agent_eval.py                 # deterministic template writer
    uv run python scripts/f5_agent_eval.py --llm --limit 40   # Gemini; needs GOOGLE_API_KEY

Needs f3_health_score and f4_case_retrieval. Uses the in-memory index and val flights.
"""

import argparse
import json
from pathlib import Path

from mtc.agent.evaluate import evaluate_agent
from mtc.agent.graph import make_gemini_writer
from mtc.agent.nodes import template_writer
from mtc.agent.service import TriageService
from mtc.config import Settings, get_settings


def run(settings: Settings, use_llm: bool = False, limit: int | None = None) -> Path:
    writer = make_gemini_writer(settings) if use_llm else template_writer
    service = TriageService.from_processed(settings, writer, local_index=True)
    scores = service.scores
    flights = scores[(scores["split"] == "val") & scores["binary_task"]]
    if limit:
        # take the strongest deviations: those are the flights where a note can be written
        flights = flights.sort_values("score_all", ascending=False).head(limit)
    report = evaluate_agent(service, flights)
    report["writer"] = "gemini:" + settings.gemini_model if use_llm else "template"
    report["gate"] = {
        "min_z": settings.gate_min_z,
        "min_similarity": settings.gate_min_similarity,
        "min_cases": settings.gate_min_cases,
        "min_vote_share": settings.gate_min_vote_share,
    }
    print(json.dumps(report, indent=2))
    name = "agent_eval_llm.json" if use_llm else "agent_eval.json"
    out = settings.results_dir / "f5" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"written {out}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--llm", action="store_true", help="write notes with Gemini")
    parser.add_argument("--limit", type=int, default=None, help="only the top-N flights by score")
    args = parser.parse_args()
    run(get_settings(), args.llm, args.limit)
