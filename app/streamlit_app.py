"""Streamlit UI: ranked flights on the left, the inspection note and review on the right.

    uv run streamlit run app/streamlit_app.py     # needs the API: uvicorn mtc.api.app:app
"""

import pandas as pd
import streamlit as st

from mtc.api.client import ApiClient
from mtc.config import get_settings

ARROW = {"high": "↑", "low": "↓"}


def symptom_text(symptoms: list[dict]) -> str:
    if not symptoms:
        return "within normal range"
    parts = [f"{s['channel'].removeprefix('E1 ')} {ARROW[s['direction']]}" for s in symptoms]
    return " · ".join(parts)


def show_claims(title: str, claims: list[dict]) -> None:
    if claims:
        st.markdown(f"**{title}**")
        for claim in claims:
            st.markdown(f"- {claim['text']}  `{' '.join(claim['citations'])}`")


def show_note(result: dict) -> None:
    note, evidence = result["note"], result["evidence"]
    if note["status"] == "inconclusive":
        st.warning(note["reason"])
    else:
        st.success(f"Inspection note · confidence: {note['confidence']}")
        show_claims("Symptoms", note["symptoms"])
        show_claims("Similar cases", note["similar_cases"])
        show_claims("Check first", note["checks"])
    with st.expander("Evidence"):
        st.caption(
            f"Deviation score {evidence['health_score']} "
            f"(higher than {evidence['score_percentile']}% of healthy training flights)"
        )
        st.markdown("Residuals: observed minus expected, in units of the healthy spread")
        st.dataframe(pd.DataFrame(evidence["residuals"]), hide_index=True)
        st.markdown("Similar past cases (training flights only)")
        st.dataframe(pd.DataFrame(evidence["cases"]), hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Maintenance Triage Copilot", layout="wide")
    st.title("Maintenance Triage Copilot")
    st.caption(
        "Decision-support prototype on public NGAFID data. Not an airworthiness tool: "
        "the score measures deviation from post-maintenance behaviour, and a technician "
        "decides."
    )
    client = ApiClient(get_settings().api_url)
    limit = st.sidebar.slider("Flights shown", 5, 50, 15)

    try:
        ranking = client.ranking(limit)
    except Exception as error:  # noqa: BLE001 - show any connection problem to the user
        st.error(f"API not reachable at {client.base_url}: {error}")
        st.stop()

    left, right = st.columns([5, 6])
    with left:
        st.subheader("Ranking")
        table = pd.DataFrame(
            {
                "rank": [r["rank"] for r in ranking],
                "flight": [r["flight_id"] for r in ranking],
                "deviation": [r["health_score"] for r in ranking],
                "symptoms": [symptom_text(r["symptoms"]) for r in ranking],
            }
        )
        st.dataframe(table, hide_index=True, height=560)
    with right:
        flight_id = st.selectbox("Flight", [r["flight_id"] for r in ranking])
        st.subheader(f"Flight {flight_id}")
        show_note(client.note(flight_id))
        comment = st.text_input("Comment", key=f"comment-{flight_id}")
        approve, reject = st.columns(2)
        for column, decision in ((approve, "approve"), (reject, "reject")):
            if column.button(decision.capitalize(), key=f"{decision}-{flight_id}"):
                record = client.review(flight_id, decision, comment)
                extra = " and added as a new case" if record["added_as_case"] else ""
                st.info(f"Review stored: {decision}{extra}.")


main()
