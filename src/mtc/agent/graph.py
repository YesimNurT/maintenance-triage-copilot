"""LangGraph orchestration: gate -> write note -> check citations, or stop as inconclusive.

    evidence ──► gate ──fail──► inconclusive ──► END
                  │
                 pass
                  ▼
                write (LLM or template) ──► check ──► END
"""

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from mtc.agent.nodes import NoteWriter, evidence_gate, finalise, template_writer
from mtc.agent.prompts import HUMAN_TEMPLATE, SYSTEM_PROMPT
from mtc.agent.schemas import Evidence, GateDecision, InspectionNote, inconclusive
from mtc.config import Settings


class AgentState(TypedDict, total=False):
    evidence: Evidence
    decision: GateDecision
    draft: InspectionNote
    note: InspectionNote


def build_graph(settings: Settings, writer: NoteWriter = template_writer) -> Any:
    """Compile the agent. ``writer`` is injected so tests never call a real LLM."""

    def gate(state: AgentState) -> AgentState:
        return {"decision": evidence_gate(state["evidence"], settings)}

    def stop(state: AgentState) -> AgentState:
        return {"note": inconclusive("; ".join(state["decision"].reasons))}

    def write(state: AgentState) -> AgentState:
        return {"draft": writer(state["evidence"])}

    def check(state: AgentState) -> AgentState:
        return {"note": finalise(state["draft"], state["evidence"])}

    graph = StateGraph(AgentState)
    graph.add_node("gate", gate)
    graph.add_node("inconclusive", stop)
    graph.add_node("write", write)
    graph.add_node("check", check)
    graph.add_edge(START, "gate")
    graph.add_conditional_edges(
        "gate", lambda state: "write" if state["decision"].passed else "inconclusive"
    )
    graph.add_edge("write", "check")
    graph.add_edge("check", END)
    graph.add_edge("inconclusive", END)
    return graph.compile()


def run_agent(graph: Any, evidence: Evidence) -> AgentState:
    """Run the compiled graph for one flight and return the final state."""
    return graph.invoke({"evidence": evidence})


def make_gemini_writer(settings: Settings) -> NoteWriter:
    """Note writer backed by Gemini with structured output (needs GOOGLE_API_KEY)."""
    from langchain_google_genai import ChatGoogleGenerativeAI  # optional dependency

    model = ChatGoogleGenerativeAI(
        model=settings.gemini_model, google_api_key=settings.google_api_key, temperature=0
    ).with_structured_output(InspectionNote)

    def write(evidence: Evidence) -> InspectionNote:
        human = HUMAN_TEMPLATE.format(evidence_json=evidence.model_dump_json(indent=2))
        return model.invoke([("system", SYSTEM_PROMPT), ("human", human)])

    return write
