"""Assemble the LangGraph Brain agent as a StateGraph.

Flow:
  plan -> retrieve -> rerank -> critique
    critique decides: good enough -> generate, or loop back to retrieve (retry)
  generate -> self_check -> confidence_gate
    confidence_gate decides: confident -> END, or interrupt for human review
"""

from __future__ import annotations

import logging

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from agent.nodes import critique_node, plan_node, rerank_node, retrieve_node
from agent.nodes_gen import confidence_gate_node, generate_node, self_check_node
from agent.state import AgentState

logger = logging.getLogger(__name__)



def _route_after_critique(state: AgentState) -> str:
    """If retrieval was judged sufficient, move on. Otherwise retry."""
    if state.get("retrieval_sufficient", False):
        return "generate"
    return "retrieve"  # critique already bumped retry_count + refined query


def _route_after_confidence(state: AgentState) -> str:
    """Low confidence -> human review interrupt. Otherwise finish."""
    if state.get("needs_human_review", False):
        return "human_review"
    return END



def human_review_node(state: AgentState) -> dict:
    """Pause the graph and surface the draft answer for a human decision.

    interrupt() suspends execution. The graph resumes when the caller sends
    a Command(resume=...) with the human's feedback.
    """
    decision = interrupt(
        {
            "reason": "Low confidence answer - human review requested.",
            "question": state["question"],
            "draft_answer": state.get("answer", ""),
            "confidence": state.get("confidence", 0.0),
            "self_check_notes": state.get("self_check_notes", ""),
        }
    )
    # decision is whatever the caller passes to Command(resume=...)
    feedback = ""
    if isinstance(decision, dict):
        feedback = decision.get("feedback", "")
    elif isinstance(decision, str):
        feedback = decision

    trace = state.get("trace", []) + [f"human_review: feedback='{feedback}'"]
    return {"human_feedback": feedback, "trace": trace}



def build_graph():
    """Build and compile the agent graph. Returns a compiled graph."""
    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("plan", plan_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("rerank", rerank_node)
    builder.add_node("critique", critique_node)
    builder.add_node("generate", generate_node)
    builder.add_node("self_check", self_check_node)
    builder.add_node("confidence_gate", confidence_gate_node)
    builder.add_node("human_review", human_review_node)

    # Linear edges
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "retrieve")
    builder.add_edge("retrieve", "rerank")
    builder.add_edge("rerank", "critique")

    # Conditional: critique -> generate (ok) or retrieve (retry)
    builder.add_conditional_edges(
        "critique",
        _route_after_critique,
        {"generate": "generate", "retrieve": "retrieve"},
    )

    builder.add_edge("generate", "self_check")
    builder.add_edge("self_check", "confidence_gate")

    # Conditional: confidence_gate -> END (ok) or human_review (low confidence)
    builder.add_conditional_edges(
        "confidence_gate",
        _route_after_confidence,
        {"human_review": "human_review", END: END},
    )

    builder.add_edge("human_review", END)

    # Checkpointer is required for interrupt() to work
    checkpointer = InMemorySaver()
    return builder.compile(checkpointer=checkpointer)
