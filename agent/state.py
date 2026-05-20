"""State definition for the LangGraph Brain agent.

The state is the shared object passed between every node in the graph.
Each node reads from it and returns updates to it.
"""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    # --- Input ---
    question: str  # the user's original question

    # --- Planning ---
    search_query: str  # query the agent decides to search with (may be refined)
    plan_notes: str  # short reasoning about how to approach the question

    # --- Retrieval ---
    candidates: list[dict[str, Any]]  # raw chunks from vector search
    reranked: list[dict[str, Any]]  # top chunks after rerank
    retry_count: int  # how many times we've retried retrieval

    # --- Critique ---
    retrieval_sufficient: bool  # did the critique node accept the retrieval?
    critique_notes: str  # why the critique passed or failed

    # --- Generation ---
    answer: str  # the generated answer
    citations: list[str]  # source URLs cited

    # --- Self-check ---
    self_check_passed: bool  # does the answer actually use the sources?
    self_check_notes: str

    # --- Confidence / HITL ---
    confidence: float  # 0.0 - 1.0
    needs_human_review: bool  # confidence below threshold -> interrupt
    human_feedback: str  # filled in if a human reviews

    # --- Trace (for debugging / observability) ---
    trace: list[str]  # human-readable log of what each node did