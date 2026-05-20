"""Node functions for the LangGraph Brain agent graph.

Each node takes the AgentState and returns a partial dict of updates.
This file holds the retrieval-side nodes: plan, retrieve, rerank, critique.
Generation-side nodes (generate, self_check, confidence) are in nodes_gen.py.
"""

from __future__ import annotations

import logging

from langchain_ollama import ChatOllama

from agent.config import DEFAULT_CONFIG, AgentConfig
from agent.prompts import (
    CRITIQUE_SYSTEM,
    CRITIQUE_USER,
    PLAN_SYSTEM,
    PLAN_USER,
    format_chunks,
)
from agent.reranker import rerank
from agent.state import AgentState
from mcp_server.retrieval import search

logger = logging.getLogger(__name__)


def _llm(config: AgentConfig = DEFAULT_CONFIG) -> ChatOllama:
    """Build the Ollama chat model."""
    return ChatOllama(model=config.llm_model, temperature=config.llm_temperature)


def _trace(state: AgentState, message: str) -> list[str]:
    """Append a line to the trace log, returning the new list."""
    existing = state.get("trace", [])
    return existing + [message]


# ---------- Node: plan ----------


def plan_node(state: AgentState, config: AgentConfig = DEFAULT_CONFIG) -> dict:
    """Turn the user question into a focused search query."""
    question = state["question"]
    llm = _llm(config)

    messages = [
        ("system", PLAN_SYSTEM),
        ("user", PLAN_USER.format(question=question)),
    ]
    resp = llm.invoke(messages)
    search_query = resp.content.strip().strip('"')

    # Safety: if the LLM returns something empty or huge, fall back to the question
    if not search_query or len(search_query) > 300:
        search_query = question

    logger.info("plan: search_query=%r", search_query)
    return {
        "search_query": search_query,
        "retry_count": 0,
        "trace": _trace(state, f"plan: query='{search_query}'"),
    }


# ---------- Node: retrieve ----------


def retrieve_node(state: AgentState, config: AgentConfig = DEFAULT_CONFIG) -> dict:
    """Vector search the brain for candidate chunks."""
    query = state.get("search_query") or state["question"]
    candidates = search(query, top_k=config.retrieve_top_k)
    logger.info("retrieve: %d candidates for %r", len(candidates), query)
    return {
        "candidates": candidates,
        "trace": _trace(state, f"retrieve: {len(candidates)} candidates"),
    }


# ---------- Node: rerank ----------


def rerank_node(state: AgentState, config: AgentConfig = DEFAULT_CONFIG) -> dict:
    """Rerank candidates with the cross-encoder, blend with trust, keep top N."""
    question = state["question"]
    candidates = state.get("candidates", [])
    reranked = rerank(
        question,
        candidates,
        top_k=config.final_top_k,
        trust_weight=config.trust_weight,
    )
    logger.info("rerank: kept top %d", len(reranked))
    titles = ", ".join(c["title"][:30] for c in reranked)
    return {
        "reranked": reranked,
        "trace": _trace(state, f"rerank: top {len(reranked)} [{titles}]"),
    }


# ---------- Node: critique ----------


def _parse_critique(text: str) -> tuple[bool, str, str]:
    """Parse the structured critique response. Returns (sufficient, reason, refined_query)."""
    sufficient = False
    reason = ""
    refined = ""
    for line in text.splitlines():
        line = line.strip()
        upper = line.upper()
        if upper.startswith("SUFFICIENT:"):
            sufficient = "YES" in upper
        elif upper.startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()
        elif upper.startswith("REFINED_QUERY:"):
            refined = line.split(":", 1)[1].strip()
    if refined.upper() == "NONE":
        refined = ""
    return sufficient, reason, refined


def critique_node(state: AgentState, config: AgentConfig = DEFAULT_CONFIG) -> dict:
    """Judge whether the reranked chunks are good enough to answer the question.

    If not, and we have retries left, set a refined query so the graph loops back.
    """
    question = state["question"]
    reranked = state.get("reranked", [])
    retry_count = state.get("retry_count", 0)
    llm = _llm(config)

    chunks_text = format_chunks(reranked)
    messages = [
        ("system", CRITIQUE_SYSTEM),
        ("user", CRITIQUE_USER.format(question=question, chunks=chunks_text)),
    ]
    resp = llm.invoke(messages)
    sufficient, reason, refined = _parse_critique(resp.content)

    # If critique fails but we are out of retries, accept what we have
    retries_left = retry_count < config.max_retries
    if not sufficient and not retries_left:
        logger.info("critique: insufficient but out of retries, proceeding")
        return {
            "retrieval_sufficient": True,  # forced accept
            "critique_notes": f"{reason} (accepted: out of retries)",
            "trace": _trace(state, f"critique: insufficient, out of retries -> proceed"),
        }

    update: dict = {
        "retrieval_sufficient": sufficient,
        "critique_notes": reason,
        "trace": _trace(
            state, f"critique: sufficient={sufficient} reason='{reason}'"
        ),
    }
    if not sufficient:
        # set up a retry: bump count, refine the query
        update["retry_count"] = retry_count + 1
        update["search_query"] = refined or state.get("search_query") or question
        update["trace"] = _trace(
            state,
            f"critique: insufficient, retry {retry_count + 1} with query='{update['search_query']}'",
        )
    return update