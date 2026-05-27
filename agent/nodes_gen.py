"""Generation-side node functions for the LangGraph Brain agent.

This file holds: generate, self_check, and confidence_gate.
Retrieval-side nodes (plan, retrieve, rerank, critique) are in nodes.py.
"""

from __future__ import annotations

import logging
import re

from agent.config import DEFAULT_CONFIG
from agent.nodes import _llm, _trace
from agent.prompts import (
    GENERATE_SYSTEM,
    GENERATE_USER,
    SELF_CHECK_SYSTEM,
    SELF_CHECK_USER,
    format_chunks,
)
from agent.state import AgentState

logger = logging.getLogger(__name__)

URL_RE = re.compile(r"https?://[^\s\)\]]+")


def generate_node(state: AgentState) -> dict:
    """Generate an answer grounded in the reranked chunks."""
    agent_config = DEFAULT_CONFIG
    question = state["question"]
    reranked = state.get("reranked", [])
    llm = _llm(agent_config)

    chunks_text = format_chunks(reranked)
    messages = [
        ("system", GENERATE_SYSTEM),
        ("user", GENERATE_USER.format(question=question, chunks=chunks_text)),
    ]
    resp = llm.invoke(messages)
    answer = resp.content.strip()

    # Collect the source URLs the chunks actually came from
    available_urls = {c.get("source_url", "") for c in reranked if c.get("source_url")}
    cited_raw = set(URL_RE.findall(answer))
    # Match cited URLs against real sources (handle trailing punctuation/.md)
    citations = sorted(
        url
        for url in available_urls
        if any(url in c or c in url for c in cited_raw)
    )

    logger.info("generate: %d chars, %d citations", len(answer), len(citations))
    return {
        "answer": answer,
        "citations": citations,
        "trace": _trace(
            state, f"generate: answer {len(answer)} chars, {len(citations)} citations"
        ),
    }



def _parse_self_check(text: str) -> tuple[bool, str, float]:
    """Parse the structured self-check response. Returns (grounded, reason, confidence)."""
    grounded = False
    reason = ""
    confidence = 0.5
    for line in text.splitlines():
        line = line.strip()
        upper = line.upper()
        if upper.startswith("GROUNDED:"):
            grounded = "YES" in upper
        elif upper.startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()
        elif upper.startswith("CONFIDENCE:"):
            raw = line.split(":", 1)[1].strip()
            m = re.search(r"[0-9]*\.?[0-9]+", raw)
            if m:
                try:
                    confidence = max(0.0, min(1.0, float(m.group())))
                except ValueError:
                    pass
    return grounded, reason, confidence


def self_check_node(state: AgentState) -> dict:
    """Verify the answer is grounded in the context. Produces a confidence score."""
    agent_config = DEFAULT_CONFIG
    question = state["question"]
    reranked = state.get("reranked", [])
    answer = state.get("answer", "")
    llm = _llm(agent_config)

    chunks_text = format_chunks(reranked)
    messages = [
        ("system", SELF_CHECK_SYSTEM),
        (
            "user",
            SELF_CHECK_USER.format(
                question=question, chunks=chunks_text, answer=answer
            ),
        ),
    ]
    resp = llm.invoke(messages)
    grounded, reason, confidence = _parse_self_check(resp.content)

    # If self-check says not grounded, cap confidence low regardless of the number
    if not grounded:
        confidence = min(confidence, 0.4)

    # No citations is itself a red flag
    if not state.get("citations"):
        confidence = min(confidence, 0.5)

    logger.info("self_check: grounded=%s confidence=%.2f", grounded, confidence)
    return {
        "self_check_passed": grounded,
        "self_check_notes": reason,
        "confidence": round(confidence, 3),
        "trace": _trace(
            state, f"self_check: grounded={grounded} confidence={confidence:.2f}"
        ),
    }


def confidence_gate_node(state: AgentState) -> dict:
    """Decide whether the answer needs human review based on confidence."""
    agent_config = DEFAULT_CONFIG
    confidence = state.get("confidence", 0.0)
    needs_review = confidence < agent_config.confidence_threshold
    logger.info(
        "confidence_gate: confidence=%.2f threshold=%.2f needs_review=%s",
        confidence,
        agent_config.confidence_threshold,
        needs_review,
    )
    return {
        "needs_human_review": needs_review,
        "trace": _trace(
            state,
            f"confidence_gate: needs_human_review={needs_review} "
            f"(conf={confidence:.2f} < {agent_config.confidence_threshold})",
        ),
    }
