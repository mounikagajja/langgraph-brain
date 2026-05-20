"""Tunable configuration for the LangGraph Brain agent."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfig:
    # LLM
    llm_model: str = "llama3.1:8b"
    llm_temperature: float = 0.1

    # Retrieval
    retrieve_top_k: int = 20  # how many chunks vector search returns
    final_top_k: int = 5  # how many survive rerank into the answer

    # Rerank: final ranking = rerank_score * (1 - trust_weight) + trust_score * trust_weight
    trust_weight: float = 0.3

    # Critique loop
    max_retries: int = 2  # extra retrieval attempts if critique says "insufficient"

    # Confidence / human-in-the-loop
    confidence_threshold: float = 0.5  # below this -> INTERRUPT for human review


DEFAULT_CONFIG = AgentConfig()