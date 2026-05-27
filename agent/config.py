"""Tunable configuration for the LangGraph Brain agent.

Supports 4 evaluation versions that each add one component:
  v0 - vector search only (no rerank, no trust, no critique loop)
  v1 - + bge reranker
  v2 - + trust-weighted reranking
  v3 - + critique/retry loop
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class AgentConfig:
    # LLM
    llm_model: str = "llama3.1:8b"
    reasoning_model: str = "llama3.2:3b"  # plan / critique / self_check
    llm_temperature: float = 0.1

    # Retrieval
    retrieve_top_k: int = 12
    final_top_k: int = 4

    # Rerank blend
    trust_weight: float = 0.3

    # Critique loop
    max_retries: int = 2

    # Confidence / human-in-the-loop
    confidence_threshold: float = 0.5

    # --- Version switches (set by version presets below) ---
    use_reranker: bool = True
    use_trust_blend: bool = True
    use_critique_loop: bool = True
    version: str = "v3"


# Version presets — each adds exactly one component over the previous
_VERSIONS = {
    "v0": dict(use_reranker=False, use_trust_blend=False, use_critique_loop=False),
    "v1": dict(use_reranker=True, use_trust_blend=False, use_critique_loop=False),
    "v2": dict(use_reranker=True, use_trust_blend=True, use_critique_loop=False),
    "v3": dict(use_reranker=True, use_trust_blend=True, use_critique_loop=True),
}


def config_for_version(version: str) -> AgentConfig:
    """Return an AgentConfig with the switches set for the given version."""
    if version not in _VERSIONS:
        raise ValueError(f"Unknown version {version!r}. Choose from {list(_VERSIONS)}")
    return replace(AgentConfig(), version=version, **_VERSIONS[version])


DEFAULT_CONFIG = AgentConfig()
