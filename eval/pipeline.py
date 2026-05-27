"""Version-aware eval pipeline for the v0-v3 comparison.

Runs the agent's logic directly (not through the StateGraph) so each version
can switch components cleanly. Retrieval is cached per question so all 4
versions share one vector-search pass.

  v0 - vector search -> take top-N by vector score -> generate
  v1 - vector search -> bge rerank -> top-N -> generate
  v2 - vector search -> bge rerank x trust -> top-N -> generate
  v3 - v2 + critique/retry loop before generate
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from langchain_ollama import ChatOllama

from agent.config import AgentConfig
from agent.nodes import _parse_critique
from agent.nodes_gen import URL_RE, _parse_self_check
from agent.prompts import (
    CRITIQUE_SYSTEM,
    CRITIQUE_USER,
    GENERATE_SYSTEM,
    GENERATE_USER,
    PLAN_SYSTEM,
    PLAN_USER,
    SELF_CHECK_SYSTEM,
    SELF_CHECK_USER,
    format_chunks,
)
from agent.reranker import rerank
from mcp_server.retrieval import search

logger = logging.getLogger(__name__)




@lru_cache(maxsize=4)
def _chat(model: str, temperature: float) -> ChatOllama:
    return ChatOllama(model=model, temperature=temperature, num_ctx=4096, num_predict=512)


def _gen_llm(cfg: AgentConfig) -> ChatOllama:
    return _chat(cfg.llm_model, cfg.llm_temperature)


def _reason_llm(cfg: AgentConfig) -> ChatOllama:
    return _chat(cfg.reasoning_model, cfg.llm_temperature)



# Cache vector-search results by (query, top_k) so all 4 versions reuse them.
_RETRIEVE_CACHE: dict[tuple[str, int], list[dict]] = {}


def cached_search(query: str, top_k: int) -> list[dict]:
    key = (query, top_k)
    if key not in _RETRIEVE_CACHE:
        _RETRIEVE_CACHE[key] = search(query, top_k=top_k)
    # return a shallow copy so callers can't mutate the cache
    return [dict(c) for c in _RETRIEVE_CACHE[key]]


def clear_retrieve_cache() -> None:
    _RETRIEVE_CACHE.clear()



def _plan(question: str, cfg: AgentConfig) -> str:
    resp = _reason_llm(cfg).invoke(
        [("system", PLAN_SYSTEM), ("user", PLAN_USER.format(question=question))]
    )
    q = resp.content.strip().strip('"')
    return q if q and len(q) <= 300 else question


def _select_chunks(question: str, candidates: list[dict], cfg: AgentConfig) -> list[dict]:
    """Pick the final chunks according to the version's switches."""
    if not cfg.use_reranker:
        # v0: no rerank — take top-N by vector score
        ordered = sorted(candidates, key=lambda c: c.get("vector_score", 0.0), reverse=True)
        return ordered[: cfg.final_top_k]

    # v1/v2/v3: bge rerank. trust_weight 0 disables the trust blend (v1).
    trust_weight = cfg.trust_weight if cfg.use_trust_blend else 0.0
    return rerank(question, candidates, top_k=cfg.final_top_k, trust_weight=trust_weight)


def _critique(question: str, chunks: list[dict], cfg: AgentConfig) -> tuple[bool, str, str]:
    resp = _reason_llm(cfg).invoke(
        [
            ("system", CRITIQUE_SYSTEM),
            ("user", CRITIQUE_USER.format(question=question, chunks=format_chunks(chunks))),
        ]
    )
    return _parse_critique(resp.content)


def _generate(question: str, chunks: list[dict], cfg: AgentConfig) -> tuple[str, list[str]]:
    resp = _gen_llm(cfg).invoke(
        [
            ("system", GENERATE_SYSTEM),
            ("user", GENERATE_USER.format(question=question, chunks=format_chunks(chunks))),
        ]
    )
    answer = resp.content.strip()
    available = {c.get("source_url", "") for c in chunks if c.get("source_url")}
    cited_raw = set(URL_RE.findall(answer))
    citations = sorted(
        url for url in available if any(url in c or c in url for c in cited_raw)
    )
    return answer, citations


def _self_check(question: str, chunks: list[dict], answer: str, cfg: AgentConfig) -> tuple[bool, float]:
    resp = _reason_llm(cfg).invoke(
        [
            ("system", SELF_CHECK_SYSTEM),
            (
                "user",
                SELF_CHECK_USER.format(
                    question=question, chunks=format_chunks(chunks), answer=answer
                ),
            ),
        ]
    )
    grounded, _reason, confidence = _parse_self_check(resp.content)
    if not grounded:
        confidence = min(confidence, 0.4)
    return grounded, confidence



def run_pipeline(question: str, cfg: AgentConfig) -> dict[str, Any]:
    """Run one question through the version's pipeline. Returns a result record."""
    search_query = _plan(question, cfg)
    candidates = cached_search(search_query, cfg.retrieve_top_k)
    chunks = _select_chunks(question, candidates, cfg)

    retries = 0
    critique_notes = ""
    if cfg.use_critique_loop:
        sufficient, critique_notes, refined = _critique(question, chunks, cfg)
        while not sufficient and retries < cfg.max_retries:
            retries += 1
            rq = refined or search_query
            candidates = cached_search(rq, cfg.retrieve_top_k)
            chunks = _select_chunks(question, candidates, cfg)
            sufficient, critique_notes, refined = _critique(question, chunks, cfg)

    answer, citations = _generate(question, chunks, cfg)
    grounded, confidence = _self_check(question, chunks, answer, cfg)

    return {
        "question": question,
        "version": cfg.version,
        "search_query": search_query,
        "answer": answer,
        "citations": citations,
        "contexts": [c.get("text", "") for c in chunks],
        "confidence": confidence,
        "grounded": grounded,
        "needed_human_review": confidence < cfg.confidence_threshold,
        "retries": retries,
        "critique_notes": critique_notes,
    }
