"""bge-reranker-v2-m3 cross-encoder reranker.

The vector search gives us semantic candidates. The reranker scores each
(query, chunk) pair directly, which is far more precise than embedding cosine
similarity. We then blend rerank score with trust score for the final ranking.

Uses sentence-transformers' CrossEncoder (works with current transformers;
FlagEmbedding's reranker is incompatible with transformers 5.x).
"""

from __future__ import annotations

import functools
import logging
import math
from typing import Any

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


@functools.lru_cache(maxsize=1)
def _reranker() -> CrossEncoder:
    """Load the reranker once."""
    logger.info("Loading reranker: %s", RERANKER_MODEL)
    return CrossEncoder(RERANKER_MODEL, max_length=512)


def _sigmoid(x: float) -> float:
    """bge-reranker outputs logits; sigmoid maps them to [0, 1]."""
    return 1.0 / (1.0 + math.exp(-x))


def rerank(
    query: str,
    chunks: list[dict[str, Any]],
    top_k: int = 5,
    trust_weight: float = 0.3,
) -> list[dict[str, Any]]:
    """Rerank chunks by (query, text) relevance, blended with trust score.

    Each returned chunk gets three new fields:
        rerank_score   - raw cross-encoder logit
        rerank_norm    - sigmoid of the logit, in [0, 1]
        final_score    - blended score used for the final sort

    Returns the top_k chunks sorted by final_score descending.
    """
    if not chunks:
        return []

    pairs = [(query, c["text"]) for c in chunks]
    raw_scores = _reranker().predict(pairs)

    scored: list[dict[str, Any]] = []
    for chunk, raw in zip(chunks, raw_scores):
        raw = float(raw)
        norm = _sigmoid(raw)
        trust = float(chunk.get("trust_score", 0.0))
        final = norm * (1 - trust_weight) + trust * trust_weight
        scored.append(
            {
                **chunk,
                "rerank_score": round(raw, 4),
                "rerank_norm": round(norm, 4),
                "final_score": round(final, 4),
            }
        )

    scored.sort(key=lambda c: c["final_score"], reverse=True)
    return scored[:top_k]