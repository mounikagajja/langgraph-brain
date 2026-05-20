"""Retrieval helpers shared by MCP tools. Wraps the embedding model + stores."""

from __future__ import annotations

import functools
from typing import Any

from sentence_transformers import SentenceTransformer

from storage.metadata_db import connect, get_chunks
from storage.vector_store import VectorStore

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@functools.lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    """Load the embedding model once and reuse it."""
    return SentenceTransformer(EMBED_MODEL)


@functools.lru_cache(maxsize=1)
def _vector_store() -> VectorStore:
    return VectorStore()


def embed_query(text: str) -> list[float]:
    return _model().encode([text], normalize_embeddings=True)[0].tolist()


def search(
    query: str,
    top_k: int = 10,
    min_trust: float | None = None,
    source_types: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Semantic search. Returns chunks with text, metadata, vector score, trust score."""
    qvec = embed_query(query)
    hits = _vector_store().search(
        qvec, top_k=top_k, min_trust=min_trust, source_types=source_types
    )
    if not hits:
        return []

    conn = connect()
    try:
        chunks = get_chunks(conn, [h["chunk_id"] for h in hits])
    finally:
        conn.close()

    score_by_id = {h["chunk_id"]: h["score"] for h in hits}
    results: list[dict[str, Any]] = []
    for c in chunks:
        results.append(
            {
                "chunk_id": c["chunk_id"],
                "title": c["title"],
                "text": c["text"],
                "source_type": c["source_type"],
                "source_url": c["source_url"],
                "trust_score": c["trust_score"],
                "vector_score": round(score_by_id.get(c["chunk_id"], 0.0), 4),
                "metadata": c["metadata"],
            }
        )
    return results


def get_chunk(chunk_id: str) -> dict[str, Any] | None:
    """Fetch a single chunk by its ID."""
    conn = connect()
    try:
        rows = get_chunks(conn, [chunk_id])
    finally:
        conn.close()
    return rows[0] if rows else None


def source_summary() -> dict[str, Any]:
    """Counts by source type, for the list_sources tool."""
    conn = connect()
    try:
        cur = conn.execute(
            "SELECT source_type, COUNT(*) AS n FROM chunks GROUP BY source_type"
        )
        by_type = {row["source_type"]: row["n"] for row in cur}
        total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    finally:
        conn.close()
    return {"total_chunks": total, "by_source_type": by_type}