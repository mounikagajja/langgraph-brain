"""Qdrant vector store. Stores embeddings + minimal payload for filtered search."""

from __future__ import annotations

import logging
import uuid
from typing import Any, Iterable

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)

logger = logging.getLogger(__name__)

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "langgraph_brain"


class VectorStore:
    def __init__(self, url: str = QDRANT_URL, collection: str = COLLECTION_NAME) -> None:
        self.client = QdrantClient(url=url, check_compatibility=False)
        self.collection = collection

    def ensure_collection(self, vector_dim: int) -> None:
        """Create the collection if it doesn't exist."""
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection in existing:
            logger.info("Collection %s already exists", self.collection)
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        )
        logger.info("Created collection %s (dim=%d)", self.collection, vector_dim)

    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
        batch_size: int = 256,
    ) -> int:
        """Upsert vectors in batches. Returns total upserted."""
        assert len(ids) == len(vectors) == len(payloads)
        total = 0
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i : i + batch_size]
            batch_vecs = vectors[i : i + batch_size]
            batch_payloads = payloads[i : i + batch_size]
            points = [
                PointStruct(
                    # Qdrant point IDs must be int or UUID; derive UUID from chunk_id
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, cid)),
                    vector=vec,
                    payload={**pl, "chunk_id": cid},
                )
                for cid, vec, pl in zip(batch_ids, batch_vecs, batch_payloads)
            ]
            self.client.upsert(collection_name=self.collection, points=points)
            total += len(points)
        return total

    def search(
        self,
        query_vector: list[float],
        top_k: int = 20,
        min_trust: float | None = None,
        source_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return list of {chunk_id, score, payload}."""
        conditions = []
        if min_trust is not None:
            conditions.append(
                FieldCondition(key="trust_score", range=Range(gte=min_trust))
            )
        if source_types:
            for st in source_types:
                conditions.append(FieldCondition(key="source_type", match=MatchValue(value=st)))
        flt = Filter(must=conditions) if conditions else None

        hits = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
            query_filter=flt,
            with_payload=True,
        ).points

        return [
            {"chunk_id": h.payload["chunk_id"], "score": h.score, "payload": h.payload}
            for h in hits
        ]

    def count(self) -> int:
        return self.client.count(collection_name=self.collection, exact=True).count