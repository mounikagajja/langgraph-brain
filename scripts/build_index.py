"""Read chunks_scored.jsonl, embed each chunk locally, populate SQLite + Qdrant."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer

from storage.metadata_db import connect, upsert_chunks
from storage.metadata_db import count as sqlite_count
from storage.vector_store import VectorStore

CHUNKS_PATH = Path("data/processed/chunks_scored.jsonl")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64  # embed batch

logger = logging.getLogger(__name__)


def load_chunks() -> list[dict]:
    chunks = []
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def build() -> None:
    logger.info("Loading chunks from %s", CHUNKS_PATH)
    chunks = load_chunks()
    logger.info("Loaded %d chunks", len(chunks))

    # 1. SQLite
    logger.info("Writing chunks to SQLite ...")
    conn = connect()
    upsert_chunks(conn, chunks)
    logger.info("SQLite now has %d rows", sqlite_count(conn))
    conn.close()

    # 2. Embeddings
    logger.info("Loading embedding model: %s", EMBED_MODEL)
    model = SentenceTransformer(EMBED_MODEL)
    dim = model.get_sentence_embedding_dimension()
    logger.info("Embedding dim = %d", dim)

    # 3. Qdrant
    vs = VectorStore()
    vs.ensure_collection(vector_dim=dim)

    ids: list[str] = []
    payloads: list[dict] = []
    texts: list[str] = []
    for c in chunks:
        ids.append(c["chunk_id"])
        payloads.append(
            {
                "source_type": c["source_type"],
                "source_id": c["source_id"],
                "trust_score": c["trust_score"],
            }
        )
        # Prefix title to give the embedding a bit of context cheaply
        title = c.get("title", "")
        body = c.get("text", "")
        text = f"{title}\n\n{body}" if title and title not in body else body
        texts.append(text)

    logger.info("Embedding %d chunks in batches of %d ...", len(texts), BATCH_SIZE)
    t0 = time.time()
    vectors: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        emb = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        vectors.extend(emb.tolist())
        if (i // BATCH_SIZE) % 20 == 0:
            elapsed = time.time() - t0
            done = i + len(batch)
            rate = done / elapsed if elapsed else 0
            eta = (len(texts) - done) / rate if rate else 0
            logger.info(
                "Embedded %d/%d (%.1f/s, ETA %.0fs)", done, len(texts), rate, eta
            )

    logger.info("Embedding done in %.1fs. Upserting to Qdrant ...", time.time() - t0)
    n = vs.upsert(ids=ids, vectors=vectors, payloads=payloads)
    logger.info("Qdrant now has %d points (upserted %d)", vs.count(), n)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    build()
