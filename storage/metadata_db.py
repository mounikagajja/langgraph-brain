"""SQLite store for chunk text and metadata. Looked up by chunk_id after vector search."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

DB_PATH = Path("data/processed/metadata.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     TEXT PRIMARY KEY,
    source_type  TEXT NOT NULL,
    source_id    TEXT NOT NULL,
    source_url   TEXT,
    title        TEXT,
    text         TEXT NOT NULL,
    metadata     TEXT,        -- JSON blob
    trust_score  REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_source_type ON chunks(source_type);
CREATE INDEX IF NOT EXISTS idx_chunks_trust       ON chunks(trust_score);
"""


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def upsert_chunks(conn: sqlite3.Connection, chunks: Iterable[dict[str, Any]]) -> int:
    rows = [
        (
            c["chunk_id"],
            c["source_type"],
            c["source_id"],
            c.get("source_url", ""),
            c.get("title", ""),
            c["text"],
            json.dumps(c.get("metadata") or {}, ensure_ascii=False),
            float(c.get("trust_score", 0.0)),
        )
        for c in chunks
    ]
    conn.executemany(
        """
        INSERT INTO chunks (chunk_id, source_type, source_id, source_url, title, text, metadata, trust_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(chunk_id) DO UPDATE SET
            source_type = excluded.source_type,
            source_id   = excluded.source_id,
            source_url  = excluded.source_url,
            title       = excluded.title,
            text        = excluded.text,
            metadata    = excluded.metadata,
            trust_score = excluded.trust_score
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def get_chunks(conn: sqlite3.Connection, chunk_ids: list[str]) -> list[dict[str, Any]]:
    if not chunk_ids:
        return []
    placeholders = ",".join("?" * len(chunk_ids))
    cur = conn.execute(
        f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})", chunk_ids
    )
    out: list[dict[str, Any]] = []
    for row in cur:
        out.append(
            {
                "chunk_id": row["chunk_id"],
                "source_type": row["source_type"],
                "source_id": row["source_id"],
                "source_url": row["source_url"],
                "title": row["title"],
                "text": row["text"],
                "metadata": json.loads(row["metadata"] or "{}"),
                "trust_score": row["trust_score"],
            }
        )
    # preserve order of input ids
    by_id = {c["chunk_id"]: c for c in out}
    return [by_id[i] for i in chunk_ids if i in by_id]


def count(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
