"""Compute a per-chunk trust score based on source type, resolution, and recency.

Score in [0.0, 1.0]. Weighted from chunk metadata. We chose:
  source_type > resolution > version_match > recency

This is a deliberate, defensible scoring choice — official docs are most
trustworthy, then merged PRs, then closed issues, then open issues. Recency
breaks ties.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
OUT_PATH = Path("data/processed/chunks_scored.jsonl")

# Base weights by source/state. Tunable.
BASE = {
    "doc": 1.00,
    "pr_merged": 0.80,
    "pr_closed": 0.50,
    "pr_open": 0.45,
    "issue_closed": 0.60,
    "issue_open": 0.40,
}

# Boosts and penalties
BOT_PENALTY = 0.15
VERSION_MATCH_BOOST = 0.05
RECENCY_MAX_BOOST = 0.10  # full boost for items < 30 days old, decays to 0 at 12 months


def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def _recency_boost(updated_at: str | None, now: datetime | None = None) -> float:
    dt = _parse_dt(updated_at)
    if not dt:
        return 0.0
    now = now or datetime.now(timezone.utc)
    age_days = (now - dt).days
    if age_days <= 30:
        return RECENCY_MAX_BOOST
    if age_days >= 365:
        return 0.0
    return RECENCY_MAX_BOOST * (1 - (age_days - 30) / (365 - 30))


def _version_match_boost(text: str) -> float:
    """Bumps score if the chunk mentions a current major version marker."""
    lowered = text.lower()
    if "langgraph 1." in lowered or "langgraph v1" in lowered:
        return VERSION_MATCH_BOOST
    if "1.0" in lowered and "langgraph" in lowered:
        return VERSION_MATCH_BOOST
    return 0.0


def _base_score(chunk: dict) -> float:
    st = chunk["source_type"]
    meta = chunk.get("metadata") or {}
    state = (meta.get("state") or "").lower()
    if st == "doc":
        return BASE["doc"]
    if st == "pr":
        if state == "merged":
            return BASE["pr_merged"]
        if state == "closed":
            return BASE["pr_closed"]
        return BASE["pr_open"]
    if st == "issue":
        if state == "closed":
            return BASE["issue_closed"]
        return BASE["issue_open"]
    return 0.3  # fallback for anything unexpected


def score_chunk(chunk: dict, now: datetime | None = None) -> float:
    score = _base_score(chunk)
    meta = chunk.get("metadata") or {}

    if meta.get("is_bot"):
        score -= BOT_PENALTY

    score += _recency_boost(meta.get("updated_at"), now=now)
    score += _version_match_boost(chunk.get("text", ""))

    return max(0.0, min(1.0, round(score, 4)))


def score_all() -> tuple[int, dict]:
    """Read chunks.jsonl, write chunks_scored.jsonl with trust_score added.

    Returns (count, distribution_stats).
    """
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    buckets = {"0.0-0.3": 0, "0.3-0.5": 0, "0.5-0.7": 0, "0.7-0.9": 0, "0.9-1.0": 0}
    count = 0
    by_source: dict[str, int] = {}

    with CHUNKS_PATH.open(encoding="utf-8") as inp, OUT_PATH.open("w", encoding="utf-8") as out:
        for line in inp:
            chunk = json.loads(line)
            chunk["trust_score"] = score_chunk(chunk)
            out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            count += 1

            s = chunk["trust_score"]
            if s < 0.3:
                buckets["0.0-0.3"] += 1
            elif s < 0.5:
                buckets["0.3-0.5"] += 1
            elif s < 0.7:
                buckets["0.5-0.7"] += 1
            elif s < 0.9:
                buckets["0.7-0.9"] += 1
            else:
                buckets["0.9-1.0"] += 1

            by_source[chunk["source_type"]] = by_source.get(chunk["source_type"], 0) + 1

    stats = {"buckets": buckets, "by_source": by_source}
    logger.info("Scored %d chunks. Distribution: %s", count, stats)
    return count, stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    score_all()