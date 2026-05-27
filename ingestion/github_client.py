"""GitHub API client with rate limit handling and disk caching."""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"
REPO_OWNER = "langchain-ai"
REPO_NAME = "langgraph"
RAW_DIR = Path("data/raw")


class GitHubClient:
    """Thin wrapper around GitHub REST API with caching and rate-limit awareness."""

    def __init__(self, token: str | None = None) -> None:
        token = token or os.getenv("GITHUB_TOKEN")
        if not token:
            raise RuntimeError("GITHUB_TOKEN not set. Add it to .env")

        self.client = httpx.Client(
            base_url=GITHUB_API,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "langgraph-brain-ingest",
            },
            timeout=30.0,
        )

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> GitHubClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _get(self, url: str, params: dict | None = None) -> httpx.Response:
        """GET with rate-limit handling. Sleeps and retries on 403 rate-limit."""
        for _attempt in range(3):
            r = self.client.get(url, params=params)
            if r.status_code == 200:
                return r
            if r.status_code in (403, 429):
                remaining = r.headers.get("X-RateLimit-Remaining", "?")
                reset = r.headers.get("X-RateLimit-Reset")
                if reset and int(remaining or 0) == 0:
                    wait = max(0, int(reset) - int(time.time())) + 5
                    logger.warning("Rate limited. Sleeping %ds.", wait)
                    time.sleep(wait)
                    continue
            r.raise_for_status()
        r.raise_for_status()
        return r

    def paginate(self, url: str, params: dict | None = None) -> Iterator[dict]:
        """Yield items across all pages of a paginated endpoint."""
        params = dict(params or {})
        params.setdefault("per_page", 100)
        next_url: str | None = url
        while next_url:
            r = self._get(next_url, params=params if next_url == url else None)
            yield from r.json()
            link = r.headers.get("Link", "")
            next_url = None
            for part in link.split(","):
                if 'rel="next"' in part:
                    next_url = part.split(";")[0].strip().strip("<>")
                    break

    def issue_comments(self, issue_number: int) -> list[dict]:
        url = f"/repos/{REPO_OWNER}/{REPO_NAME}/issues/{issue_number}/comments"
        return list(self.paginate(url))


def cache_path(kind: str, number: int) -> Path:
    """Disk path for a cached item. kind = 'issues' or 'prs'."""
    d = RAW_DIR / kind
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{number}.json"


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def cutoff_date(months: int = 12) -> datetime:
    return datetime.now(UTC) - timedelta(days=months * 30)
