"""Fetch LangGraph issues + their comments, cache to disk.

Issues only (not PRs). GitHub's /issues endpoint returns both — we filter PRs out
and handle them in fetch_prs.py.
"""

from __future__ import annotations

import logging

from ingestion.github_client import (
    REPO_NAME,
    REPO_OWNER,
    GitHubClient,
    cache_path,
    cutoff_date,
    save_json,
)

logger = logging.getLogger(__name__)


def fetch_issues(months: int = 12) -> int:
    """Pull issues updated within the last N months. Returns count saved."""
    since = cutoff_date(months).isoformat()
    url = f"/repos/{REPO_OWNER}/{REPO_NAME}/issues"
    params = {"state": "all", "since": since, "sort": "updated", "direction": "desc"}

    saved = 0
    with GitHubClient() as gh:
        for item in gh.paginate(url, params=params):
            if "pull_request" in item:  # skip PRs from this endpoint
                continue
            number = item["number"]
            path = cache_path("issues", number)
            if path.exists():
                continue  # already cached, skip
            item["comments_data"] = gh.issue_comments(number)
            save_json(path, item)
            saved += 1
            if saved % 25 == 0:
                logger.info("Saved %d issues so far...", saved)
    return saved


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    n = fetch_issues(months=12)
    logger.info("Done. Saved %d new issues.", n)
