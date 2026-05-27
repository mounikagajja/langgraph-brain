"""Fetch LangGraph pull requests + their review comments, cache to disk."""

from __future__ import annotations

import logging
from datetime import datetime

from ingestion.github_client import (
    REPO_NAME,
    REPO_OWNER,
    GitHubClient,
    cache_path,
    cutoff_date,
    save_json,
)

logger = logging.getLogger(__name__)


def fetch_prs(months: int = 12) -> int:
    """Pull PRs updated within the last N months. Returns count saved.

    PRs endpoint doesn't support `since`, so we paginate sorted by updated desc
    and stop when we cross the cutoff.
    """
    cutoff = cutoff_date(months)
    url = f"/repos/{REPO_OWNER}/{REPO_NAME}/pulls"
    params = {"state": "all", "sort": "updated", "direction": "desc"}

    saved = 0
    with GitHubClient() as gh:
        for item in gh.paginate(url, params=params):
            updated = datetime.fromisoformat(item["updated_at"].replace("Z", "+00:00"))
            if updated < cutoff:
                break  # rest are older, stop pagination

            number = item["number"]
            path = cache_path("prs", number)
            if path.exists():
                continue

            # PR review comments live at a different endpoint
            review_comments_url = (
                f"/repos/{REPO_OWNER}/{REPO_NAME}/pulls/{number}/comments"
            )
            issue_comments_url = (
                f"/repos/{REPO_OWNER}/{REPO_NAME}/issues/{number}/comments"
            )
            item["review_comments_data"] = list(gh.paginate(review_comments_url))
            item["issue_comments_data"] = list(gh.paginate(issue_comments_url))

            save_json(path, item)
            saved += 1
            if saved % 25 == 0:
                logger.info("Saved %d PRs so far...", saved)
    return saved


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    n = fetch_prs(months=12)
    logger.info("Done. Saved %d new PRs.", n)
