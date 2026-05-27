"""Fetch LangGraph docs from docs.langchain.com.

The repo's local docs/llms.txt only lists ~13 highlight pages. The full doc
index lives at docs.langchain.com/llms.txt and contains hundreds of pages.
We pull that, filter to LangGraph + LangSmith content (skipping granular API
references), and fetch each page as markdown.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

LLMS_INDEX_URL = "https://docs.langchain.com/llms.txt"
DOCS_OUT = Path("data/raw/docs")

# URL path prefixes we want to keep
KEEP_PREFIXES = (
    "/oss/python/langgraph/",
    "/oss/python/common-errors",
    "/langsmith/",
)

# URL path prefixes we skip (too granular, mostly REST endpoint stubs)
SKIP_PREFIXES = (
    "/api-reference/",
    "/langsmith/agent-server-api/",  # 100+ tiny endpoint pages
)


def _extract_doc_urls(llms_txt: str) -> list[tuple[str, str]]:
    """Parse llms.txt markdown links. Returns list of (title, url)."""
    pattern = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for match in pattern.finditer(llms_txt):
        title, url = match.group(1), match.group(2)
        if "docs.langchain.com" not in url:
            continue
        path = urlparse(url).path
        if any(path.startswith(p) for p in SKIP_PREFIXES):
            continue
        if not any(path.startswith(p) for p in KEEP_PREFIXES):
            continue
        if url in seen:
            continue
        seen.add(url)
        out.append((title, url))
    return out


def _url_to_filename(url: str) -> str:
    path = urlparse(url).path.strip("/")
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", path)
    return f"{safe}.md" if safe else "index.md"


def fetch_docs() -> int:
    """Fetch the docs index, filter, then download each filtered page as markdown."""
    DOCS_OUT.mkdir(parents=True, exist_ok=True)

    with httpx.Client(
        timeout=30.0,
        headers={"User-Agent": "langgraph-brain-ingest"},
        follow_redirects=True,
    ) as client:
        logger.info("Fetching docs index: %s", LLMS_INDEX_URL)
        r = client.get(LLMS_INDEX_URL)
        r.raise_for_status()
        llms_txt = r.text

        (DOCS_OUT / "_llms_index.md").write_text(llms_txt, encoding="utf-8")

        doc_links = _extract_doc_urls(llms_txt)
        logger.info(
            "Index has %d total links; %d match our filter (LangGraph + LangSmith)",
            llms_txt.count("](http"),
            len(doc_links),
        )

        saved = 0
        skipped = 0
        for title, url in doc_links:
            md_url = url if url.endswith(".md") else url + ".md"
            try:
                resp = client.get(md_url)
                if resp.status_code != 200:
                    resp = client.get(url)
                if resp.status_code != 200:
                    logger.warning("Skip %s (status %d)", url, resp.status_code)
                    skipped += 1
                    continue
                content = resp.text
            except httpx.HTTPError as e:
                logger.warning("Skip %s (%s)", url, e)
                skipped += 1
                continue

            fname = _url_to_filename(url)
            out_path = DOCS_OUT / fname
            header = f"<!-- source: {url}\ntitle: {title} -->\n\n"
            out_path.write_text(header + content, encoding="utf-8")
            saved += 1
            if saved % 20 == 0:
                logger.info("Fetched %d docs so far...", saved)
            time.sleep(0.15)  # be polite

        logger.info("Done. Saved %d docs, skipped %d, output %s", saved, skipped, DOCS_OUT)
        return saved


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    fetch_docs()
