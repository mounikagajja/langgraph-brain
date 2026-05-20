"""Fetch LangGraph docs by shallow-cloning the repo and copying markdown files."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_URL = "https://github.com/langchain-ai/langgraph.git"
CLONE_DIR = Path("data/raw/_repo_clone")
DOCS_OUT = Path("data/raw/docs")


def fetch_docs() -> int:
    """Shallow-clone the LangGraph repo and copy all markdown files from docs/.

    Returns the count of markdown files copied.
    """
    if CLONE_DIR.exists():
        logger.info("Removing existing clone at %s", CLONE_DIR)
        shutil.rmtree(CLONE_DIR, onerror=_force_remove)

    CLONE_DIR.parent.mkdir(parents=True, exist_ok=True)
    DOCS_OUT.mkdir(parents=True, exist_ok=True)

    logger.info("Shallow-cloning %s ...", REPO_URL)
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(CLONE_DIR)],
        check=True,
    )

    docs_src = CLONE_DIR / "docs"
    if not docs_src.exists():
        # Fallback: some repos put docs at the root
        docs_src = CLONE_DIR

    saved = 0
    for md in docs_src.rglob("*.md"):
        rel = md.relative_to(docs_src)
        dest = DOCS_OUT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(md, dest)
        saved += 1

    # Cleanup the clone, we only needed the markdown
    shutil.rmtree(CLONE_DIR, onerror=_force_remove)
    logger.info("Copied %d markdown files to %s", saved, DOCS_OUT)
    return saved


def _force_remove(func, path, exc_info):
    """Windows-safe rmtree handler for read-only .git files."""
    import os
    import stat

    os.chmod(path, stat.S_IWRITE)
    func(path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    n = fetch_docs()
    logger.info("Done. Copied %d markdown files.", n)