"""Entry point: ingest LangGraph issues from the last 12 months."""

from ingestion.fetch_issues import fetch_issues

if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    saved = fetch_issues(months=12)
    print(f"\nDone. {saved} new issues cached to data/raw/issues/")