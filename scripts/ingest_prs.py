"""Entry point: ingest LangGraph pull requests from the last 12 months."""

from ingestion.fetch_prs import fetch_prs

if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    saved = fetch_prs(months=12)
    print(f"\nDone. {saved} new PRs cached to data/raw/prs/")