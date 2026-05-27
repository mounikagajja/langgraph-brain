"""Entry point: clone LangGraph repo and copy all markdown docs."""

from ingestion.fetch_docs import fetch_docs

if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    saved = fetch_docs()
    print(f"\nDone. {saved} markdown files copied to data/raw/docs/")
