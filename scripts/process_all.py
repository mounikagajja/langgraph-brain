"""Entry point: chunk all raw data, then trust-score every chunk."""

import logging

from ingestion.chunking import chunk_all
from storage.trust_score import score_all

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    print("Step 1/2: chunking raw data")
    n_chunks = chunk_all()
    print(f"  -> {n_chunks} chunks written")

    print("\nStep 2/2: trust-scoring chunks")
    n_scored, stats = score_all()
    print(f"  -> {n_scored} scored chunks")
    print("\nTrust score distribution:")
    for bucket, count in stats["buckets"].items():
        print(f"  {bucket}: {count}")
    print("\nBy source type:")
    for src, count in stats["by_source"].items():
        print(f"  {src}: {count}")