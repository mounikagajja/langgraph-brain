"""Smoke test: run 2 questions through all 4 versions."""

import logging

from agent.config import config_for_version
from eval.dataset import EVAL_SET
from eval.pipeline import clear_retrieve_cache, run_pipeline

logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(message)s")

if __name__ == "__main__":
    for item in EVAL_SET[:2]:
        q = item["question"]
        print(f"\n=== {q} ===")
        clear_retrieve_cache()
        for v in ["v0", "v1", "v2", "v3"]:
            cfg = config_for_version(v)
            r = run_pipeline(q, cfg)
            print(
                f"  {v}: answer={len(r['answer'])}c  contexts={len(r['contexts'])}  "
                f"conf={r['confidence']:.2f}  citations={len(r['citations'])}  "
                f"retries={r['retries']}"
            )