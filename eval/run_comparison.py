"""Run the full v0-v3 comparison: 50 questions x 4 versions.

For each question we run all 4 versions back-to-back so they share the
retrieval cache. Results are saved per version, resumable after a crash.

Usage: python -m eval.run_comparison
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from agent.config import config_for_version
from eval.dataset import EVAL_SET
from eval.pipeline import clear_retrieve_cache, run_pipeline

logger = logging.getLogger(__name__)

OUT_DIR = Path("data/processed/eval_runs")
VERSIONS = ["v0", "v1", "v2", "v3"]


def _done_questions(version: str) -> set[str]:
    path = OUT_DIR / f"{version}.jsonl"
    done: set[str] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    done.add(json.loads(line)["question"])
                except Exception:
                    pass
    return done


def run_comparison() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Pre-load which questions are already done per version (resume support)
    done = {v: _done_questions(v) for v in VERSIONS}
    files = {v: (OUT_DIR / f"{v}.jsonl").open("a", encoding="utf-8") for v in VERSIONS}

    try:
        for i, item in enumerate(EVAL_SET, 1):
            q = item["question"]
            reference = item["reference"]
            # New question -> clear the retrieval cache (cache is per-question)
            clear_retrieve_cache()

            for v in VERSIONS:
                if q in done[v]:
                    continue
                cfg = config_for_version(v)
                t0 = time.time()
                try:
                    result = run_pipeline(q, cfg)
                except Exception as e:
                    logger.error("Q%d %s failed: %s", i, v, e)
                    result = {
                        "question": q,
                        "version": v,
                        "answer": "",
                        "citations": [],
                        "contexts": [],
                        "confidence": 0.0,
                        "grounded": False,
                        "needed_human_review": False,
                        "retries": 0,
                        "error": str(e),
                    }
                result["reference"] = reference
                files[v].write(json.dumps(result, ensure_ascii=False) + "\n")
                files[v].flush()
                logger.info(
                    "Q%d/%d %s done in %.0fs  conf=%.2f",
                    i,
                    len(EVAL_SET),
                    v,
                    time.time() - t0,
                    result.get("confidence", 0.0),
                )
    finally:
        for f in files.values():
            f.close()

    logger.info("Comparison run complete. Files in %s", OUT_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    run_comparison()
