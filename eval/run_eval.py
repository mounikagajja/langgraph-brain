"""Run the agent over the full eval set, collect answers + contexts.

Output: data/processed/eval_runs/<version>.jsonl
One line per question with: question, reference, answer, contexts, confidence.

This is the slow step. 50 questions x ~1-2 min each = roughly 1-2 hours on CPU.
Progress is saved after every question, so a crash can be resumed.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from langgraph.types import Command

from agent.graph import build_graph
from eval.dataset import EVAL_SET

logger = logging.getLogger(__name__)

OUT_DIR = Path("data/processed/eval_runs")


def _run_one(graph, question: str) -> dict:
    """Run the agent on one question. Auto-approves any human-review interrupt."""
    config = {"configurable": {"thread_id": f"eval-{abs(hash(question))}"}}
    config["recursion_limit"] = 25

    result = graph.invoke({"question": question}, config=config)

    if "__interrupt__" in result:
        # auto-approve so eval measures the agent's own answer
        result = graph.invoke(
            Command(resume={"feedback": "auto-approved for eval"}), config=config
        )

    contexts = [c.get("text", "") for c in result.get("reranked", [])]
    return {
        "answer": result.get("answer", ""),
        "contexts": contexts,
        "confidence": result.get("confidence", 0.0),
        "needed_human_review": result.get("needs_human_review", False),
        "citations": result.get("citations", []),
    }


def run_eval(version: str) -> Path:
    """Run all 50 questions, write results to a versioned jsonl file."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{version}.jsonl"

    # Resume support: skip questions already done
    done_questions: set[str] = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    done_questions.add(json.loads(line)["question"])
                except Exception:
                    pass
        logger.info("Resuming: %d questions already done", len(done_questions))

    graph = build_graph()

    with out_path.open("a", encoding="utf-8") as out:
        for i, item in enumerate(EVAL_SET, 1):
            q = item["question"]
            if q in done_questions:
                continue
            t0 = time.time()
            try:
                run_result = _run_one(graph, q)
            except Exception as e:
                logger.error("Q%d failed: %s", i, e)
                run_result = {
                    "answer": "",
                    "contexts": [],
                    "confidence": 0.0,
                    "needed_human_review": False,
                    "citations": [],
                    "error": str(e),
                }
            record = {
                "question": q,
                "reference": item["reference"],
                **run_result,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            elapsed = time.time() - t0
            logger.info(
                "Q%d/%d done in %.0fs  conf=%.2f  %s",
                i,
                len(EVAL_SET),
                elapsed,
                run_result.get("confidence", 0.0),
                q[:50],
            )

    logger.info("Eval run complete: %s", out_path)
    return out_path


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    version = sys.argv[1] if len(sys.argv) > 1 else "v1_full"
    run_eval(version)
