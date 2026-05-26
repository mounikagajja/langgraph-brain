"""Smoke test: run the eval runner on just 2 questions."""

import logging

from agent.graph import build_graph
from eval.dataset import EVAL_SET
from eval.run_eval import _run_one

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

if __name__ == "__main__":
    graph = build_graph()
    for item in EVAL_SET[:2]:
        q = item["question"]
        print(f"\n=== {q} ===")
        result = _run_one(graph, q)
        print(f"answer ({len(result['answer'])} chars): {result['answer'][:200]}")
        print(f"contexts: {len(result['contexts'])}")
        print(f"confidence: {result['confidence']}")