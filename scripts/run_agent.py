"""Run the LangGraph Brain agent end to end.

Handles the human-review interrupt: if the agent pauses for low confidence,
this script prints the draft and resumes with simulated human feedback.
"""

from __future__ import annotations

import logging
import sys

from langgraph.types import Command

from agent.graph import build_graph

logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(message)s")

DEFAULT_QUESTION = "How do I add short-term memory to a LangGraph agent?"


def run(question: str) -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": "demo-1"}}

    print(f"\nQuestion: {question}\n")
    print("Running agent ...\n")

    result = graph.invoke({"question": question}, config=config)

    # If the graph hit an interrupt, result contains an "__interrupt__" key
    if "__interrupt__" in result:
        intr = result["__interrupt__"][0]
        payload = intr.value
        print("=" * 60)
        print("AGENT PAUSED FOR HUMAN REVIEW")
        print("=" * 60)
        print(f"Reason:      {payload['reason']}")
        print(f"Confidence:  {payload['confidence']}")
        print(f"Self-check:  {payload['self_check_notes']}")
        print(f"\nDraft answer:\n{payload['draft_answer']}")
        print("=" * 60)

        # Simulate a human approving the draft
        feedback = "Reviewed: answer is acceptable, approved."
        print(f"\n[human] {feedback}\n")

        result = graph.invoke(Command(resume={"feedback": feedback}), config=config)

    # Final state
    print("=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(result.get("answer", "(no answer)"))
    print()

    if result.get("citations"):
        print("Citations:")
        for url in result["citations"]:
            print(f"  {url}")
        print()

    print(f"Confidence:         {result.get('confidence')}")
    print(f"Needed human review: {result.get('needs_human_review')}")
    if result.get("human_feedback"):
        print(f"Human feedback:      {result['human_feedback']}")

    print("\n--- TRACE ---")
    for line in result.get("trace", []):
        print(f"  {line}")


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else DEFAULT_QUESTION
    run(q)