"""Test the retrieval-side nodes: plan -> retrieve -> rerank -> critique."""

import logging

from agent.nodes import critique_node, plan_node, rerank_node, retrieve_node
from agent.state import AgentState

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

QUESTION = "How do I add short-term memory to a LangGraph agent?"

if __name__ == "__main__":
    state: AgentState = {"question": QUESTION}

    print(f"Question: {QUESTION}\n")

    print("=== plan ===")
    state.update(plan_node(state))
    print(f"search_query: {state['search_query']}\n")

    print("=== retrieve ===")
    state.update(retrieve_node(state))
    print(f"candidates: {len(state['candidates'])}\n")

    print("=== rerank ===")
    state.update(rerank_node(state))
    for c in state["reranked"]:
        print(f"  final={c['final_score']:.3f} [{c['source_type']}] {c['title']}")
    print()

    print("=== critique ===")
    state.update(critique_node(state))
    print(f"sufficient: {state['retrieval_sufficient']}")
    print(f"notes: {state['critique_notes']}\n")

    print("=== trace ===")
    for line in state["trace"]:
        print(f"  {line}")