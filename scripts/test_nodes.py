"""Test the full node chain: plan -> retrieve -> rerank -> critique
-> generate -> self_check -> confidence_gate."""

import logging

from agent.nodes import critique_node, plan_node, rerank_node, retrieve_node
from agent.nodes_gen import confidence_gate_node, generate_node, self_check_node
from agent.state import AgentState

logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(message)s")

QUESTION = "How do I add short-term memory to a LangGraph agent?"

if __name__ == "__main__":
    state: AgentState = {"question": QUESTION}
    print(f"Question: {QUESTION}\n")

    state.update(plan_node(state))
    state.update(retrieve_node(state))
    state.update(rerank_node(state))
    state.update(critique_node(state))
    state.update(generate_node(state))
    state.update(self_check_node(state))
    state.update(confidence_gate_node(state))

    print("=== ANSWER ===")
    print(state["answer"])
    print()

    print("=== CITATIONS ===")
    for url in state.get("citations", []):
        print(f"  {url}")
    print()

    print("=== VERDICT ===")
    print(f"  self_check_passed:  {state['self_check_passed']}")
    print(f"  confidence:         {state['confidence']}")
    print(f"  needs_human_review: {state['needs_human_review']}")
    print()

    print("=== TRACE ===")
    for line in state["trace"]:
        print(f"  {line}")
