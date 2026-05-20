"""Test the reranker: search returns 20 candidates, rerank picks the best 5."""

from agent.reranker import rerank
from mcp_server.retrieval import search

QUERY = "how do I add memory to a langgraph agent"

if __name__ == "__main__":
    print(f"Query: {QUERY}\n")

    candidates = search(QUERY, top_k=20)
    print(f"Vector search returned {len(candidates)} candidates\n")

    print("Top 5 by vector score (before rerank):")
    for c in candidates[:5]:
        print(f"  vec={c['vector_score']:.3f} trust={c['trust_score']:.2f} [{c['source_type']}] {c['title']}")

    reranked = rerank(QUERY, candidates, top_k=5, trust_weight=0.3)
    print("\nTop 5 after rerank (final_score = rerank_norm blended with trust):")
    for c in reranked:
        print(
            f"  final={c['final_score']:.3f} rerank={c['rerank_score']:.2f} "
            f"trust={c['trust_score']:.2f} [{c['source_type']}] {c['title']}"
        )