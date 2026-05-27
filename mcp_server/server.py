"""LangGraph Brain MCP server.

Exposes the trust-scored knowledge base as MCP tools, resources, and prompts.
Run with:  python -m mcp_server.server
"""

from __future__ import annotations

from fastmcp import FastMCP

from mcp_server.retrieval import get_chunk, search, source_summary

mcp = FastMCP(
    name="langgraph-brain",
    instructions=(
        "Knowledge base over the LangGraph open source repo: issues, pull "
        "requests, and official docs. Every result carries a trust score "
        "(docs highest, open issues lowest). Use search_brain to find "
        "relevant context, get_doc to fetch a full chunk by id, and "
        "list_sources to see what is indexed."
    ),
)


# ---------- Tools ----------


@mcp.tool
def search_brain(
    query: str,
    top_k: int = 10,
    min_trust: float = 0.0,
    source_types: list[str] | None = None,
) -> list[dict]:
    """Semantic search over the LangGraph knowledge base.

    Args:
        query: Natural language question or keywords.
        top_k: How many results to return (1-50).
        min_trust: Minimum trust score, 0.0 to 1.0. Use 0.7+ for docs-only quality.
        source_types: Optional filter, e.g. ["doc"], ["issue", "pr"].

    Returns:
        A list of chunks, each with chunk_id, title, text, source_type,
        source_url, trust_score, and vector_score.
    """
    top_k = max(1, min(top_k, 50))
    results = search(
        query=query,
        top_k=top_k,
        min_trust=min_trust if min_trust > 0 else None,
        source_types=source_types or None,
    )
    return results


@mcp.tool
def get_doc(chunk_id: str) -> dict | None:
    """Fetch the full content and metadata of a single chunk by its chunk_id.

    Args:
        chunk_id: The id returned by search_brain, e.g. "doc-overview-2-0".

    Returns:
        The chunk with full text and metadata, or null if not found.
    """
    return get_chunk(chunk_id)


@mcp.tool
def list_sources() -> dict:
    """Return an overview of what is indexed: total chunks and counts per source type."""
    return source_summary()


# ---------- Resources ----------


@mcp.resource("brain://chunk/{chunk_id}")
def chunk_resource(chunk_id: str) -> str:
    """Expose a chunk as a readable resource."""
    chunk = get_chunk(chunk_id)
    if not chunk:
        return f"Chunk {chunk_id} not found."
    meta = chunk.get("metadata", {})
    return (
        f"# {chunk['title']}\n\n"
        f"Source: {chunk['source_type']} | Trust: {chunk['trust_score']}\n"
        f"URL: {chunk['source_url']}\n"
        f"Metadata: {meta}\n\n"
        f"---\n\n{chunk['text']}"
    )


# ---------- Prompts ----------


@mcp.prompt
def explain_in_context(topic: str) -> str:
    """A prompt template that asks for an explanation grounded in the brain."""
    return (
        f"Search the LangGraph brain for '{topic}'. Using only the retrieved "
        f"chunks, explain {topic} clearly. Cite each source URL you rely on. "
        f"If the brain does not contain enough information, say so explicitly."
    )


@mcp.prompt
def debug_issue(error_message: str) -> str:
    """A prompt template for troubleshooting a LangGraph error using the brain."""
    return (
        f"A user is hitting this LangGraph error:\n\n{error_message}\n\n"
        f"Search the brain for related issues and docs. Prioritize chunks with "
        f"high trust scores and resolved/closed issues. Summarize the likely "
        f"cause and the fix, citing source URLs."
    )


if __name__ == "__main__":
    mcp.run()
