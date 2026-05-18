# LangGraph Brain

Agentic RAG layer over the LangGraph open source repo. Ingests issues, PRs, and docs into a trust-scored vector store, serves them via an MCP server, and answers questions through a LangGraph agent that plans, retrieves, reranks, critiques, self-checks, and pauses for human review when confidence is low.

## Architecture

\`\`\`
LangGraph repo (issues + PRs + docs)
  ↓
Ingestion (Python, GitHub API, markdown parser)
  ↓
Cleaning (dedup, bot/signature removal, version tagging, resolved status)
  ↓
Chunking (header-based for docs, Q+A pairs for issues, code blocks preserved)
  ↓
Embedding (Voyage voyage-3-lite)
  ↓
Storage: Qdrant (vectors) + SQLite (metadata, trust score, permissions)
       Trust score = weighted(recency + resolved + version_match + source_type)
  ↓
MCP Server (FastMCP 3.x)
   ├── Tools: search_brain, get_doc, list_sources
   ├── Resources: individual docs as readable items
   └── Prompts: reusable templates
  ↓
LangGraph Agent (LangGraph 1.0)
   Plan → Retrieve via MCP → Rerank (bge-reranker-v2-m3)
        → Re-sort by (rerank_score × trust_score) → top 3-5 chunks
        → Critique → retry up to 2x with refined query
        → Generate answer with citations
        → Self-check
        → Confidence check → if low, INTERRUPT for human review
        → Return final answer
  ↓
Observability: LangSmith (traces) + Langfuse (datasets + experiments)
Evaluation: RAGAS (faithfulness, context precision, context recall, answer relevance)
\`\`\`

## Stack

| Layer | Tool |
|---|---|
| Vector DB | Qdrant |
| Embeddings | Voyage voyage-3-lite |
| Reranker | bge-reranker-v2-m3 |
| MCP server | FastMCP 3.x |
| Agent framework | LangGraph 1.0 |
| Agent observability | LangSmith |
| Datasets / experiments | Langfuse |
| Eval metrics | RAGAS |

## Evaluation results

_Filled in at Week 3. Version-over-version comparison across 50 questions._

| Version | Faithfulness | Context Precision | Context Recall | Answer Relevance |
|---|---|---|---|---|
| v0 baseline | TBD | TBD | TBD | TBD |
| v1 + rerank | TBD | TBD | TBD | TBD |
| v2 + trust score | TBD | TBD | TBD | TBD |
| v3 + critique loop | TBD | TBD | TBD | TBD |

## Status

Week 1 — repo scaffold in place. Ingestion next.