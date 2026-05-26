# LangGraph Brain

A question-answering system over the LangGraph repository (https://github.com/langchain-ai/langgraph). It indexes the repo's issues, pull requests, and documentation, and answers questions about LangGraph by retrieving relevant chunks and generating an answer with citations.

Retrieval is exposed as an MCP server. The answering logic is a LangGraph `StateGraph` with separate nodes for query planning, retrieval, reranking, critique, generation, and a self-check. Low-confidence answers are routed to a human-in-the-loop review step before being returned.

The system runs entirely on local infrastructure (Ollama, Qdrant, local embedding and reranker models). No external paid APIs are used.

## Architecture

```
LangGraph repo  (issues + pull requests + documentation)
        |
   Ingestion          GitHub API client; docs parsed from llms.txt
        |
   Cleaning           dedup, bot/signature removal, version + resolved tagging
        |
   Chunking           header-based for docs, title+body+comments for issues/PRs;
                      code blocks kept intact
        |
   Embedding          sentence-transformers/all-MiniLM-L6-v2  (384-dim)
        |
   Storage            Qdrant (vectors) + SQLite (metadata + trust score)
                      trust score = weighted(source_type, resolved,
                                              version_match, recency)
        |
   MCP Server         FastMCP
                      tools: search_brain, get_doc, list_sources
                      resources: chunks as readable items
                      prompts: reusable templates
        |
   LangGraph Agent    plan -> retrieve (MCP) -> rerank (bge-reranker-v2-m3)
                      -> re-sort by rerank_score x trust_score -> top 4 chunks
                      -> critique -> retry up to 2x with a refined query
                      -> generate answer with citations
                      -> self-check -> confidence gate
                      -> interrupt for human review if confidence is low
                      -> final answer
```

### Components

The corpus is the last 12 months of the LangGraph repo: 788 issues, 2,199 pull
requests, and 425 documentation pages, chunked into 17,116 passages. Issues and
PRs are chunked as title + body + per-comment; docs are split on headers. Code
blocks are never split mid-block.

Each chunk carries a trust score computed from its source type (docs rank above
PRs above issues), whether the issue/PR was resolved, whether it matches the
queried LangGraph version, and recency. Bot-authored content is penalised.

The agent is an 8-node `StateGraph`. The critique node inspects the retrieved
chunks and, if they look insufficient, refines the query and retries (up to
twice). The self-check node inspects the generated answer against its sources
and assigns a confidence score; below a threshold the graph interrupts and
surfaces the draft for human review before returning.

Query planning, critique, and self-check run on a smaller model
(`llama3.2:3b`); answer generation runs on `llama3.1:8b`. Splitting the work
this way keeps total latency manageable on CPU.

## Stack

| Layer | Tool |
|---|---|
| Language | Python 3.12 |
| Dependency management | uv |
| Containerization | Docker (Qdrant) |
| Vector DB | Qdrant |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Reranker | BAAI/bge-reranker-v2-m3 (cross-encoder) |
| MCP server | FastMCP |
| Agent framework | LangGraph 1.0 |
| Model runtime | Ollama |
| Generation model | llama3.1:8b |
| Reasoning model | llama3.2:3b |
| Metadata store | SQLite |

## Evaluation

The agent was evaluated on a 50-question test set covering LangGraph concepts,
each question paired with a hand-written reference answer. To isolate the effect
of each component, the agent was run in four configurations, each adding one
component over the previous, and all four were scored with the same metrics.

| Version | Components | Context relevance | Answer similarity | Citation rate | Grounding rate | Human-review rate |
|---|---|---|---|---|---|---|
| v0 | vector search only | 0.392 | 0.471 | 76% | 50% | 74% |
| v1 | + reranker | 0.519 | 0.554 | 90% | 60% | 60% |
| v2 | + trust blend | 0.487 | 0.535 | 80% | 60% | 70% |
| v3 | + critique loop | 0.513 | 0.564 | 92% | 62% | 56% |

Context relevance is the mean cosine similarity between the question and its
retrieved chunks. Answer similarity is cosine similarity between the generated
answer and the reference answer. Grounding rate is the share of answers the
self-check passed; human-review rate is the share routed to the confidence gate.
All embeddings use the MiniLM model above.

Notes on the results:

- The reranker accounts for the largest single change. Context relevance rises
  0.127 from v0 to v1, with corresponding gains in citation and grounding rates.
- The critique loop (v3) produces the best answer similarity and citation rate
  and the lowest human-review rate. It triggered 46 retries across the 50
  questions.
- Trust-weighted reranking (v2) scores slightly below v1 on context relevance.
  Trust scoring biases retrieval toward authoritative sources over closest
  semantic match, and the relevance metric only measures the latter. It is
  retained because it matters when sources disagree, which this test set does
  not stress.

The absolute scores are bounded by the use of an 8B generation model on CPU.
The version-over-version deltas measure the components; the absolute level
reflects the model. A larger generation model would be expected to raise the
absolute numbers without changing the architecture.

Metrics are deterministic (semantic similarity and rate-based) rather than
LLM-judged. A local judge model would introduce variance that is hard to
quantify; an LLM-judge evaluation with a strong judge model is left as future
work.

## Repository layout

```
ingestion/     GitHub + docs ingestion, cleaning, chunking
storage/       Qdrant vector store, SQLite metadata, trust scoring
mcp_server/    FastMCP server exposing retrieval as MCP tools
agent/         LangGraph agent: state, nodes, graph, reranker, prompts
eval/          test set, comparison harness, scoring
scripts/       runnable entry points and smoke tests
```

## Running it

Requirements: Python 3.12, uv, Docker, and Ollama with `llama3.1:8b`,
`llama3.2:3b`, and `nomic-embed-text` pulled.

```bash
uv sync
docker compose up -d

# build the index
python -m scripts.ingest_issues
python -m scripts.ingest_prs
python -m scripts.ingest_docs
python -m scripts.process_all
python -m scripts.build_index

# ask a question
python -m scripts.run_agent

# run the v0-v3 evaluation
python -m eval.run_comparison
python -m eval.score_comparison
```
