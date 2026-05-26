"""50-question evaluation set for the LangGraph Brain agent.

Each item has:
  question  - what we ask the agent
  reference - a short ground-truth answer (used by RAGAS for context recall
              and answer correctness)

Questions span docs, issues, and PRs, at mixed difficulty.
"""

from __future__ import annotations

EVAL_SET: list[dict[str, str]] = [
    {
        "question": "How do I add short-term memory to a LangGraph agent?",
        "reference": "Short-term memory is added as part of the agent's state, "
        "using a checkpointer so state persists across turns in a thread.",
    },
    {
        "question": "What is a checkpointer in LangGraph and why is it needed?",
        "reference": "A checkpointer saves graph state at each step, enabling "
        "persistence, memory across turns, and resuming after interrupts.",
    },
    {
        "question": "How do interrupts work in LangGraph?",
        "reference": "interrupt() pauses graph execution and surfaces a value to "
        "the caller; the graph resumes when a Command(resume=...) is sent.",
    },
    {
        "question": "What is the difference between the Graph API and the Functional API?",
        "reference": "The Graph API builds an explicit StateGraph of nodes and "
        "edges; the Functional API uses decorators on plain functions.",
    },
    {
        "question": "How do I stream outputs from a LangGraph graph?",
        "reference": "Use the graph's stream method with a stream_mode such as "
        "values, updates, or messages to get incremental output.",
    },
    {
        "question": "What is a StateGraph?",
        "reference": "A StateGraph is the core LangGraph builder where you define "
        "a shared state schema, add nodes, and connect them with edges.",
    },
    {
        "question": "How do I add conditional edges to a graph?",
        "reference": "Use add_conditional_edges with a router function that "
        "returns the name of the next node based on the current state.",
    },
    {
        "question": "What are subgraphs and when should I use them?",
        "reference": "Subgraphs are graphs used as nodes inside another graph, "
        "letting you compose and reuse multi-step logic.",
    },
    {
        "question": "How does LangGraph handle persistence across sessions?",
        "reference": "Persistence is handled by checkpointers that store state "
        "per thread id, so a thread can be resumed later.",
    },
    {
        "question": "What is the difference between short-term and long-term memory?",
        "reference": "Short-term memory is per-thread state within a conversation; "
        "long-term memory is stored across threads, often in a store.",
    },
    {
        "question": "How do I add human-in-the-loop review to a LangGraph agent?",
        "reference": "Use interrupt() inside a node to pause for human input, then "
        "resume with a Command carrying the human's decision.",
    },
    {
        "question": "What is the recursion limit and what happens when it is hit?",
        "reference": "The recursion limit caps how many steps a graph runs; "
        "exceeding it raises a GraphRecursionError.",
    },
    {
        "question": "How do I define the state schema for a graph?",
        "reference": "Define state as a TypedDict (or Pydantic model) describing "
        "the keys, optionally with reducer annotations for merging updates.",
    },
    {
        "question": "What are reducers in LangGraph state?",
        "reference": "Reducers are functions that define how updates to a state "
        "key are merged, such as appending to a list instead of replacing it.",
    },
    {
        "question": "How can I add tools to a LangGraph agent?",
        "reference": "Bind tools to the model and route to a tool-executing node; "
        "prebuilt helpers like ToolNode handle tool calls.",
    },
    {
        "question": "What is time travel in LangGraph?",
        "reference": "Time travel uses checkpoints to inspect or rewind to a past "
        "state of the graph and resume execution from there.",
    },
    {
        "question": "How do I run a LangGraph application locally?",
        "reference": "Use the LangGraph CLI or local server to run the graph and "
        "test it through a local development environment.",
    },
    {
        "question": "What observability options does LangGraph integrate with?",
        "reference": "LangGraph integrates with LangSmith for tracing, debugging, "
        "and monitoring graph runs.",
    },
    {
        "question": "How do I update graph state manually?",
        "reference": "Use the graph's update_state method with a thread config to "
        "write new values into the checkpointed state.",
    },
    {
        "question": "What is the START node in a graph?",
        "reference": "START is the virtual entry point; an edge from START to a "
        "node defines where graph execution begins.",
    },
    {
        "question": "What is the END node?",
        "reference": "END is the virtual terminal node; routing to END finishes "
        "the graph run.",
    },
    {
        "question": "How do I handle errors or retries inside a node?",
        "reference": "LangGraph supports retry policies on nodes, and you can also "
        "handle errors in node code and route accordingly.",
    },
    {
        "question": "Can a LangGraph node be asynchronous?",
        "reference": "Yes, nodes can be async functions; LangGraph supports async "
        "invocation and streaming.",
    },
    {
        "question": "How do I pass configuration into a graph at runtime?",
        "reference": "Pass a config dict with a configurable section; nodes can "
        "read runtime values from the RunnableConfig.",
    },
    {
        "question": "What is a thread id used for?",
        "reference": "A thread id groups checkpoints for one conversation so state "
        "persists and can be resumed for that thread.",
    },
    {
        "question": "How do I build a RAG agent with LangGraph?",
        "reference": "Build a graph that retrieves documents, optionally grades "
        "them, and generates an answer, looping if retrieval is insufficient.",
    },
    {
        "question": "What is the difference between invoke and stream?",
        "reference": "invoke runs the graph and returns the final result; stream "
        "yields intermediate output as the graph runs.",
    },
    {
        "question": "How do I visualize a LangGraph graph?",
        "reference": "Compiled graphs can be drawn as a diagram, for example as "
        "a Mermaid or PNG representation of nodes and edges.",
    },
    {
        "question": "What is durable execution in LangGraph?",
        "reference": "Durable execution means graph progress is checkpointed so a "
        "run can survive failures and resume from the last step.",
    },
    {
        "question": "How does LangGraph support multi-agent systems?",
        "reference": "Multiple agents can be composed as nodes or subgraphs that "
        "hand off control by routing through shared state.",
    },
    {
        "question": "What is a ToolNode?",
        "reference": "ToolNode is a prebuilt node that executes tool calls "
        "produced by the model and returns the results to the graph.",
    },
    {
        "question": "How do I add a system prompt to an agent?",
        "reference": "Include a system message in the messages passed to the "
        "model, or configure it on the model or prebuilt agent.",
    },
    {
        "question": "What is the create_react_agent prebuilt?",
        "reference": "create_react_agent is a prebuilt that builds a tool-calling "
        "ReAct-style agent graph from a model and a set of tools.",
    },
    {
        "question": "How do I limit how many times a retrieval loop retries?",
        "reference": "Track a retry count in state and use a conditional edge to "
        "stop looping once a maximum is reached.",
    },
    {
        "question": "What happens to state between graph runs without a checkpointer?",
        "reference": "Without a checkpointer, state is not persisted; each run "
        "starts fresh with no memory of previous runs.",
    },
    {
        "question": "How do I store data that outlives a single thread?",
        "reference": "Use a long-term memory store, which keeps data across "
        "threads rather than only within one conversation.",
    },
    {
        "question": "Can I nest a compiled graph as a node in another graph?",
        "reference": "Yes, a compiled graph can be added as a node, making it a "
        "subgraph within a larger graph.",
    },
    {
        "question": "How does streaming of LLM tokens work in LangGraph?",
        "reference": "Use the messages stream mode to receive LLM tokens as they "
        "are generated by model nodes.",
    },
    {
        "question": "What is the purpose of the checkpointer's thread config?",
        "reference": "The thread config identifies which thread's checkpoints to "
        "read and write, scoping persistence to that conversation.",
    },
    {
        "question": "How do I resume a graph after an interrupt?",
        "reference": "Invoke the graph again with a Command(resume=value); the "
        "value is returned by the interrupt call that paused it.",
    },
    {
        "question": "What is a node in LangGraph?",
        "reference": "A node is a function that receives the state and returns a "
        "partial update to the state.",
    },
    {
        "question": "How do edges differ from conditional edges?",
        "reference": "A normal edge always goes to a fixed next node; a "
        "conditional edge chooses the next node via a router function.",
    },
    {
        "question": "How can I debug why a graph took a certain path?",
        "reference": "Inspect the state and the trace, or use LangSmith to view "
        "the full execution trace of each node.",
    },
    {
        "question": "What model providers can LangGraph work with?",
        "reference": "LangGraph is model-agnostic and works with any chat model "
        "integration, including local models via Ollama.",
    },
    {
        "question": "How do I keep a conversation context across multiple user turns?",
        "reference": "Use a checkpointer with a consistent thread id so each turn "
        "appends to the same persisted state.",
    },
    {
        "question": "What is the functional API entrypoint decorator?",
        "reference": "The functional API uses an entrypoint decorator to mark the "
        "main function and task decorators for steps.",
    },
    {
        "question": "How do I test a LangGraph agent?",
        "reference": "Run the graph with sample inputs and assert on outputs, and "
        "optionally use LangSmith datasets and evaluation.",
    },
    {
        "question": "Why might retrieval return irrelevant chunks?",
        "reference": "Poor query phrasing, weak embeddings, or missing reranking "
        "can cause irrelevant chunks; reranking improves precision.",
    },
    {
        "question": "What is the benefit of reranking retrieved results?",
        "reference": "Reranking scores query-document pairs directly, improving "
        "precision over embedding similarity alone.",
    },
    {
        "question": "How does a critique step improve a RAG agent?",
        "reference": "A critique step judges whether retrieved context is "
        "sufficient and can trigger a retry with a refined query.",
    },
]

assert len(EVAL_SET) == 50, f"expected 50 questions, got {len(EVAL_SET)}"