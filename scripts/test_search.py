"""Smoke test: embed a query, search Qdrant, look up text in SQLite, print top results."""

from sentence_transformers import SentenceTransformer

from storage.metadata_db import connect, get_chunks
from storage.vector_store import VectorStore

QUERY = "how do I add memory to a langgraph agent"

if __name__ == "__main__":
    print(f"Query: {QUERY}\n")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    qvec = model.encode([QUERY], normalize_embeddings=True)[0].tolist()

    vs = VectorStore()
    hits = vs.search(qvec, top_k=3)

    conn = connect()
    chunks = get_chunks(conn, [h["chunk_id"] for h in hits])

    for h, c in zip(hits, chunks):
        print(f"--- score={h['score']:.3f}  trust={c['trust_score']:.2f}  {c['source_type']} ---")
        print(c["title"])
        print(c["text"][:300].replace("\n", " "))
        print(f"URL: {c['source_url']}\n")