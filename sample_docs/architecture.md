# docchat architecture (for backend engineers)

docchat is intentionally shaped like a normal backend service so the AI parts
feel like ordinary dependencies.

## Components

- Chunking (chunking.py): splits documents into overlapping word windows so
  search can return a specific passage rather than a whole file.
- Embedder (embeddings.py): turns text into vectors. Defined as an interface so
  the implementation (TF-IDF today) can be replaced without touching callers.
- Vector store (store.py): holds the chunk vectors in a numpy matrix and does
  nearest-neighbour search. In production this would be FAISS, pgvector, Qdrant,
  or a managed vector database.
- RAG engine (rag.py): orchestrates ingest and query. One instance is shared
  across requests.
- HTTP layer (main.py): FastAPI endpoints — /health, /docs, /ingest, /ask —
  plus a tiny built-in web UI at /.

## Request flow for /ask

1. The question text is embedded into a query vector.
2. The vector store computes similarity against every chunk via one matrix
   multiply, then sorts to get the top-k.
3. The top chunks are returned as the answer with their source and score.

## Where an LLM would plug in

The only missing piece for full RAG is a final generation step: take the top
chunks, put them in a prompt, and call an LLM to write a natural-language
answer. That is a single function you could add to rag.py without changing
retrieval, storage, or the API.
