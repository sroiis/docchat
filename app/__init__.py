"""docchat — a production-grade 'chat with your documents' RAG service.

Read the files in this order to understand how Retrieval-Augmented Generation
(RAG) works:

    1. chunking.py    -> how we split documents into searchable pieces
    2. embeddings.py  -> how we turn text into vectors (numbers) we can compare
    3. store.py       -> where vectors live and how we search them
    4. rag.py         -> the pipeline that ties ingest + retrieval together
    5. db.py          -> SQLite persistence
    6. auth.py        -> JWT auth
    7. generator.py   -> the LLM "generation" half of RAG
    8. api.py         -> every HTTP endpoint under /api
    9. main.py        -> the FastAPI application wiring
"""

__version__ = "2.0.0"
