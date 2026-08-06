"""docchat — a fully-offline 'chat with your docs' service.

The package is deliberately small and readable. If you're new to AI, read the
files in this order to understand how Retrieval-Augmented Generation (RAG) works:

    1. chunking.py    -> how we split documents into searchable pieces
    2. embeddings.py  -> how we turn text into vectors (numbers) we can compare
    3. store.py       -> where vectors live and how we search them
    4. rag.py         -> the pipeline that ties ingest + retrieval together
    5. main.py        -> the FastAPI HTTP layer on top of all of it
"""

__version__ = "1.0.0"
