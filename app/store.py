"""The vector store: where chunk vectors live and how we search them.

In production you'd use a dedicated vector database (FAISS, pgvector, Pinecone,
Qdrant...). They all do the same core thing this class does — hold a matrix of
vectors and, given a query vector, return the nearest ones. We keep the whole
thing in a numpy matrix so you can see there's no magic: "nearest neighbour
search" is one matrix-multiply plus a sort.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chunking import Chunk


@dataclass
class SearchHit:
    chunk: Chunk
    score: float  # cosine similarity in [0, 1]; higher = more relevant


class VectorStore:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None  # shape (n_chunks, vector_dim)

    @property
    def size(self) -> int:
        return len(self._chunks)

    def reset(self) -> None:
        self._chunks = []
        self._matrix = None

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != vectors.shape[0]:
            raise ValueError("chunks and vectors count mismatch")
        self._chunks = list(chunks)
        self._matrix = vectors

    def search(self, query_vector: np.ndarray, k: int = 4) -> list[SearchHit]:
        """Return the top-k most similar chunks to `query_vector`.

        Because every vector is L2-normalised, the dot product IS the cosine
        similarity. `matrix @ query` computes it against all chunks at once.
        """
        if self._matrix is None or self.size == 0:
            return []

        scores = self._matrix @ query_vector.ravel()  # (n_chunks,)
        # argsort ascending, take the tail, reverse -> highest scores first.
        top_idx = np.argsort(scores)[::-1][:k]
        return [
            SearchHit(chunk=self._chunks[i], score=float(scores[i]))
            for i in top_idx
            if scores[i] > 0.0  # drop chunks with zero overlap
        ]

    def documents(self) -> dict[str, int]:
        """Map of doc_id -> number of chunks, for the /docs endpoint."""
        counts: dict[str, int] = {}
        for c in self._chunks:
            counts[c.doc_id] = counts.get(c.doc_id, 0) + 1
        return counts
