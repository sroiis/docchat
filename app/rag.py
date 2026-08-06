"""The RAG pipeline: ingest documents, then answer questions from them.

RAG = Retrieval-Augmented Generation. In a full system the flow is:

    question -> RETRIEVE relevant chunks -> feed them to an LLM -> GENERATE answer

This build is "retrieval-only" (you chose the fully-offline option), so we stop
after retrieval and return the best chunks *as* the answer, each with its source
and a confidence score. That's genuinely useful on its own, and it's the half of
RAG that backend engineers actually own — the LLM is just an optional last step
you could bolt on (see README for how).
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

from .chunking import chunk_text
from .embeddings import Embedder, TfidfEmbedder
from .store import VectorStore


@dataclass
class Answer:
    question: str
    answer: str                 # human-readable synthesis of the top hits
    sources: list[dict]         # [{doc_id, chunk_id, score, text}, ...]
    confidence: float           # score of the single best hit


class RagEngine:
    """Owns the embedder + store and exposes ingest/ask.

    One instance is created at app startup and shared across requests. It's
    read-mostly after ingest, so this is safe for the simple single-process
    server we run here.
    """

    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder: Embedder = embedder or TfidfEmbedder()
        self.store = VectorStore()
        self._fitted = False

    # ---- ingest -----------------------------------------------------------

    def ingest_texts(self, docs: dict[str, str]) -> int:
        """Index a mapping of {doc_id: full_text}. Returns chunk count.

        Note: TF-IDF must see the whole corpus to compute IDF, so we (re)fit on
        every ingest. That's fine for a local tool with a handful of docs. A
        neural embedder wouldn't need refitting — another reason the interface
        is nice.
        """
        chunks = []
        for doc_id, text in docs.items():
            chunks.extend(chunk_text(doc_id, text))

        if not chunks:
            self.store.reset()
            self._fitted = False
            return 0

        corpus = [c.text for c in chunks]
        self.embedder.fit(corpus)
        vectors = self.embedder.transform(corpus)
        self.store.add(chunks, vectors)
        self._fitted = True
        return len(chunks)

    def ingest_directory(self, path: str) -> int:
        """Read every .md/.txt file under `path` and index it."""
        docs: dict[str, str] = {}
        for root, _dirs, files in os.walk(path):
            for name in files:
                if not name.lower().endswith((".md", ".txt")):
                    continue
                full = os.path.join(root, name)
                with open(full, "r", encoding="utf-8", errors="replace") as fh:
                    docs[name] = fh.read()
        return self.ingest_texts(docs)

    # ---- query ------------------------------------------------------------

    def ask(self, question: str, k: int = 4) -> Answer:
        if not self._fitted or self.store.size == 0:
            return Answer(question=question, answer="No documents indexed yet. "
                          "Ingest some docs first (POST /ingest).",
                          sources=[], confidence=0.0)

        query_vec = self.embedder.transform([question])[0]
        hits = self.store.search(query_vec, k=k)

        if not hits:
            return Answer(question=question,
                          answer="I couldn't find anything relevant in the "
                                 "indexed documents for that question.",
                          sources=[], confidence=0.0)

        sources = [
            {
                "doc_id": h.chunk.doc_id,
                "chunk_id": h.chunk.chunk_id,
                "score": round(h.score, 4),
                "text": h.chunk.text,
            }
            for h in hits
        ]
        # The "answer" is a readable synthesis: lead with the strongest chunk.
        best = hits[0]
        answer = (
            f"Based on '{best.chunk.doc_id}' (relevance {best.score:.0%}):\n\n"
            f"{best.chunk.text}"
        )
        return Answer(
            question=question,
            answer=answer,
            sources=sources,
            confidence=round(best.score, 4),
        )
