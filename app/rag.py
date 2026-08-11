"""The RAG pipeline: ingest documents, then answer questions from them.

RAG = Retrieval-Augmented Generation. In a full system the flow is:

    question -> RETRIEVE relevant chunks -> feed them to an LLM -> GENERATE answer

This build does both halves:

  - retrieval: chunking + embeddings + nearest-neighbour search (the original
    docchat core), and
  - generation: an optional LLM step (Ollama or any OpenAI-compatible API)
    that writes a fluent answer grounded in the retrieved chunks.

The engine is per-user: each user has their own embedder, vector store and
persistent index in the database. Everything below is user-scoped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import db
from .chunking import Chunk, chunk_text
from .config import settings
from .embeddings import Embedder, build_embedder
from .generator import Generator, build_generator
from .lexical import LexicalIndex
from .search import rrf_fuse
from .store import VectorStore


@dataclass
class Answer:
    question: str
    answer: str                 # human-readable synthesis of the top hits
    sources: list[dict]         # [{doc_id, chunk_id, score, text}, ...]
    confidence: float           # score of the single best hit


class RagEngine:
    """Owns the embedder + store + generator for a single user.

    An instance is created per user and cached; it is rebuilt whenever the
    user's index changes (ingest or delete).
    """

    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder: Embedder = embedder or build_embedder()
        self.store = VectorStore()
        self.lexical: LexicalIndex | None = None
        self.generator: Generator | None = None
        self.last_answer: Answer | None = None
        self._fitted = False

    # ---- index lifecycle ---------------------------------------------------

    def load(self, user_id: int) -> None:
        """Restore a user's index from the persistent database."""
        rows = db.load_chunks(user_id)
        if not rows:
            self.store.reset()
            self.lexical = None
            self._fitted = False
            return

        chunks = [
            Chunk(doc_id=r["doc_id"], chunk_id=r["chunk_id"],
                  text=r["text"], order=r["order"])
            for r in rows
        ]
        vectors = np.vstack([r["vector"] for r in rows])
        self._fit_if_needed(chunks)
        self._build_lexical(chunks)
        self.store.add(chunks, vectors)
        self._fitted = True

    def ingest_texts(self, user_id: int, docs: dict[str, str]) -> int:
        """Index a mapping of {doc_id: full_text} for a user. Returns chunk count.

        A user's whole corpus is re-chunked and re-embedded on each ingest so
        TF-IDF sees the complete vocabulary (it must to compute IDF). For
        neural embedders this is still consistent, just slightly wasteful for
        large corpora — acceptable at this scale.
        """
        for doc_id, text in docs.items():
            if text is None or not text.strip():
                continue
            db.add_document(user_id, doc_id, file_name=doc_id, content=text)

        # Rebuild the index from every stored document, including the new ones.
        all_docs = db.list_all_docs(user_id)

        chunks: list[Chunk] = []
        for doc_id, text in all_docs.items():
            chunks.extend(chunk_text(doc_id, text))
        if not chunks:
            db.replace_chunks(user_id, [])
            self.store.reset()
            self.lexical = None
            self._fitted = False
            return 0

        texts = [c.text for c in chunks]
        self._fit_if_needed(chunks)
        self._build_lexical(chunks)
        vectors = self.embedder.transform(texts)
        self.store.add(chunks, vectors)
        db.replace_chunks(
            user_id,
            [
                {
                    "doc_id": c.doc_id,
                    "chunk_id": c.chunk_id,
                    "order": c.order,
                    "text": c.text,
                    "vector": v,
                }
                for c, v in zip(chunks, vectors)
            ],
        )
        self._fitted = True
        return len(chunks)

    def ingest_directory(self, user_id: int, path: str) -> int:
        """Read every .md/.txt file under `path` and index it for the user."""
        import os

        docs: dict[str, str] = {}
        for root, _dirs, files in os.walk(path):
            for name in files:
                if not name.lower().endswith((".md", ".txt")):
                    continue
                full = os.path.join(root, name)
                with open(full, encoding="utf-8", errors="replace") as fh:
                    docs[name] = fh.read()
        return self.ingest_texts(user_id, docs)

    def delete_document(self, user_id: int, doc_id: str) -> bool:
        """Remove a document and rebuild the index without it."""
        removed = db.delete_document(user_id, doc_id)
        if not removed:
            return False
        self.load(user_id)
        return True

    def documents(self, user_id: int) -> list[dict]:
        return db.list_documents(user_id)

    # ---- embedding helper ---------------------------------------------------

    def _fit_if_needed(self, chunks: list[Chunk]) -> None:
        """TF-IDF needs a corpus fit; neural embedders ignore it."""
        fit = getattr(self.embedder, "fit", None)
        if fit is not None:
            fit([c.text for c in chunks])

    def _build_lexical(self, chunks: list[Chunk]) -> None:
        """Always-available sparse index; pairs with the vector store for
        hybrid retrieval when the embedder is neural (local/openai)."""
        self.lexical = LexicalIndex()
        self.lexical.build(chunks)

    # ---- query -------------------------------------------------------------

    def retrieve(self, question: str, k: int = 4) -> list[dict]:
        """Return the top-k most relevant chunks with scores.

        Dispatches on DOCCHAT_SEARCH_MODE:

        - hybrid (default): fuse the neural vector hits and the BM25 hits
          with RRF. With a TF-IDF embedder there is no neural branch, so this
          is vector-only — identical to the pre-hybrid behaviour.
        - sparse: BM25 lexical search only.
        - dense: the configured vector store only (TF-IDF or neural).
        """
        if not self._fitted or self.store.size == 0:
            return []

        neural = self.embedder.name in ("openai", "local")
        mode = settings.search_mode.lower()
        has_lexical = self.lexical is not None and self.lexical.size > 0

        if mode == "hybrid" and neural and has_lexical:
            dense = self._dense_hits(question, k)
            sparse = self._lexical_hits(question, k)
            by_id = {**{h["chunk_id"]: h for h in sparse},
                     **{h["chunk_id"]: h for h in dense}}
            fused = rrf_fuse(
                [h["chunk_id"] for h in dense],
                [h["chunk_id"] for h in sparse],
            )
            return [by_id[chunk_id] for chunk_id in fused][:k]

        if mode == "sparse" and has_lexical:
            return self._lexical_hits(question, k)

        return self._dense_hits(question, k)

    def _dense_hits(self, question: str, k: int) -> list[dict]:
        query_vec = self.embedder.transform([question])[0]
        hits = self.store.search(query_vec, k=k)
        return [
            {
                "doc_id": h.chunk.doc_id,
                "chunk_id": h.chunk.chunk_id,
                "score": round(h.score, 4),
                "text": h.chunk.text,
            }
            for h in hits
        ]

    def _lexical_hits(self, question: str, k: int) -> list[dict]:
        hits = self.lexical.search(question, k=k)
        max_score = max((h.score for h in hits), default=0.0) or 1.0
        return [
            {
                "doc_id": h.chunk.doc_id,
                "chunk_id": h.chunk.chunk_id,
                "score": round(h.score / max_score, 4),
                "text": h.chunk.text,
            }
            for h in hits
        ]

    def ask(self, user_id: int, question: str, k: int = 4) -> Answer:
        """Retrieve, then generate (if an LLM is configured)."""
        self.load(user_id)
        sources = self.retrieve(question, k)
        if not sources:
            answer = Answer(
                question=question,
                answer="No documents indexed yet. "
                       "Ingest some docs first (POST /ingest).",
                sources=[],
                confidence=0.0,
            )
            self.last_answer = answer
            return answer

        best = sources[0]
        answer = Answer(
            question=question,
            answer=f"Based on '{best['doc_id']}' (relevance {best['score']:.0%}):\n\n"
                   f"{best['text']}",
            sources=sources,
            confidence=best["score"],
        )
        self.last_answer = answer
        return answer

    async def stream_answer(
        self, user_id: int, question: str, k: int = 4
    ) -> dict:
        """Retrieve sources and set up generation for streaming.

        Returns {"sources": [...], "generator": Generator} — the HTTP layer
        builds the prompt and streams tokens.
        """
        self.load(user_id)
        sources = self.retrieve(question, k)
        if not sources:
            return {"sources": [], "generator": None}

        best = sources[0]
        self.last_answer = Answer(
            question=question,
            answer=f"Based on '{best['doc_id']}' (relevance {best['score']:.0%}):\n\n"
                   f"{best['text']}",
            sources=sources,
            confidence=best["score"],
        )
        if self.generator is None:
            self.generator = build_generator(self)
        return {"sources": sources, "generator": self.generator}

    @staticmethod
    def build_prompt(question: str, sources: list[dict]) -> tuple[str, str]:
        """Turn retrieved chunks into system + user prompts for the LLM."""
        system = (
            "You are docchat, a precise Q&A assistant. Answer the user's "
            "question using ONLY the provided excerpts. If the excerpts do not "
            "contain the answer, say you couldn't find it. Cite the source "
            "filename of each fact you use, e.g. (source: architecture.md). "
            "Be concise and factual."
        )
        context = "\n\n".join(
            f"[{i + 1}] (source: {s['doc_id']})\n{s['text']}"
            for i, s in enumerate(sources)
        )
        user_prompt = (
            f"Question: {question}\n\n"
            f"Relevant excerpts:\n{context}\n\n"
            f"Answer the question using the excerpts above."
        )
        return system, user_prompt
