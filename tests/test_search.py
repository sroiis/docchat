"""Tests for hybrid retrieval: BM25 lexical index, RRF fusion, search modes.

Covers the pure components (BM25, RRF) plus the engine-level dispatch. A
deterministic fake "neural" embedder exercises the real hybrid path without
torch; the real sentence-transformers test is opt-in (see the marker below).
"""

import importlib.util
import os
import zlib

import numpy as np
import pytest

from app.embeddings import tokenize
from app.lexical import LexicalIndex
from app.rag import RagEngine
from app.search import rrf_fuse

K1 = 1.5


class FakeNeuralEmbedder:
    """Deterministic stand-in for a dense embedder (name says 'openai' so the
    engine treats it as neural and enables the hybrid branch). Uses crc32 for
    stable token hashing so results don't depend on process hash seed."""

    name = "openai"

    def __init__(self) -> None:
        rng = np.random.default_rng(0)
        self._table = rng.normal(size=(64, 32))  # token-hash -> vector

    def _vec(self, token: str) -> np.ndarray:
        return self._table[zlib.crc32(token.encode("utf-8")) % 64]

    def transform(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            vec = np.zeros(32)
            for token in set(tokenize(text)):
                vec += self._vec(token)
            norm = np.linalg.norm(vec)
            vectors.append(vec / norm if norm else vec)
        return np.vstack(vectors)


@pytest.fixture
def small_corpus():
    return {
        "payments.md": "A settlement moves the funds to the merchant bank "
                       "account on the next business day.",
        "cooking.md": "To bake bread you need flour, water, yeast and salt.",
    }


# ---- BM25 -------------------------------------------------------------------


def test_bm25_ranks_matching_doc_first(small_corpus):
    chunks = []
    for doc_id, text in small_corpus.items():
        from app.chunking import chunk_text

        chunks.extend(chunk_text(doc_id, text))

    index = LexicalIndex()
    index.build(chunks)
    hits = index.search("settlement merchant bank", k=2)
    assert hits
    assert hits[0].chunk.doc_id == "payments.md"


def test_bm25_empty_corpus_and_missing_terms():
    index = LexicalIndex()
    index.build([])
    assert index.search("anything", k=4) == []

    chunks = []
    from app.chunking import chunk_text

    chunks.extend(chunk_text("d.md", "alpha beta gamma delta"))
    index.build(chunks)
    assert index.search("zzzqwq") == []  # unknown term -> no hits, no crash


# ---- RRF --------------------------------------------------------------------


def test_rrf_rewards_rank_agreement():
    # b ranks well in both lists; a and c only in one.
    fused = rrf_fuse(["a", "b"], ["b", "c"])
    assert fused == ["b", "a", "c"]


def test_rrf_keeps_chunks_only_in_one_list():
    fused = rrf_fuse(["x", "y"], ["y"])
    assert fused[0] == "y"
    assert "x" in fused


# ---- engine dispatch --------------------------------------------------------


def test_hybrid_fuses_dense_and_bm25(small_corpus):
    engine = RagEngine(embedder=FakeNeuralEmbedder())
    engine.ingest_texts(1, small_corpus)

    k = 4
    sparse = [h["chunk_id"] for h in engine._lexical_hits("settlement", k)]
    dense = [h["chunk_id"] for h in engine._dense_hits("settlement", k)]
    expected = rrf_fuse(dense, sparse)[:k]

    hits = engine.retrieve("settlement", k=k)
    assert [h["chunk_id"] for h in hits] == expected
    assert hits[0]["doc_id"] == "payments.md"


def test_hybrid_with_tfidf_is_unchanged(small_corpus):
    engine = RagEngine()  # default embedder = tfidf (not neural)
    engine.ingest_texts(1, small_corpus)
    ans = engine.ask(1, "How does money reach the merchant bank?", k=1)
    assert ans.sources[0]["doc_id"] == "payments.md"
    assert ans.confidence > 0


def test_sparse_mode_uses_bm25_only(monkeypatch, small_corpus):
    from app.config import settings

    monkeypatch.setattr(settings, "search_mode", "sparse")
    engine = RagEngine(embedder=FakeNeuralEmbedder())
    engine.ingest_texts(1, small_corpus)
    hits = engine.retrieve("settlement merchant", k=2)
    assert hits[0]["doc_id"] == "payments.md"


def test_dense_mode_uses_vectors_only(monkeypatch, small_corpus):
    from app.config import settings

    monkeypatch.setattr(settings, "search_mode", "dense")
    engine = RagEngine(embedder=FakeNeuralEmbedder())
    engine.ingest_texts(1, small_corpus)
    hits = engine.retrieve("settlement merchant", k=2)
    assert hits[0]["doc_id"] == "payments.md"


# ---- opt-in real neural test ------------------------------------------------

@pytest.mark.skipif(
    os.environ.get("DOCCHAT_RUN_NEURAL_TESTS") != "1"
    or importlib.util.find_spec("sentence_transformers") is None,
    reason="pip install -r requirements-optional.txt and set "
    "DOCCHAT_RUN_NEURAL_TESTS=1 to run the real neural embedding test",
)
def test_hybrid_neural_matches_synonym_pair():
    """The whole point of the dense branch: 'car' must match 'automobile'."""
    from app.embeddings import LocalEmbedder

    engine = RagEngine(embedder=LocalEmbedder())
    engine.ingest_texts(
        1,
        {
            "autos.md": "An automobile is the most common way to get around the city.",
            "cooking.md": "To bake bread you need flour, water, yeast and salt.",
        },
    )
    ans = engine.ask(1, "What is a car used for?", k=1)
    assert ans.sources[0]["doc_id"] == "autos.md"
