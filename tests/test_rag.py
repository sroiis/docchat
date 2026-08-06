"""Tests you can actually run: `pytest -q`.

They double as documentation — read them to see how each piece behaves.
"""

from app.chunking import chunk_text
from app.embeddings import TfidfEmbedder, tokenize
from app.rag import RagEngine


def test_chunking_overlap_and_ids():
    text = " ".join(str(i) for i in range(300))  # 300 "words"
    chunks = chunk_text("doc.md", text, words_per_chunk=100, overlap=20)
    assert len(chunks) >= 3
    assert chunks[0].chunk_id == "doc.md#0"
    # Consecutive chunks should overlap (share trailing/leading words).
    first_words = set(chunks[0].text.split())
    second_words = set(chunks[1].text.split())
    assert first_words & second_words, "expected overlap between chunks"


def test_tokenize_lowercases_and_splits():
    assert tokenize("Hello, WORLD! 123") == ["hello", "world", "123"]


def test_tfidf_vectors_are_normalised():
    emb = TfidfEmbedder()
    emb.fit(["the cat sat", "the dog ran", "cats and dogs"])
    vecs = emb.transform(["the cat sat"])
    norm = (vecs[0] ** 2).sum() ** 0.5
    assert abs(norm - 1.0) < 1e-9  # L2-normalised


def test_ask_returns_relevant_source():
    engine = RagEngine()
    engine.ingest_texts(
        {
            "payments.md": "A settlement moves money to a merchant bank account.",
            "cooking.md": "To bake bread you need flour, water, yeast and salt.",
        }
    )
    ans = engine.ask("How does money reach a merchant?", k=1)
    assert ans.sources, "expected at least one source"
    assert ans.sources[0]["doc_id"] == "payments.md"
    assert ans.confidence > 0


def test_ask_before_ingest_is_graceful():
    engine = RagEngine()
    ans = engine.ask("anything?")
    assert ans.sources == []
    assert ans.confidence == 0.0
