"""Turn text into vectors (numbers) so we can measure similarity.

THE KEY IDEA behind all "AI search": you can't compare two strings with `==`
and expect to find *relevant* text. Instead you convert each piece of text into
a vector — a list of numbers — such that texts about similar topics end up with
similar vectors. Then "find relevant text" becomes "find the nearest vectors",
which is just geometry.

Here we use TF-IDF (Term Frequency - Inverse Document Frequency), a classic,
transparent way to build those vectors:

  - Term Frequency: words that appear a lot in a chunk are important *to that chunk*.
  - Inverse Document Frequency: words that appear in *every* chunk (like "the")
    are not distinctive, so we down-weight them.

Real systems swap TF-IDF for a neural "embedding model" that captures meaning
(so "car" and "automobile" score as similar). We deliberately don't, because:
  (a) it needs no downloads and runs instantly, and
  (b) you can read every line and see exactly why a result was returned.

The `Embedder` interface below is the seam where you'd plug in a real model
later (e.g. sentence-transformers) without changing anything else.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase and split into alphanumeric word tokens."""
    return _TOKEN_RE.findall(text.lower())


class Embedder(Protocol):
    """Anything that can turn text into fixed-meaning vectors.

    Swap this out for a neural embedding model and the rest of the app is
    unchanged — that's the whole point of coding to an interface.
    """

    def fit(self, corpus: list[str]) -> None: ...
    def transform(self, texts: list[str]) -> np.ndarray: ...


class TfidfEmbedder:
    """A small, dependency-free TF-IDF vectorizer.

    Call `fit()` once on your whole corpus to learn the vocabulary and the
    IDF weights, then `transform()` any text (chunks or a user question) into
    L2-normalised vectors. Normalising means cosine similarity is just a dot
    product later — one less thing to get wrong.
    """

    def __init__(self, min_df: int = 1):
        self.min_df = min_df
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None

    def fit(self, corpus: list[str]) -> None:
        # Count in how many documents each term appears (document frequency).
        doc_freq: Counter[str] = Counter()
        tokenized_docs = [tokenize(doc) for doc in corpus]
        for tokens in tokenized_docs:
            for term in set(tokens):
                doc_freq[term] += 1

        # Build the vocabulary, dropping ultra-rare terms if min_df > 1.
        self.vocab = {
            term: i
            for i, term in enumerate(
                sorted(t for t, df in doc_freq.items() if df >= self.min_df)
            )
        }

        n_docs = max(len(corpus), 1)
        idf = np.zeros(len(self.vocab), dtype=np.float64)
        for term, idx in self.vocab.items():
            # Smoothed IDF: rarer term -> higher weight.
            idf[idx] = math.log((1 + n_docs) / (1 + doc_freq[term])) + 1.0
        self.idf = idf

    def transform(self, texts: list[str]) -> np.ndarray:
        if self.idf is None:
            raise RuntimeError("call fit() before transform()")

        vectors = np.zeros((len(texts), len(self.vocab)), dtype=np.float64)
        for row, text in enumerate(texts):
            counts = Counter(tokenize(text))
            if not counts:
                continue
            total = sum(counts.values())
            for term, count in counts.items():
                idx = self.vocab.get(term)
                if idx is None:
                    continue  # a word we've never seen — ignore it
                tf = count / total
                vectors[row, idx] = tf * self.idf[idx]

        # L2-normalise each row so cosine similarity == dot product.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms
