"""Turn text into vectors (numbers) so we can measure similarity.

THE KEY IDEA behind all "AI search": you can't compare two strings with `==`
and expect to find *relevant* text. Instead you convert each piece of text into
a vector — a list of numbers — such that texts about similar topics end up with
similar vectors. Then "find relevant text" becomes "find the nearest vectors",
which is just geometry.

docchat ships with three embedding backends behind one interface:

  tfidf   -> classic, transparent, fully offline (default)
  openai  -> cloud embeddings via any OpenAI-compatible /embeddings endpoint
  local   -> sentence-transformers running on your own machine (offline neural)

The `Embedder` interface is the seam: swap the backend with
DOCCHAT_EMBEDDING_PROVIDER and nothing else in the app changes.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol

import httpx
import numpy as np

from .config import settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase and split into alphanumeric word tokens."""
    return _TOKEN_RE.findall(text.lower())


class Embedder(Protocol):
    """Anything that can turn text into fixed-meaning vectors."""

    name: str

    def fit(self, corpus: list[str]) -> None: ...
    def transform(self, texts: list[str]) -> np.ndarray: ...


class TfidfEmbedder:
    """A small, dependency-free TF-IDF vectorizer.

    Call `fit()` once on your whole corpus to learn the vocabulary and the
    IDF weights, then `transform()` any text (chunks or a user question) into
    L2-normalised vectors. Normalising means cosine similarity is just a dot
    product later — one less thing to get wrong.
    """

    name = "tfidf"

    def __init__(self, min_df: int = 1):
        self.min_df = min_df
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None

    def fit(self, corpus: list[str]) -> None:
        doc_freq: Counter[str] = Counter()
        tokenized_docs = [tokenize(doc) for doc in corpus]
        for tokens in tokenized_docs:
            for term in set(tokens):
                doc_freq[term] += 1

        self.vocab = {
            term: i
            for i, term in enumerate(
                sorted(t for t, df in doc_freq.items() if df >= self.min_df)
            )
        }

        n_docs = max(len(corpus), 1)
        idf = np.zeros(len(self.vocab), dtype=np.float64)
        for term, idx in self.vocab.items():
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
                    continue
                tf = count / total
                vectors[row, idx] = tf * self.idf[idx]

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


class OpenAIEmbedder:
    """Cloud embeddings through any OpenAI-compatible `/embeddings` endpoint.

    `fit()` is a no-op — cloud models need no corpus training — which is why
    a persistent index matters: we store the vectors so restarts don't
    re-bill you for the same documents.
    """

    name = "openai"

    def __init__(
        self,
        api_key: str = settings.openai_api_key,
        base_url: str = settings.openai_base_url,
        model: str = settings.embedding_model,
    ):
        if not api_key:
            raise ValueError("DOCCHAT_OPENAI_API_KEY is required for openai embeddings")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    def fit(self, corpus: list[str]) -> None:
        return None

    def transform(self, texts: list[str]) -> np.ndarray:
        vectors: list[np.ndarray] = []
        # Batch in chunks of 100 to stay under typical request limits.
        for start in range(0, len(texts), 100):
            batch = texts[start : start + 100]
            resp = httpx.post(
                f"{self.base_url}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "input": batch},
                timeout=60.0,
            )
            resp.raise_for_status()
            data = resp.json()["data"]
            data.sort(key=lambda d: d["index"])
            vectors.extend(np.array(d["embedding"], dtype=np.float64) for d in data)

        matrix = np.vstack(vectors) if vectors else np.zeros((0, 0))
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


class LocalEmbedder:
    """Offline neural embeddings via sentence-transformers.

    Imported lazily so the heavy torch dependency is only loaded when this
    provider is actually selected. Install extras first:
        pip install -r requirements-optional.txt
    """

    name = "local"

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model = SentenceTransformer(model)

    def fit(self, corpus: list[str]) -> None:
        return None

    def transform(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0))
        return self._model.encode(texts, normalize_embeddings=True)


def build_embedder() -> Embedder:
    """Instantiate the embedder selected by DOCCHAT_EMBEDDING_PROVIDER."""
    provider = settings.embedding_provider.lower()
    if provider == "tfidf":
        return TfidfEmbedder()
    if provider == "openai":
        return OpenAIEmbedder()
    if provider == "local":
        return LocalEmbedder()
    raise ValueError(
        f"Unknown embedding provider {provider!r}; choose tfidf | openai | local"
    )
