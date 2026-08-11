"""BM25 lexical index — the "sparse" half of hybrid retrieval.

TF-IDF-vector search is one way to match on words; BM25 is the ranking
function used by real search engines (Lucene, Elasticsearch). Both are
*lexical*: they can only match exact terms. The reason docchat keeps a
dedicated lexical index is so retrieval can *combine* it with a neural
(dense) index — synonyms never match lexically, but exact jargon and rare
terms match densely poorly. Fuse both lists and you get meaning plus recall.

This is a minimal, dependency-free BM25 implementation (only numpy for
vectorised scoring) following the standard Okapi formulation.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .chunking import Chunk
from .embeddings import tokenize

K1 = 1.5
B = 0.75


@dataclass
class LexicalHit:
    chunk: Chunk
    score: float  # raw BM25 score, larger = more relevant


class LexicalIndex:
    """Sparse per-term posterior index with BM25 scoring.

    Build once from the corpus, then `search()` any query text.
    """

    def __init__(self, k1: float = K1, b: float = B) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._postings: dict[str, list[tuple[int, int]]] = {}  # term -> [(doc_idx, tf)]
        self._idf: dict[str, float] = {}
        self._doc_len: np.ndarray | None = None
        self._avgdl = 0.0

    @property
    def size(self) -> int:
        return len(self._chunks)

    def build(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        n = len(chunks)
        if n == 0:
            self._postings, self._idf = {}, {}
            self._doc_len = np.zeros(0)
            self._avgdl = 0.0
            return

        df: Counter[str] = Counter()
        postings: dict[str, list[tuple[int, int]]] = {}
        doc_len = np.zeros(n, dtype=int)
        for i, chunk in enumerate(chunks):
            counts = Counter(tokenize(chunk.text))
            doc_len[i] = len(chunk.text.split())
            for term, tf in counts.items():
                postings.setdefault(term, []).append((i, tf))
                df[term] += 1

        self._postings = postings
        self._doc_len = doc_len
        self._avgdl = float(doc_len.mean())
        self._idf = {
            term: math.log(1.0 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def search(self, text: str, k: int = 4) -> list[LexicalHit]:
        """Return the top-k chunks by BM25 score for `text`."""
        if self.size == 0:
            return []

        query = Counter(tokenize(text))
        scores = np.zeros(self.size)
        for term, qtf in query.items():
            idf = self._idf.get(term)
            if idf is None:
                continue
            avgdl = self._avgdl
            for doc_idx, tf in self._postings.get(term, []):
                dl = self._doc_len[doc_idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / avgdl)
                scores[doc_idx] += idf * qtf * (tf * (self.k1 + 1.0)) / denom

        order = np.argsort(scores)[::-1][:k]
        return [
            LexicalHit(chunk=self._chunks[i], score=float(scores[i]))
            for i in order
            if scores[i] > 0.0
        ]
