#!/usr/bin/env python
"""Reproduce the numbers in docs/experiments.md.

Builds a TF-IDF index from sample_docs/ with the exact chunking + embedding +
vector-search code the app uses at runtime, then evaluates retrieval on a small
labelled query set (precision@k, hit@1) and measures latency.

Usage, from the repo root:

    python scripts/eval_rag.py
"""

import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

from app.chunking import chunk_text  # noqa: E402
from app.embeddings import TfidfEmbedder  # noqa: E402
from app.store import VectorStore  # noqa: E402

SAMPLE_DIR = os.path.join(_ROOT, "sample_docs")

QUERIES = [
    ("What is RAG and how does its retrieval step work?", "what_is_rag.md"),
    ("What are the two halves of a RAG system?", "what_is_rag.md"),
    ("How does TF-IDF decide which words matter?", "embeddings_and_tfidf.md"),
    ("What is cosine similarity and why does L2 normalisation matter?", "embeddings_and_tfidf.md"),
    ("How do neural embeddings differ from TF-IDF?", "embeddings_and_tfidf.md"),
    ("How does the vector store find the nearest chunks?", "architecture.md"),
    ("What components make up the docchat backend?", "architecture.md"),
    ("Where would an LLM plug into docchat?", "architecture.md"),
]

K = 4


def load_docs(directory: str) -> dict[str, str]:
    docs: dict[str, str] = {}
    for name in sorted(os.listdir(directory)):
        if name.lower().endswith((".md", ".txt")):
            with open(os.path.join(directory, name), encoding="utf-8", errors="replace") as fh:
                docs[name] = fh.read()
    return docs


def build_index(docs: dict[str, str]):
    chunks = []
    for doc_id, text in docs.items():
        chunks.extend(chunk_text(doc_id, text))

    embedder = TfidfEmbedder()
    t0 = time.perf_counter()
    texts = [c.text for c in chunks]
    embedder.fit(texts)
    vectors = embedder.transform(texts)
    build_s = time.perf_counter() - t0

    store = VectorStore()
    store.add(chunks, vectors)
    return chunks, embedder, store, build_s


def main() -> int:
    docs = load_docs(SAMPLE_DIR)
    chunks, embedder, store, build_s = build_index(docs)

    rows = []
    latencies_ms = []
    for question, expected in QUERIES:
        t0 = time.perf_counter()
        query_vec = embedder.transform([question])[0]
        hits = store.search(query_vec, k=K)
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(latency_ms)

        doc_ids = [h.chunk.doc_id for h in hits]
        correct = sum(1 for d in doc_ids if d == expected)
        rows.append(
            {
                "question": question,
                "expected": expected,
                "hit@1": doc_ids[0] if doc_ids else "-",
                "precision@k": correct / K,
                "latency_ms": latency_ms,
            }
        )

    print(f"OS                : {sys.platform}")
    print(f"Python            : {sys.version.split()[0]}")
    print(f"numpy             : {np.__version__}")
    print("")
    print(f"sample_docs       : {len(docs)} files "
          f"({sum(len(t.split()) for t in docs.values())} words)")
    print(f"chunks            : {store.size} (window=120, overlap=30)")
    print(f"vocabulary        : {embedder.vocab.__len__()} terms")
    print(f"vector dim        : {len(embedder.idf)}")
    print(f"index build       : {build_s * 1000:.1f} ms  (fit + transform)")
    print("")
    print(f"{'query':<70} {'expected':<24} {'hit@1':<24} {'P@4':<5} {'ms':<7}")
    print("-" * 132)
    for r in rows:
        print(
            f"{r['question'][:70]:<70} "
            f"{r['expected']:<24} "
            f"{r['hit@1'][:24]:<24} "
            f"{r['precision@k']:<5.1f} "
            f"{r['latency_ms']:<7.1f}"
        )

    p_at_k = np.mean([r["precision@k"] for r in rows])
    hit_at_1 = np.mean([1.0 if r["hit@1"] == r["expected"] else 0.0 for r in rows])
    lat = np.array(latencies_ms)
    print("-" * 132)
    print(f"mean precision@4 : {p_at_k:.2f}")
    print(f"hit@1 rate       : {hit_at_1:.2f}")
    print(f"mean latency     : {lat.mean():.1f} ms")
    print(f"p50 / p95        : {np.percentile(lat, 50):.1f} / {np.percentile(lat, 95):.1f} ms")
    print(f"search op        : {store.size} x {len(embedder.idf)} dot product per query")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
