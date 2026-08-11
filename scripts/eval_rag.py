#!/usr/bin/env python
"""Reproduce the numbers in docs/experiments.md.

Builds indexes from sample_docs/ with the exact chunking + embedding +
retrieval code the app uses at runtime, then evaluates retrieval on a small
labelled query set (precision@k, hit@1) and measures latency.

Usage, from the repo root:

    python scripts/eval_rag.py          # TF-IDF lexical baseline
    python scripts/eval_rag.py --local  # add dense & hybrid (needs optional extras)
"""

import argparse
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

from app.chunking import chunk_text  # noqa: E402
from app.embeddings import TfidfEmbedder  # noqa: E402
from app.lexical import LexicalIndex  # noqa: E402
from app.search import rrf_fuse  # noqa: E402
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


def chunks_from_docs(docs: dict[str, str]):
    chunks = []
    for doc_id, text in docs.items():
        chunks.extend(chunk_text(doc_id, text))
    return chunks


def build_tfidf(chunks):
    embedder = TfidfEmbedder()
    t0 = time.perf_counter()
    texts = [c.text for c in chunks]
    embedder.fit(texts)
    vectors = embedder.transform(texts)
    build_s = time.perf_counter() - t0
    store = VectorStore()
    store.add(chunks, vectors)
    return store, embedder, build_s


def build_local(chunks):
    """Neural (dense) index + BM25 (sparse) index for hybrid comparison."""
    from app.embeddings import LocalEmbedder

    embedder = LocalEmbedder()
    t0 = time.perf_counter()
    vectors = embedder.transform([c.text for c in chunks])
    build_s = time.perf_counter() - t0
    store = VectorStore()
    store.add(chunks, vectors)
    lexical = LexicalIndex()
    lexical.build(chunks)
    return store, lexical, embedder, build_s


def evaluate(hits_fn, question, expected, k: int = K) -> dict:
    t0 = time.perf_counter()
    hits = hits_fn(question, k)
    latency_ms = (time.perf_counter() - t0) * 1000
    doc_ids = [h.chunk.doc_id for h in hits]
    correct = sum(1 for d in doc_ids if d == expected)
    return {
        "question": question,
        "hit@1": doc_ids[0] if doc_ids else "-",
        "expected": expected,
        "precision@k": correct / k,
        "latency_ms": latency_ms,
    }


def dense_hits(store, embedder):
    def _fn(question, k):
        qv = embedder.transform([question])[0]
        return store.search(qv, k=k)

    return _fn


def hybrid_hits(store, embedder, lexical):
    k_dense = 8  # retrieve more from each branch so RRF has material to fuse
    def _fn(question, k):
        qv = embedder.transform([question])[0]
        dense_rank = store.search(qv, k=k_dense)
        sparse_rank = lexical.search(question, k=k_dense)
        chosen = {h.chunk.chunk_id: h for h in dense_rank + sparse_rank}
        fused = rrf_fuse(
            [h.chunk.chunk_id for h in dense_rank],
            [h.chunk.chunk_id for h in sparse_rank],
        )
        return [chosen[cid] for cid in fused if cid in chosen][:k]

    return _fn


def report(label: str, rows: list[dict]) -> None:
    p_at_k = np.mean([r["precision@k"] for r in rows])
    hit_at_1 = np.mean([1.0 if r["hit@1"] == r["expected"] else 0.0 for r in rows])
    lat = np.array([r["latency_ms"] for r in rows])
    print(f"{label:<10} precision@4={p_at_k:.2f}   hit@1={hit_at_1:.2f}   "
          f"latency mean={lat.mean():.1f} ms   p95={np.percentile(lat, 95):.1f} ms")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true",
                        help="also build a neural (sentence-transformers) index "
                             "and report dense + hybrid retrieval")
    args = parser.parse_args()

    docs = load_docs(SAMPLE_DIR)
    chunks = chunks_from_docs(docs)

    store, tfidf, build_s = build_tfidf(chunks)
    rows = [evaluate(dense_hits(store, tfidf), q, exp) for q, exp in QUERIES]

    print(f"OS                : {sys.platform}")
    print(f"Python            : {sys.version.split()[0]}")
    print(f"numpy             : {np.__version__}")
    print("")
    print(f"sample_docs       : {len(docs)} files "
          f"({sum(len(t.split()) for t in docs.values())} words)")
    print(f"chunks            : {store.size} (window=120, overlap=30)")
    print(f"vocabulary        : {tfidf.vocab.__len__()} terms")
    print(f"TF-IDF build      : {build_s * 1000:.1f} ms  (fit + transform)")
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
    print("-" * 132)
    report("TF-IDF", rows)

    if args.local:
        try:
            store_n, lexical, local, build_s2 = build_local(chunks)
        except ImportError as exc:
            print(f"\n--local requires sentence-transformers: "
                  f"pip install -r requirements-optional.txt ({exc})")
            return 1
        print("")
        print(f"local build       : {build_s2 * 1000:.1f} ms  (transform only; "
              f"~1 model download on first run)")
        dense_rows = [evaluate(dense_hits(store_n, local), q, exp) for q, exp in QUERIES]
        hybrid_rows = [evaluate(hybrid_hits(store_n, local, lexical), q, exp)
                       for q, exp in QUERIES]
        report("dense", dense_rows)
        report("hybrid", hybrid_rows)
        print("\nhybrid = BM25 sparse + neural dense, fused with RRF (k=60)")

    print(f"\nsearch op        : {store.size} x {len(tfidf.idf)} dot product per query "
          f"(brute-force; ANNs irrelevant at this corpus size)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
