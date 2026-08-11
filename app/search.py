"""Hybrid retrieval: fuse sparse (BM25) and dense (neural vector) results.

RRF (Reciprocal Rank Fusion) combines two ranked lists without any tuning:
        score(chunk) = sum over lists of  1 / (k + rank_in_list)

`k = 60` is the constant from the original RRF paper and the default in most
open-source implementations. It rewards agreement (a chunk ranked highly in
*both* lists) rather than absolute scores, which is what makes fusing
incomparable signals (BM25 scores vs cosine similarities) work well.
"""

from __future__ import annotations

RRF_K = 60


def rrf_fuse(*ranked: list[str], k: int = RRF_K) -> list[str]:
    """Fuse ranked `chunk_id` lists into a single ranking by RRF score.

    Returns chunk_ids ordered best-first. A chunk only present in one list
    still scores, but a chunk ranked well in several lists wins.
    """
    totals: dict[str, float] = {}
    for ranked_list in ranked:
        for rank, chunk_id in enumerate(ranked_list, start=1):
            totals[chunk_id] = totals.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(totals, key=totals.__getitem__, reverse=True)
