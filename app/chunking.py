"""Split documents into chunks.

Why chunk at all? A whole document is too coarse to search: if you ask a
question, you want the *specific paragraph* that answers it, not a 5-page file.
So we break each document into overlapping windows of words. The overlap makes
sure a sentence that straddles a boundary isn't lost.

This is the same idea used by every production RAG system — only the tokenizer
differs (they count model tokens; we count words, which is close enough and has
zero dependencies).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    doc_id: str          # which source file this came from
    chunk_id: str        # unique id, e.g. "kafka_basics.md#3"
    text: str            # the actual chunk content
    order: int           # position of this chunk within the document


def chunk_text(
    doc_id: str,
    text: str,
    words_per_chunk: int = 120,
    overlap: int = 30,
) -> list[Chunk]:
    """Break `text` into overlapping word-windows.

    words_per_chunk: target size of each chunk.
    overlap:         how many words the next chunk repeats from the previous one.
    """
    if overlap >= words_per_chunk:
        raise ValueError("overlap must be smaller than words_per_chunk")

    words = text.split()
    if not words:
        return []

    chunks: list[Chunk] = []
    step = words_per_chunk - overlap
    order = 0
    for start in range(0, len(words), step):
        window = words[start : start + words_per_chunk]
        if not window:
            break
        chunks.append(
            Chunk(
                doc_id=doc_id,
                chunk_id=f"{doc_id}#{order}",
                text=" ".join(window),
                order=order,
            )
        )
        order += 1
        # If this window already reached the end, stop (avoid a tiny trailing dup).
        if start + words_per_chunk >= len(words):
            break
    return chunks
