# Embeddings and TF-IDF

An embedding is a vector — a list of numbers — that represents a piece of text.
The goal is that texts about similar topics get similar vectors, so that
"finding relevant text" becomes "finding nearby vectors" (simple geometry).

## TF-IDF

TF-IDF is a classic, fully transparent way to build such vectors without any
machine learning model. It combines two signals:

- Term Frequency (TF): how often a word appears in a chunk. Frequent words are
  important to that chunk.
- Inverse Document Frequency (IDF): how rare a word is across all chunks. Words
  that appear everywhere (like "the", "and") carry little meaning, so they get a
  low weight. Rare, distinctive words get a high weight.

Multiply TF by IDF for every word and you get a vector per chunk. TF-IDF is fast,
needs no downloads, and is easy to reason about — but it matches on exact words,
so it treats "car" and "automobile" as unrelated.

## Neural embeddings

Neural embedding models (like sentence-transformers) capture meaning, so
"car" and "automobile" land close together. They are more powerful but need a
model download and more compute. Because good systems code retrieval against an
interface, you can start with TF-IDF and later swap in a neural embedder without
changing the rest of the application.

## Cosine similarity

To compare two vectors we use cosine similarity: the cosine of the angle between
them, ranging from 0 (unrelated) to 1 (identical direction). If you L2-normalise
every vector first, cosine similarity is simply their dot product — one
multiplication instead of a division.
