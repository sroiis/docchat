# What is RAG (Retrieval-Augmented Generation)?

RAG is a pattern for making a language model answer questions using your own
documents instead of only what it memorised during training. The flow has two
halves:

1. Retrieval: given a question, find the most relevant pieces of your documents.
   This is a search problem. You convert text into vectors and find the nearest
   ones to the question vector.

2. Generation: hand those relevant pieces to a language model and ask it to
   write an answer grounded in them, citing the sources.

The retrieval half is owned by backend and infra engineers. It involves
chunking documents, computing embeddings, storing vectors in a vector database,
and serving low-latency nearest-neighbour queries. The generation half is an
optional final call to an LLM.

Retrieval-only systems (no generation step) are still very useful: they behave
like a smart search engine that returns the exact passage answering your
question, with a relevance score and a source. This project is retrieval-only.
