# Retrieval-augmented generation (RAG)

Retrieval-augmented generation gives a model relevant information at the moment it answers, instead of relying only on what it learned in training. It is the standard way to let a model answer questions about private, recent or fast-changing documents.

A RAG pipeline has two phases. Indexing happens ahead of time: documents are loaded, split into chunks and stored in a search index. Retrieval happens per question: the system finds the most relevant chunks and places them in the prompt.

Chunking matters. Chunks that are too large waste context and dilute relevance; chunks that are too small lose meaning. A common approach splits on sentence or paragraph boundaries into a few hundred tokens, with a small overlap between neighbouring chunks so an idea is not cut in half.

Keyword search, such as BM25, ranks chunks by shared terms and is strong for exact names, codes and numbers. Vector search compares embeddings, which are numeric representations of meaning, and finds passages that say the same thing in different words.

Hybrid search runs both and merges the rankings. Reciprocal rank fusion (RRF) is a simple, robust way to merge: each result scores 1 divided by (k plus its rank) in every list, and the scores are added. Results ranked well by both methods rise to the top.

Good RAG answers cite their sources, for example with numbered references like [1], so readers can check them. When nothing relevant is retrieved, the system should say it does not know rather than guess.
