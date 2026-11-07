import unittest

from agentkit.rag import BM25Index, Document, HybridRetriever, chunk_documents, chunk_text, format_context, tokenize

DOCS = [
    Document("cats", "Cats", "Cats are small carnivorous mammals. They purr when content. House cats sleep most of the day."),
    Document("rag", "RAG", "Retrieval-augmented generation retrieves passages before the model answers. Reciprocal rank fusion merges rankings."),
    Document("mcp", "MCP", "The Model Context Protocol uses JSON-RPC. The stdio transport runs servers as subprocesses."),
]


class RagTests(unittest.TestCase):
    def test_tokenize_stems_and_drops_stopwords(self):
        self.assertEqual(tokenize("The agents are running"), ["agent", "runn"])

    def test_chunks_respect_size_and_overlap(self):
        text = " ".join(f"Sentence number {i} is here." for i in range(30))
        chunks = chunk_text(text, max_words=20, overlap=1)
        self.assertGreater(len(chunks), 3)
        self.assertTrue(all(len(c.split()) <= 25 for c in chunks))
        last_sentence = chunks[0].split(". ")[-1]
        self.assertIn(last_sentence.rstrip("."), chunks[1])

    def test_bm25_ranks_exact_terms_first(self):
        hits = BM25Index(chunk_documents(DOCS)).search("stdio transport")
        self.assertEqual(hits[0][0].doc_id, "mcp")

    def test_hybrid_search_finds_and_refuses(self):
        retriever = HybridRetriever(DOCS)
        self.assertEqual(retriever.search("how are rankings merged?")[0].chunk.doc_id, "rag")
        self.assertEqual(retriever.search("quantum chromodynamics lattice"), [])

    def test_context_is_numbered_for_citations(self):
        context = format_context(HybridRetriever(DOCS).search("cats purr"))
        self.assertTrue(context.startswith("[1] Cats"))


if __name__ == "__main__":
    unittest.main()
