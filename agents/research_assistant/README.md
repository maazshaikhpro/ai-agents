# Research Assistant (RAG)

Answers questions from the documents in `data/`, a short knowledge base on LLMs, RAG, agents, MCP, evals and agent safety, and cites the passage behind every claim.

```bash
python3 -m agents.research_assistant "What is reciprocal rank fusion?"
```

**How it works**

1. At start-up the documents are split into overlapping, sentence-aligned chunks and indexed twice: BM25 for exact terms and vectors for meaning.
2. The agent calls `search_docs`. Both indexes are searched and the rankings merged with reciprocal rank fusion.
3. Passages come back numbered `[1]`, `[2]`, and the answer must cite them.
4. If nothing shares a keyword with the question, search returns nothing and the agent says it couldn't find the answer, instead of guessing.

**Tools:** `search_docs(query, k)`, `read_doc(doc_id)`

**Try your own documents:** put Markdown files in `data/`. The first `#` heading becomes the title.
