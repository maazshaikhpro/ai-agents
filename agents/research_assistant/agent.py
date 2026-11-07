"""Research Assistant: answers questions from a document collection, with citations.

Shows retrieval-augmented generation (RAG): hybrid search (BM25 plus vectors,
merged with reciprocal rank fusion), numbered citations, and refusing to answer
when the documents do not cover the question.
"""

from __future__ import annotations

import os
import re

from agentkit import Agent, HybridRetriever, ToolRegistry, choose_llm, deny_all, format_context, load_markdown_dir, tool
from agentkit.llm import LLMResponse, call, last_user_text, say, tool_results
from agentkit.rag import tokenize

DATA = os.path.join(os.path.dirname(__file__), "data")

SYSTEM = """You are a research assistant. Answer only from the passages returned by search_docs.
- Always search before answering. Search again with different words if the first results miss.
- Cite every claim with the passage number in square brackets, like [1] or [2][3].
- If the passages do not answer the question, say "I couldn't find that in the knowledge base." Do not use outside knowledge.
- Be concise: a short paragraph, then a "Sources:" line listing the titles you cited."""

NOT_FOUND = "I couldn't find that in the knowledge base."


def build_tools(retriever: HybridRetriever) -> ToolRegistry:
    @tool
    def search_docs(query: str, k: int = 4) -> str:
        """Searches the knowledge base and returns numbered passages to cite.

        Args:
            query: What to look for, in natural language or keywords.
            k: How many passages to return (1-8).
        """
        return format_context(retriever.search(query, max(1, min(8, k))))

    @tool
    def read_doc(doc_id: str) -> str:
        """Returns a whole document, for when a passage is not enough.

        Args:
            doc_id: The document id shown in brackets after a passage title, without .md.
        """
        doc = retriever.documents.get(doc_id.removesuffix(".md"))
        if not doc:
            raise KeyError(f"No document '{doc_id}'. Known: {', '.join(retriever.documents)}")
        return doc.text

    return ToolRegistry([search_docs, read_doc])


PASSAGE = re.compile(r"^\[(\d+)\] (.+?) \((.+?)\)\n(.+?)(?=\n\n\[\d+\] |\Z)", re.S | re.M)


def offline_planner(system: str, messages: list, tools: list) -> LLMResponse:
    """Offline stand-in for the model: search once, then answer extractively.

    It picks the sentences that best match the question and cites the passage
    each came from, which is what a grounded model answer should look like.
    """
    question = last_user_text(messages)
    results = tool_results(messages)
    if not results:
        return call("search_docs", query=question, k=4)
    passages = PASSAGE.findall(results[-1][1])
    if not passages:
        return say(NOT_FOUND)
    terms = set(tokenize(question))
    scored = []
    for number, title, _source, text in passages:
        for position, sentence in enumerate(re.split(r"(?<=[.!?])\s+", text.strip())):
            overlap = len(terms & set(tokenize(sentence)))
            if overlap:
                scored.append((overlap, -int(number), -position, sentence, int(number), title))
    if not scored or max(s[0] for s in scored) < 2 and len(terms) > 2:
        return say(NOT_FOUND)
    # Keep sentences nearly as relevant as the best one, once each (chunks overlap).
    top = max(s[0] for s in scored)
    seen: set[str] = set()
    best = []
    for item in sorted(scored, reverse=True):
        if item[0] * 2 < top or item[3] in seen:
            continue
        seen.add(item[3])
        best.append(item)
        if len(best) == 3:
            break
    best.sort(key=lambda s: (-s[1], -s[2]))  # Keep the source order so the answer reads naturally.
    answer = " ".join(f"{s[3]} [{s[4]}]" for s in best)
    cited = sorted({(s[4], s[5]) for s in best})
    sources = "; ".join(f"[{n}] {t}" for n, t in cited)
    return say(f"{answer}\n\nSources: {sources}")


def make_agent(approver=deny_all, llm=None) -> Agent:
    retriever = HybridRetriever(load_markdown_dir(DATA), max_words=90)
    return Agent("Research Assistant", SYSTEM, llm or choose_llm(offline_planner), build_tools(retriever),
                 max_steps=6, approver=approver)
