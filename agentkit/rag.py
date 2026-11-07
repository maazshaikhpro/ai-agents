"""Retrieval-augmented generation (RAG).

The pipeline: load documents -> split into overlapping chunks -> index each chunk
two ways -> retrieve with both -> fuse the rankings -> hand the best chunks to the
model with numbered citations.

* Keyword search uses BM25, which is strong on exact terms such as names and codes.
* Vector search uses embeddings, which match meaning even when words differ.
* Reciprocal rank fusion combines both rankings (hybrid search).

``HashingEmbedder`` turns text into vectors with the hashing trick, so the kit
works offline with no model downloads. Swap in any real embedding model by
giving ``VectorIndex`` an object with an ``embed(text) -> list[float]`` method.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Protocol

STOPWORDS = set("""a an and are as at be been but by can do does for from has have how i if in into is it its
of on or our so that the their them then there these this to was we were what when where which who why will with
you your""".split())


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*", text.lower())
    return [stem(w) for w in words if w not in STOPWORDS]


def stem(word: str) -> str:
    """A light suffix stripper, enough to match "agents" with "agent"."""
    for suffix in ("ing", "ies", "es", "ed", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


@dataclass
class Document:
    id: str
    title: str
    text: str
    source: str = ""


@dataclass
class Chunk:
    id: str
    doc_id: str
    title: str
    text: str
    source: str = ""
    meta: dict = field(default_factory=dict)


def chunk_text(text: str, max_words: int = 120, overlap: int = 1) -> list[str]:
    """Splits text at sentence boundaries into chunks of about ``max_words``,
    repeating ``overlap`` sentences between neighbours so no idea is cut in half."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    sentences: list[str] = []
    for paragraph in paragraphs:
        if paragraph.startswith("#"):
            continue  # Headings become chunk titles instead.
        sentences.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", paragraph) if s.strip())
    chunks: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        if current and len(" ".join(current + [sentence]).split()) > max_words:
            chunks.append(" ".join(current))
            current = current[-overlap:] if overlap else []
        current.append(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_documents(documents: Iterable[Document], max_words: int = 120) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in documents:
        for i, text in enumerate(chunk_text(doc.text, max_words)):
            chunks.append(Chunk(f"{doc.id}#{i}", doc.id, doc.title, text, doc.source))
    return chunks


def load_markdown_dir(path: str) -> list[Document]:
    """Loads every .md file in a folder; the first heading becomes the title."""
    documents = []
    for name in sorted(os.listdir(path)):
        if not name.endswith(".md"):
            continue
        with open(os.path.join(path, name), encoding="utf-8") as handle:
            text = handle.read()
        heading = re.search(r"^#\s+(.+)$", text, re.M)
        documents.append(Document(name[:-3], heading.group(1).strip() if heading else name[:-3], text, name))
    return documents


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...


class HashingEmbedder:
    """Maps words and word pairs into a fixed-size vector by hashing, then
    normalises it. Similar texts share features, so cosine similarity works."""

    def __init__(self, dimensions: int = 512):
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        words = tokenize(text)
        features = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
        for feature in features:
            digest = hashlib.md5(feature.encode()).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            vector[index] += 1.0 if digest[4] % 2 else -1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(f"{c.title} {c.text}")) for c in chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lengths) / max(1, len(self.lengths))
        df = Counter(term for d in self.docs for term in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        terms = tokenize(query)
        scores = []
        for i, doc in enumerate(self.docs):
            score = 0.0
            for term in terms:
                tf = doc.get(term, 0)
                if tf:
                    norm = tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg))
                    score += self.idf.get(term, 0) * norm
            if score > 0:
                scores.append((self.chunks[i], score))
        return sorted(scores, key=lambda p: -p[1])[:k]


class VectorIndex:
    def __init__(self, chunks: list[Chunk], embedder: Embedder | None = None):
        self.chunks = chunks
        self.embedder = embedder or HashingEmbedder()
        self.vectors = [self.embedder.embed(f"{c.title} {c.text}") for c in chunks]

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        q = self.embedder.embed(query)
        scored = [(c, cosine(q, v)) for c, v in zip(self.chunks, self.vectors)]
        return sorted([p for p in scored if p[1] > 0], key=lambda p: -p[1])[:k]


@dataclass
class Hit:
    chunk: Chunk
    score: float
    keyword_rank: int | None
    vector_rank: int | None


class HybridRetriever:
    """BM25 plus vector search, combined with reciprocal rank fusion (RRF)."""

    def __init__(self, documents: list[Document], max_words: int = 120, embedder: Embedder | None = None, rrf_k: int = 60):
        self.documents = {d.id: d for d in documents}
        self.chunks = chunk_documents(documents, max_words)
        self.keyword = BM25Index(self.chunks)
        self.vector = VectorIndex(self.chunks, embedder)
        self.rrf_k = rrf_k

    def search(self, query: str, k: int = 4, min_keyword_hits: int = 1) -> list[Hit]:
        keyword = self.keyword.search(query, k * 4)
        vector = self.vector.search(query, k * 4)
        # With no keyword overlap at all, the vector matches are likely noise.
        if len(keyword) < min_keyword_hits:
            return []
        fused: dict[str, Hit] = {}
        for rank, (chunk, _) in enumerate(keyword):
            hit = fused.setdefault(chunk.id, Hit(chunk, 0.0, None, None))
            hit.score += 1 / (self.rrf_k + rank + 1)
            hit.keyword_rank = rank + 1
        for rank, (chunk, _) in enumerate(vector):
            hit = fused.setdefault(chunk.id, Hit(chunk, 0.0, None, None))
            hit.score += 1 / (self.rrf_k + rank + 1)
            hit.vector_rank = rank + 1
        return sorted(fused.values(), key=lambda h: -h.score)[:k]


def format_context(hits: list[Hit]) -> str:
    """Numbers each passage so the model can cite it as [1], [2], ..."""
    if not hits:
        return "No relevant passages found."
    return "\n\n".join(f"[{i}] {h.chunk.title} ({h.chunk.source or h.chunk.doc_id})\n{h.chunk.text}" for i, h in enumerate(hits, 1))
