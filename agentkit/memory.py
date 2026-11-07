"""Memory.

* ``ConversationMemory`` keeps recent turns for multi-turn chats and trims the
  oldest ones to stay within a budget (short-term memory).
* ``LongTermMemory`` stores facts the agent learned and recalls the most relevant
  ones for a new task by similarity search (long-term memory). It can persist to
  a JSON file so knowledge survives between runs.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from .rag import HashingEmbedder, cosine


class ConversationMemory:
    def __init__(self, max_turns: int = 12):
        self.max_turns = max_turns
        self.turns: list[dict[str, Any]] = []

    def add(self, user: str, assistant: str) -> None:
        self.turns.append({"role": "user", "content": user})
        self.turns.append({"role": "assistant", "content": assistant})
        excess = len(self.turns) - self.max_turns * 2
        if excess > 0:
            self.turns = self.turns[excess:]

    def messages(self) -> list[dict[str, Any]]:
        return list(self.turns)


@dataclass
class MemoryItem:
    text: str
    tags: list[str] = field(default_factory=list)
    created: float = field(default_factory=time.time)


class LongTermMemory:
    def __init__(self, path: str | None = None, embedder: HashingEmbedder | None = None):
        self.path = path
        self.embedder = embedder or HashingEmbedder()
        self.items: list[MemoryItem] = []
        self._vectors: list[list[float]] = []
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                for raw in json.load(handle):
                    self._append(MemoryItem(**raw))

    def _append(self, item: MemoryItem) -> None:
        self.items.append(item)
        self._vectors.append(self.embedder.embed(item.text + " " + " ".join(item.tags)))

    def remember(self, text: str, tags: list[str] | None = None) -> None:
        if any(existing.text == text for existing in self.items):
            return
        self._append(MemoryItem(text, tags or []))
        self.save()

    def recall(self, query: str, k: int = 3, min_score: float = 0.15) -> list[MemoryItem]:
        if not self.items:
            return []
        q = self.embedder.embed(query)
        scored = sorted(((cosine(q, v), item) for v, item in zip(self._vectors, self.items)), key=lambda p: -p[0])
        return [item for score, item in scored[:k] if score >= min_score]

    def save(self) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump([asdict(item) for item in self.items], handle, indent=2)
