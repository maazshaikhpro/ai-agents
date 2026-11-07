"""Tracing: every model call and tool call becomes an event you can inspect.

Traces print live with ``AGENT_TRACE=1`` and can be written as JSON Lines, so a
failed run can be replayed step by step.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Event:
    kind: str  # run_start, llm, tool, approval, run_end, error
    at: float
    data: dict[str, Any] = field(default_factory=dict)


class Tracer:
    def __init__(self, echo: bool | None = None):
        self.events: list[Event] = []
        self.echo = os.environ.get("AGENT_TRACE") == "1" if echo is None else echo

    def emit(self, kind: str, **data: Any) -> None:
        event = Event(kind, time.time(), data)
        self.events.append(event)
        if self.echo:
            print(f"  · {kind:<10} {self._summary(data)}", file=sys.stderr)

    @staticmethod
    def _summary(data: dict[str, Any]) -> str:
        text = json.dumps(data, default=str)
        return text if len(text) < 160 else text[:157] + "..."

    def of(self, kind: str) -> list[Event]:
        return [e for e in self.events if e.kind == kind]

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            for event in self.events:
                handle.write(json.dumps(asdict(event), default=str) + "\n")
