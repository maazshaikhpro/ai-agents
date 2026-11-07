"""A shared command line for the agents: ``python -m agents.<name> "task"``."""

from __future__ import annotations

import argparse
import sys
from typing import Callable

from .harness import Agent, ask_in_terminal
from .tracing import Tracer


def main(make_agent: Callable[..., Agent], description: str, example: str, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("task", nargs="*", help=f'what to ask, e.g. "{example}"')
    parser.add_argument("--approve", action="store_true", help="ask in the terminal before risky actions (default: never approve)")
    parser.add_argument("--trace", action="store_true", help="print every model and tool call")
    parser.add_argument("--save-trace", metavar="FILE", help="write the trace as JSON Lines")
    args = parser.parse_args(argv)

    task = " ".join(args.task) or example
    options = {"approver": ask_in_terminal} if args.approve else {}
    agent = make_agent(**options)
    tracer = Tracer(echo=args.trace or None)
    try:
        print(f"\n{agent.name} · {agent.llm.name}\n> {task}\n", file=sys.stderr)
        result = agent.run(task, tracer)
    finally:
        close = getattr(agent, "close", None)
        if callable(close):
            close()
    print(result.answer)
    tools = ", ".join(c.name + (" (awaiting approval)" if c.approved is False else "") for c in result.tool_calls) or "none"
    print(f"\n[{result.steps} steps · tools: {tools} · {result.seconds}s]", file=sys.stderr)
    if args.save_trace:
        tracer.save(args.save_trace)
    return 0 if result.stop == "answered" else 1
