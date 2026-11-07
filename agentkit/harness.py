"""The agent harness: the loop that turns a model into an agent.

Each step the model sees the task, the conversation so far and the tool schemas,
then either answers or asks for tool calls. The harness runs those calls and
feeds the results back, until the model answers or a limit is reached.

Around that loop the harness adds what production agents need:

* **Step and tool-call budgets**, so a confused agent cannot loop forever.
* **Human approval** for risky tools (refunds, restarts). Without approval the
  call is not run and the model is told why, so it can explain or escalate.
* **Memory**: relevant long-term memories are added to the task, and the agent
  can store new ones.
* **Tracing** of every model and tool call, for debugging and evals.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .llm import LLM, LLMResponse
from .memory import ConversationMemory, LongTermMemory
from .tools import ToolRegistry, ToolResult
from .tracing import Tracer

Approver = Callable[[str, dict[str, Any]], bool]


def deny_all(name: str, arguments: dict[str, Any]) -> bool:
    """The safe default: nothing risky runs without a human."""
    return False


def ask_in_terminal(name: str, arguments: dict[str, Any]) -> bool:
    answer = input(f"\nApprove {name}({arguments})? [y/N] ")
    return answer.strip().lower() in ("y", "yes")


@dataclass
class ToolCallRecord:
    name: str
    arguments: dict[str, Any]
    result: str
    is_error: bool
    approved: bool | None = None  # None when no approval was needed.
    seconds: float = 0.0


@dataclass
class RunResult:
    answer: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    steps: int = 0
    stop: str = "answered"  # answered, max_steps, max_tool_calls, error
    seconds: float = 0.0
    usage: dict[str, int] = field(default_factory=dict)
    trace: Tracer | None = None

    def called(self, name: str) -> list[ToolCallRecord]:
        return [c for c in self.tool_calls if c.name == name]


class Agent:
    def __init__(self, name: str, system: str, llm: LLM, tools: ToolRegistry | None = None, *,
                 max_steps: int = 8, max_tool_calls: int = 16, approver: Approver = deny_all,
                 long_term: LongTermMemory | None = None, conversation: ConversationMemory | None = None):
        self.name = name
        self.system = system
        self.llm = llm
        self.tools = tools or ToolRegistry()
        self.max_steps = max_steps
        self.max_tool_calls = max_tool_calls
        self.approver = approver
        self.long_term = long_term
        self.conversation = conversation

    def _task_message(self, task: str) -> dict[str, Any]:
        text = task
        if self.long_term:
            memories = self.long_term.recall(task)
            if memories:
                notes = "\n".join(f"- {m.text}" for m in memories)
                text = f"{task}\n\n<memory>\nRelevant notes from earlier work:\n{notes}\n</memory>"
        return {"role": "user", "content": text}

    def run(self, task: str, tracer: Tracer | None = None) -> RunResult:
        tracer = tracer or Tracer()
        started = time.time()
        messages = (self.conversation.messages() if self.conversation else []) + [self._task_message(task)]
        result = RunResult(answer="", trace=tracer)
        tracer.emit("run_start", agent=self.name, model=self.llm.name, task=task)

        while True:
            if result.steps >= self.max_steps:
                result.stop = "max_steps"
                result.answer = result.answer or "I stopped before finishing because I reached my step limit."
                break
            result.steps += 1
            try:
                response: LLMResponse = self.llm.complete(self.system, messages, self.tools.schemas())
            except Exception as error:
                tracer.emit("error", where="llm", error=str(error))
                result.stop, result.answer = "error", f"The model call failed: {error}"
                break
            for key, value in response.usage.items():
                if isinstance(value, int):
                    result.usage[key] = result.usage.get(key, 0) + value
            tracer.emit("llm", step=result.steps, text=response.text, tool_calls=[(c.name, c.arguments) for c in response.tool_calls])
            messages.append(response.to_message())

            if not response.tool_calls:
                result.answer = response.text
                break
            if len(result.tool_calls) + len(response.tool_calls) > self.max_tool_calls:
                result.stop = "max_tool_calls"
                result.answer = response.text or "I stopped because I reached my tool-call limit."
                break

            blocks = []
            for call in response.tool_calls:
                blocks.append(self._run_tool(call.id, call.name, call.arguments, result, tracer))
            messages.append({"role": "user", "content": blocks})

        result.seconds = round(time.time() - started, 3)
        if self.conversation and result.stop == "answered":
            self.conversation.add(task, result.answer)
        tracer.emit("run_end", stop=result.stop, steps=result.steps, tool_calls=len(result.tool_calls), seconds=result.seconds)
        return result

    def _run_tool(self, call_id: str, name: str, arguments: dict[str, Any], result: RunResult, tracer: Tracer) -> dict[str, Any]:
        tool = self.tools.get(name)
        approved = None
        started = time.time()
        if tool and tool.needs_approval(arguments):
            approved = bool(self.approver(name, arguments))
            tracer.emit("approval", tool=name, arguments=arguments, approved=approved)
        if approved is False:
            outcome = ToolResult(f"Not run: {name} needs human approval and it was not given. "
                                 "Tell the user what you would do and that a person must approve it.", True)
        else:
            outcome = self.tools.call(name, arguments)
        record = ToolCallRecord(name, arguments, outcome.content, outcome.is_error, approved, round(time.time() - started, 3))
        result.tool_calls.append(record)
        tracer.emit("tool", name=name, arguments=arguments, is_error=outcome.is_error, result=outcome.content[:300])
        block = {"type": "tool_result", "tool_use_id": call_id, "content": outcome.content}
        if outcome.is_error:
            block["is_error"] = True
        return block
