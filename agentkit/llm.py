"""Model clients.

Messages use the Anthropic Messages API shape throughout the kit:
``{"role": "user" | "assistant", "content": str | list[block]}`` where blocks are
``text``, ``tool_use`` and ``tool_result``. Any model can sit behind the ``LLM``
interface; two are included:

* ``AnthropicLLM`` calls Claude over HTTPS using only the standard library.
* ``ScriptedLLM`` runs a deterministic planner function, so every agent works and
  can be tested without an API key or network access.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

DEFAULT_MODEL = os.environ.get("AGENT_MODEL", "claude-sonnet-5-5")
API_URL = "https://api.anthropic.com/v1/messages"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"
    usage: dict[str, int] = field(default_factory=dict)

    def to_message(self) -> dict[str, Any]:
        """The assistant message to append to the conversation."""
        blocks: list[dict[str, Any]] = []
        if self.text:
            blocks.append({"type": "text", "text": self.text})
        for call in self.tool_calls:
            blocks.append({"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments})
        return {"role": "assistant", "content": blocks}


class LLM(Protocol):
    name: str

    def complete(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse: ...


class LLMError(RuntimeError):
    pass


class AnthropicLLM:
    """Claude through the Messages API, with retries on rate limits and server errors."""

    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None, max_tokens: int = 2048,
                 temperature: float = 0.0, retries: int = 3, timeout: float = 120.0):
        self.model = model
        self.name = f"anthropic:{model}"
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.retries = retries
        self.timeout = timeout

    def complete(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system": system,
            "messages": messages,
        }
        if tools:
            body["tools"] = tools
        data = self._post(body)
        text = "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")
        calls = [ToolCall(b["id"], b["name"], b.get("input") or {}) for b in data.get("content", []) if b.get("type") == "tool_use"]
        return LLMResponse(text=text.strip(), tool_calls=calls, stop_reason=data.get("stop_reason", ""), usage=data.get("usage", {}))

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        payload = json.dumps(body).encode()
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(API_URL, data=payload, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read())
            except urllib.error.HTTPError as error:
                detail = error.read().decode(errors="replace")[:500]
                if error.code in (429, 500, 502, 503, 529) and attempt < self.retries:
                    time.sleep(min(30, 2 ** attempt + float(error.headers.get("retry-after", 0) or 0)))
                    continue
                raise LLMError(f"Anthropic API error {error.code}: {detail}") from error
            except urllib.error.URLError as error:
                if attempt < self.retries:
                    time.sleep(2 ** attempt)
                    continue
                raise LLMError(f"Could not reach the Anthropic API: {error.reason}") from error
        raise LLMError("unreachable")


Policy = Callable[[str, list[dict[str, Any]], list[dict[str, Any]]], LLMResponse]


class ScriptedLLM:
    """Runs a deterministic planner in place of a model.

    The planner sees exactly what a model would (system prompt, conversation and
    tool schemas) and returns text or tool calls. Agents ship one so they run
    offline, and tests use them to exercise the harness reproducibly.
    """

    def __init__(self, policy: Policy, name: str = "offline-planner"):
        self.policy = policy
        self.name = name
        self._ids = 0

    def complete(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        response = self.policy(system, messages, tools)
        for call in response.tool_calls:
            if not call.id:
                self._ids += 1
                call.id = f"call_{self._ids}"
        return response


def choose_llm(offline_policy: Policy | None = None) -> LLM:
    """Claude when ``ANTHROPIC_API_KEY`` is set (and ``AGENT_OFFLINE`` is not), else the offline planner."""
    if os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("AGENT_OFFLINE"):
        return AnthropicLLM()
    if offline_policy is None:
        raise LLMError("Set ANTHROPIC_API_KEY, or give this agent an offline planner")
    return ScriptedLLM(offline_policy)


# Helpers for planners: read the conversation the way a model would.

def last_user_text(messages: list[dict[str, Any]]) -> str:
    """The most recent plain-text user message (the task)."""
    for message in reversed(messages):
        if message["role"] != "user":
            continue
        content = message["content"]
        if isinstance(content, str):
            return content
        texts = [b.get("text", "") for b in content if b.get("type") == "text"]
        if texts:
            return "\n".join(texts)
    return ""


def tool_results(messages: list[dict[str, Any]]) -> list[tuple[str, str, bool]]:
    """Every tool result so far as (tool name, content, is_error), oldest first."""
    names: dict[str, str] = {}
    out: list[tuple[str, str, bool]] = []
    for message in messages:
        if not isinstance(message["content"], list):
            continue
        for block in message["content"]:
            if block.get("type") == "tool_use":
                names[block["id"]] = block["name"]
            elif block.get("type") == "tool_result":
                content = block.get("content", "")
                if isinstance(content, list):
                    content = "".join(part.get("text", "") for part in content)
                out.append((names.get(block["tool_use_id"], "?"), content, bool(block.get("is_error"))))
    return out


def call(name: str, **arguments: Any) -> LLMResponse:
    """A response that calls one tool."""
    return LLMResponse(tool_calls=[ToolCall("", name, arguments)], stop_reason="tool_use")


def say(text: str) -> LLMResponse:
    """A final answer."""
    return LLMResponse(text=text, stop_reason="end_turn")
