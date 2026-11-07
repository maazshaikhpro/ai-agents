"""Tool calling: typed Python functions become tools a model can call.

``@tool`` reads a function's signature, type hints and docstring and produces the
JSON Schema the model sees. ``ToolRegistry`` validates arguments before running a
tool, turns exceptions into error results the model can recover from, and caps
output size so one tool cannot flood the context window.
"""

from __future__ import annotations

import inspect
import json
import re
import typing
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, get_args, get_origin

MAX_RESULT_CHARS = 8000

_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean", dict: "object", list: "array"}


@dataclass
class ToolResult:
    content: str
    is_error: bool = False


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    fn: Callable[..., Any]
    requires_approval: bool | Callable[[dict[str, Any]], bool] = False
    tags: list[str] = field(default_factory=list)

    def schema(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}

    def needs_approval(self, arguments: dict[str, Any]) -> bool:
        if callable(self.requires_approval):
            return bool(self.requires_approval(arguments))
        return bool(self.requires_approval)


def _schema_for(annotation: Any) -> dict[str, Any]:
    origin = get_origin(annotation)
    if origin is Literal:
        values = list(get_args(annotation))
        return {"type": _JSON_TYPES.get(type(values[0]), "string"), "enum": values}
    if origin in (list, typing.List):
        args = get_args(annotation)
        return {"type": "array", "items": _schema_for(args[0]) if args else {}}
    if origin is typing.Union or str(origin) == "types.UnionType":
        options = [a for a in get_args(annotation) if a is not type(None)]
        return _schema_for(options[0]) if len(options) == 1 else {}
    return {"type": _JSON_TYPES.get(annotation, "string")}


def _parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """Splits a Google-style docstring into a summary and per-argument descriptions."""
    summary_lines: list[str] = []
    params: dict[str, str] = {}
    current = None
    in_args = False
    for line in inspect.cleandoc(doc or "").splitlines():
        if line.strip().lower() in ("args:", "arguments:", "parameters:"):
            in_args = True
            continue
        if in_args:
            match = re.match(r"\s*(\w+)\s*(\(.*?\))?:\s*(.*)", line)
            if match and not line.startswith("        "):
                current = match.group(1)
                params[current] = match.group(3).strip()
            elif current and line.strip():
                params[current] += " " + line.strip()
            elif not line.strip():
                in_args = False
        else:
            summary_lines.append(line)
    return " ".join(l.strip() for l in summary_lines if l.strip()), params


def tool(fn: Callable[..., Any] | None = None, *, name: str | None = None,
         requires_approval: bool | Callable[[dict[str, Any]], bool] = False, tags: list[str] | None = None):
    """Turns a function into a ``Tool``. Use as ``@tool`` or ``@tool(requires_approval=True)``."""

    def wrap(func: Callable[..., Any]) -> Tool:
        summary, docs = _parse_docstring(func.__doc__ or "")
        hints = typing.get_type_hints(func)
        properties: dict[str, Any] = {}
        required: list[str] = []
        for param in inspect.signature(func).parameters.values():
            prop = _schema_for(hints.get(param.name, str))
            if param.name in docs:
                prop["description"] = docs[param.name]
            if param.default is inspect.Parameter.empty:
                required.append(param.name)
            else:
                prop["default"] = param.default
            properties[param.name] = prop
        schema = {"type": "object", "properties": properties, "required": required}
        return Tool(name or func.__name__, summary or func.__name__, schema, func, requires_approval, tags or [])

    return wrap(fn) if fn else wrap


def _check(value: Any, schema: dict[str, Any], path: str) -> str | None:
    kind = schema.get("type")
    checks = {
        "string": lambda v: isinstance(v, str),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "boolean": lambda v: isinstance(v, bool),
        "object": lambda v: isinstance(v, dict),
        "array": lambda v: isinstance(v, list),
    }
    if kind in checks and not checks[kind](value):
        return f"{path} should be {kind}, got {type(value).__name__}"
    if "enum" in schema and value not in schema["enum"]:
        return f"{path} must be one of {schema['enum']}"
    if kind == "array" and schema.get("items"):
        for i, item in enumerate(value):
            problem = _check(item, schema["items"], f"{path}[{i}]")
            if problem:
                return problem
    return None


def validate(arguments: dict[str, Any], schema: dict[str, Any]) -> str | None:
    """Returns a readable problem with the arguments, or None if they fit the schema."""
    if not isinstance(arguments, dict):
        return "arguments must be an object"
    for name in schema.get("required", []):
        if name not in arguments:
            return f"missing required argument '{name}'"
    properties = schema.get("properties", {})
    for name, value in arguments.items():
        if name not in properties:
            return f"unknown argument '{name}'"
        problem = _check(value, properties[name], name)
        if problem:
            return problem
    return None


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for item in tools or []:
            self.add(item)

    def add(self, item: Tool) -> None:
        if item.name in self._tools:
            raise ValueError(f"duplicate tool name: {item.name}")
        self._tools[item.name] = item

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        item = self._tools.get(name)
        if item is None:
            return ToolResult(f"Unknown tool '{name}'. Available tools: {', '.join(self._tools)}", True)
        problem = validate(arguments, item.input_schema)
        if problem:
            return ToolResult(f"Invalid arguments for {name}: {problem}", True)
        try:
            value = item.fn(**arguments)
        except Exception as error:  # The model sees the error and can try again.
            return ToolResult(f"{type(error).__name__}: {error}", True)
        if isinstance(value, ToolResult):
            return value
        text = value if isinstance(value, str) else json.dumps(value, indent=2, default=str)
        if len(text) > MAX_RESULT_CHARS:
            text = text[:MAX_RESULT_CHARS] + f"\n... [truncated {len(text) - MAX_RESULT_CHARS} characters]"
        return ToolResult(text)
