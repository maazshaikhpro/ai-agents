"""A minimal Model Context Protocol (MCP) server.

MCP is an open protocol for connecting AI applications to tools and data. A
server exposes tools (and optionally resources) over JSON-RPC 2.0; this one uses
the stdio transport, where each message is one line of JSON on stdin/stdout.

Supported methods: ``initialize``, ``notifications/initialized``, ``ping``,
``tools/list``, ``tools/call``, ``resources/list`` and ``resources/read``.
Any MCP client can use these servers, including Claude Desktop, and the
``MCPClient`` in this kit.

    server = MCPServer("weather")

    @server.tool
    def forecast(city: str) -> str:
        "Today's forecast for a city."
        ...

    if __name__ == "__main__":
        server.run_stdio()
"""

from __future__ import annotations

import json
import sys
from typing import Any, Callable, TextIO

from ..tools import Tool, ToolRegistry, tool as make_tool

PROTOCOL_VERSION = "2025-06-18"

PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS = -32700, -32600, -32601, -32602


class MCPServer:
    def __init__(self, name: str, version: str = "0.1.0", instructions: str = ""):
        self.name = name
        self.version = version
        self.instructions = instructions
        self.registry = ToolRegistry()
        self.resources: dict[str, tuple[str, str, Callable[[], str]]] = {}

    def tool(self, fn: Callable[..., Any] | None = None, **options: Any):
        """Registers a function as an MCP tool, with its schema built from type hints."""

        def wrap(func: Callable[..., Any]) -> Callable[..., Any]:
            item: Tool = make_tool(func, **options)
            self.registry.add(item)
            return func

        return wrap(fn) if fn else wrap

    def resource(self, uri: str, name: str, mime_type: str = "text/plain"):
        def wrap(func: Callable[[], str]) -> Callable[[], str]:
            self.resources[uri] = (name, mime_type, func)
            return func

        return wrap

    def handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """Handles one JSON-RPC message. Returns the response, or None for notifications."""
        if message.get("jsonrpc") != "2.0" or "method" not in message:
            return self._error(message.get("id"), INVALID_REQUEST, "Invalid request")
        method, params, msg_id = message["method"], message.get("params") or {}, message.get("id")
        if msg_id is None:  # Notifications get no reply.
            return None
        try:
            if method == "initialize":
                result = {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}, **({"resources": {}} if self.resources else {})},
                    "serverInfo": {"name": self.name, "version": self.version},
                    **({"instructions": self.instructions} if self.instructions else {}),
                }
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": [{"name": t["name"], "description": t["description"], "inputSchema": t["input_schema"]}
                                    for t in self.registry.schemas()]}
            elif method == "tools/call":
                name = params.get("name")
                if not self.registry.get(name):
                    return self._error(msg_id, INVALID_PARAMS, f"Unknown tool: {name}")
                outcome = self.registry.call(name, params.get("arguments") or {})
                result = {"content": [{"type": "text", "text": outcome.content}], "isError": outcome.is_error}
            elif method == "resources/list":
                result = {"resources": [{"uri": uri, "name": n, "mimeType": m} for uri, (n, m, _) in self.resources.items()]}
            elif method == "resources/read":
                uri = params.get("uri")
                if uri not in self.resources:
                    return self._error(msg_id, INVALID_PARAMS, f"Unknown resource: {uri}")
                _, mime, read = self.resources[uri]
                result = {"contents": [{"uri": uri, "mimeType": mime, "text": read()}]}
            else:
                return self._error(msg_id, METHOD_NOT_FOUND, f"Method not found: {method}")
        except Exception as error:
            return self._error(msg_id, -32603, f"Internal error: {error}")
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    @staticmethod
    def _error(msg_id: Any, code: int, text: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": text}}

    def run_stdio(self, stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> None:
        """Serves requests until stdin closes. Logs go to stderr; stdout is protocol only."""
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                response = self._error(None, PARSE_ERROR, "Parse error")
            else:
                response = self.handle(message)
            if response is not None:
                stdout.write(json.dumps(response) + "\n")
                stdout.flush()
