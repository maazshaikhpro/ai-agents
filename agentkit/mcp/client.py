"""An MCP client for the stdio transport.

It starts a server as a subprocess, performs the initialize handshake, lists the
server's tools and turns each into a ``Tool`` the harness can call, so an agent
uses MCP tools exactly like local ones.
"""

from __future__ import annotations

import itertools
import json
import subprocess
import sys
import threading
from typing import Any

from ..tools import Tool, ToolResult
from .server import PROTOCOL_VERSION


class MCPError(RuntimeError):
    pass


class MCPClient:
    def __init__(self, command: list[str], name: str | None = None, timeout: float = 30.0, env: dict[str, str] | None = None):
        self.command = command
        self.name = name or command[-1]
        self.timeout = timeout
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True, bufsize=1, env=env)
        try:
            info = self.request("initialize", {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "agentkit", "version": "0.1.0"},
            })
        except Exception:
            self.close()
            raise
        self.server_info = info.get("serverInfo", {})
        self.instructions = info.get("instructions", "")
        self.notify("notifications/initialized")

    @classmethod
    def python_module(cls, module: str, **options: Any) -> "MCPClient":
        """Starts a server written as a Python module, e.g. ``agents.data_analyst.server``."""
        return cls([sys.executable, "-m", module], name=module, **options)

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, **({"params": params} if params else {})})

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            msg_id = next(self._ids)
            self._send({"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params or {}})
            while True:
                line = self._read_line()
                message = json.loads(line)
                if message.get("id") != msg_id:
                    continue  # Server notifications or stale replies.
                if "error" in message:
                    raise MCPError(f"{method}: {message['error'].get('message')}")
                return message.get("result", {})

    def _send(self, message: dict[str, Any]) -> None:
        if self.process.poll() is not None:
            raise MCPError(f"MCP server {self.name} has exited: {self.process.stderr.read()[-500:]}")
        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()

    def _read_line(self) -> str:
        holder: list[str] = []
        reader = threading.Thread(target=lambda: holder.append(self.process.stdout.readline()), daemon=True)
        reader.start()
        reader.join(self.timeout)
        if not holder:
            raise MCPError(f"MCP server {self.name} did not answer within {self.timeout}s")
        if not holder[0]:
            raise MCPError(f"MCP server {self.name} closed the connection: {self.process.stderr.read()[-500:]}")
        return holder[0]

    def list_tools(self) -> list[dict[str, Any]]:
        return self.request("tools/list").get("tools", [])

    def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        result = self.request("tools/call", {"name": name, "arguments": arguments})
        text = "\n".join(part.get("text", "") for part in result.get("content", []) if part.get("type") == "text")
        return ToolResult(text, bool(result.get("isError")))

    def list_resources(self) -> list[dict[str, Any]]:
        return self.request("resources/list").get("resources", [])

    def read_resource(self, uri: str) -> str:
        contents = self.request("resources/read", {"uri": uri}).get("contents", [])
        return "\n".join(c.get("text", "") for c in contents)

    def as_tools(self, prefix: str = "", requires_approval: dict[str, Any] | None = None) -> list[Tool]:
        """The server's tools as harness tools. ``requires_approval`` maps tool names
        to True or a function of the arguments, for actions a person must approve."""
        approvals = requires_approval or {}
        tools = []
        for spec in self.list_tools():
            remote = spec["name"]
            tools.append(Tool(
                name=f"{prefix}{remote}",
                description=spec.get("description", ""),
                input_schema=spec.get("inputSchema", {"type": "object", "properties": {}}),
                fn=lambda _remote=remote, **arguments: self.call_tool(_remote, arguments),
                requires_approval=approvals.get(remote, False),
                tags=["mcp", self.name],
            ))
        return tools

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()
                self.process.wait()
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            try:
                stream.close()
            except Exception:
                pass

    def __enter__(self) -> "MCPClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
