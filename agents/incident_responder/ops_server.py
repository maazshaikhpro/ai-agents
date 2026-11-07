"""MCP server: production actions (rollback, scale, restart).

These tools change production, so the agent marks every one as needing human
approval; the harness will not run them until a person says yes. This server
only records the action; in production it would call your deploy platform.
"""

from __future__ import annotations

import json
import time

from agentkit.mcp import MCPServer


def build_server() -> MCPServer:
    server = MCPServer("operations", instructions="Every tool here changes production and must be approved by on-call.")
    log: list[dict] = []

    def record(action: str, **details) -> str:
        entry = {"action": action, **details, "status": "done", "at": time.strftime("%H:%M:%SZ", time.gmtime())}
        log.append(entry)
        return json.dumps(entry)

    @server.tool
    def rollback_deploy(service: str, to_version: str) -> str:
        """Rolls a service back to an earlier version.

        Args:
            service: The service to roll back.
            to_version: The version to return to, e.g. v2.3.9.
        """
        return record("rollback", service=service, to_version=to_version)

    @server.tool
    def scale_service(service: str, replicas: int) -> str:
        """Changes how many replicas of a service run.

        Args:
            service: The service to scale.
            replicas: The new replica count (1-20).
        """
        if not 1 <= replicas <= 20:
            raise ValueError("replicas must be between 1 and 20")
        return record("scale", service=service, replicas=replicas)

    @server.tool
    def restart_service(service: str) -> str:
        """Restarts every pod of a service, one at a time.

        Args:
            service: The service to restart.
        """
        return record("restart", service=service)

    @server.tool
    def action_log() -> str:
        """Lists actions taken in this session."""
        return json.dumps(log, indent=2)

    return server


if __name__ == "__main__":
    build_server().run_stdio()
