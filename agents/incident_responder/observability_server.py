"""MCP server: read-only observability data (alerts, metrics, logs, deploys).

Backed by fixture data so it runs anywhere; in production these tools would call
your monitoring, logging and CI/CD systems.
"""

from __future__ import annotations

import json
import os
from typing import Literal

from agentkit.mcp import MCPServer

DATA = os.path.join(os.path.dirname(__file__), "data", "scenarios.json")

Metric = Literal["error_rate", "latency_p95_ms", "requests_per_min", "db_connections_used", "memory_pct", "restarts"]


def build_server() -> MCPServer:
    with open(DATA, encoding="utf-8") as handle:
        data = json.load(handle)
    server = MCPServer("observability", instructions="Read-only. Times are UTC.")

    def known(service: str) -> str:
        if service not in data["metrics"]:
            raise KeyError(f"Unknown service '{service}'. Services: {', '.join(data['metrics'])}")
        return service

    @server.tool
    def list_alerts() -> str:
        """Lists alerts that are currently firing."""
        return json.dumps([{"id": k, **v} for k, v in data["alerts"].items()], indent=2)

    @server.tool
    def get_alert(alert_id: str) -> str:
        """Gets one alert's service, title, metric, start time and severity.

        Args:
            alert_id: The alert id, like ALRT-101.
        """
        alert = data["alerts"].get(alert_id.upper())
        if not alert:
            raise KeyError(f"No alert {alert_id}")
        return json.dumps({"id": alert_id.upper(), **alert}, indent=2)

    @server.tool
    def query_metrics(service: str, metric: Metric) -> str:
        """Returns a metric's recent time series for a service, oldest first.

        Args:
            service: The service name, e.g. checkout.
            metric: Which metric to read.
        """
        series = data["metrics"][known(service)].get(metric)
        if not series:
            return f"No {metric} data for {service}. Available: {', '.join(data['metrics'][service])}"
        return "\n".join(f"{t}  {v}" for t, v in series)

    @server.tool
    def search_logs(service: str, query: str = "", limit: int = 10) -> str:
        """Searches a service's recent logs. An empty query returns the latest lines.

        Args:
            service: The service name.
            query: Words to look for, e.g. "error" or "timeout"; any word may match.
            limit: Maximum lines to return.
        """
        lines = data["logs"][known(service)]
        words = [w.lower() for w in query.split()]
        matches = [l for l in lines if not words or any(w in l.lower() for w in words)]
        return "\n".join(matches[:limit]) or "No matching log lines."

    @server.tool
    def recent_deploys(service: str) -> str:
        """Lists a service's recent deploys, newest first, and its current replica count.

        Args:
            service: The service name.
        """
        return json.dumps({"service": service, "replicas": data["replicas"][known(service)], "deploys": data["deploys"][service]}, indent=2)

    return server


if __name__ == "__main__":
    build_server().run_stdio()
