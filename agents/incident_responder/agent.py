"""Incident Responder: investigates a production alert and proposes the fix.

The most complete example. It combines:
* two MCP servers: read-only observability, and production operations,
* RAG over runbooks, with citations,
* long-term memory of past incidents, recalled for similar alerts,
* human approval before any production change (rollback, scale, restart).
"""

from __future__ import annotations

import json
import os
import re

from agentkit import Agent, HybridRetriever, LongTermMemory, ToolRegistry, choose_llm, deny_all, format_context, load_markdown_dir, tool
from agentkit.llm import LLMResponse, call, last_user_text, say, tool_results
from agentkit.mcp import MCPClient

HERE = os.path.dirname(__file__)

SYSTEM = """You are an on-call SRE agent. Investigate the alert, find the root cause and mitigate.
1. Read the alert. 2. Check the metric it fired on, then the logs and recent deploys.
3. Search the runbooks for the matching procedure and follow it.
4. Propose one mitigation and call the tool for it. Production changes need on-call approval; if it is not
   given, say the action is awaiting approval and do not claim it ran.
5. Save a one-line summary with remember_incident.
Report: Summary, Root cause, Evidence (bullets with numbers and log lines), Action and status, Runbook (cite [n]).
Mention similar past incidents from memory if any. Base every claim on tool output."""

PAST_INCIDENTS = [
    ("2026-09-14 checkout: error spike right after deploy v2.3.0 (null shipping address in new code); rolled back to v2.2.8, "
     "errors cleared in 6 minutes.", ["checkout", "deploy", "rollback"]),
    ("2026-08-02 search: OOMKilled restarts from an unbounded autocomplete cache; restarted, then capped the cache at 50k entries.",
     ["search", "memory", "oom"]),
]


class IncidentResponder(Agent):
    def __init__(self, clients: list[MCPClient], **options):
        self.clients = clients
        super().__init__(**options)

    def close(self) -> None:
        for client in self.clients:
            client.close()


def build_local_tools(runbooks: HybridRetriever, memory: LongTermMemory) -> list:
    @tool
    def search_runbooks(query: str) -> str:
        """Searches the team's runbooks and returns numbered passages to cite.

        Args:
            query: Symptoms or the suspected cause, e.g. "connection pool exhausted".
        """
        return format_context(runbooks.search(query, 3))

    @tool
    def remember_incident(summary: str, service: str) -> str:
        """Saves a one-line incident summary to long-term memory, for future incidents.

        Args:
            summary: What happened, the cause and the fix.
            service: The affected service.
        """
        memory.remember(summary, [service])
        return "Saved."

    @tool
    def recall_incidents(query: str) -> str:
        """Finds similar past incidents in long-term memory.

        Args:
            query: The service and symptoms, e.g. "checkout error spike after deploy".
        """
        found = memory.recall(query, k=2, min_score=0.1)
        return "\n".join(f"- {m.text}" for m in found) or "No similar past incidents."

    return [search_runbooks, remember_incident, recall_incidents]


ALERT = re.compile(r"\bALRT-\d+\b", re.I)


def diagnose(alert: dict, logs: str, deploys: dict, metrics: str) -> dict:
    """The offline planner's reasoning: match the evidence to a known failure pattern."""
    service = alert["service"]
    started = alert["started"][11:16]
    if "pool exhausted" in logs.lower() or "connectionpooltimeout" in logs.lower():
        replicas = deploys["replicas"] * 2
        return {"cause": f"The {service} database connection pool is exhausted (100/100 in use) during a traffic spike, so requests wait for connections.",
                "query": "database connection pool exhausted traffic spike",
                "action": ("scale_service", {"service": service, "replicas": replicas}),
                "action_text": f"scale {service} from {deploys['replicas']} to {replicas} replicas",
                "evidence_words": ["pool exhausted", "traffic"]}
    if "oomkilled" in logs.lower() or "outofmemoryerror" in logs.lower():
        return {"cause": f"{service.title()} pods are running out of memory (OOMKilled), most likely a leak in the query cache added in {deploys['deploys'][0]['version']}.",
                "query": "memory exhaustion OOMKilled pods restart leak",
                "action": ("restart_service", {"service": service}),
                "action_text": f"restart {service} to restore capacity, then cap the cache",
                "evidence_words": ["OOMKilled", "OutOfMemoryError"]}
    latest = deploys["deploys"][0]
    if len(deploys["deploys"]) > 1 and latest["at"][11:16] <= started:
        previous = deploys["deploys"][1]["version"]
        return {"cause": f"Deploy {latest['version']} ({latest['change'].lower()}) introduced a bug: errors started minutes after it rolled out.",
                "query": "errors after a deploy rollback",
                "action": ("rollback_deploy", {"service": service, "to_version": previous}),
                "action_text": f"roll {service} back from {latest['version']} to {previous}",
                "evidence_words": ["TypeError", "deploy"]}
    return {}


def offline_planner(system: str, messages: list, tools: list) -> LLMResponse:
    task = last_user_text(messages)
    results = tool_results(messages)
    done = {name: content for name, content, _ in results}
    found = ALERT.search(task)

    if not found and "list_alerts" not in done:
        return call("list_alerts")
    alert_id = found.group(0).upper() if found else json.loads(done["list_alerts"])[0]["id"]
    if "get_alert" not in done:
        return call("get_alert", alert_id=alert_id)
    alert = json.loads(done["get_alert"])
    service = alert["service"]
    if "recall_incidents" not in done:
        return call("recall_incidents", query=f"{service} {alert['title']}")
    if "query_metrics" not in done:
        return call("query_metrics", service=service, metric=alert["metric"])
    if "search_logs" not in done:
        return call("search_logs", service=service, query="error warn oom pool", limit=8)
    if "recent_deploys" not in done:
        return call("recent_deploys", service=service)
    plan = diagnose(alert, done["search_logs"], json.loads(done["recent_deploys"]), done["query_metrics"])
    if not plan:
        return say(f"I investigated {alert_id} ({alert['title']}) but couldn't match the evidence to a known cause. Escalating to a human on-call engineer.")
    if "search_runbooks" not in done:
        return call("search_runbooks", query=plan["query"])
    action, arguments = plan["action"]
    if action not in done:
        return call(action, **arguments)
    if "remember_incident" not in done:
        return call("remember_incident", service=service, summary=f"{alert_id} {service}: {plan['cause']} Mitigation: {plan['action_text']}.")

    action_result = next((c, e) for n, c, e in results if n == action)
    status = (f"Done: {plan['action_text']}." if not action_result[1]
              else f"Awaiting approval: {plan['action_text']}. Nothing has been changed in production yet.")
    evidence_lines = [l for l in done["search_logs"].splitlines() if any(w.lower() in l.lower() for w in plan["evidence_words"])][:2]
    points = [line.split() for line in done["query_metrics"].splitlines() if len(line.split()) == 2]
    peak = max(points, key=lambda p: float(p[1]))
    runbook = re.search(r"^\[1\] (?:Runbook: )?(.+?) \(", done["search_runbooks"], re.M)
    past = done["recall_incidents"]
    memory = None if past.startswith("No similar") else past.splitlines()[0].removeprefix("- ")
    lines = [
        f"**Summary:** {alert['title']} on {service} since {alert['started'][11:16]} UTC ({alert['severity']}).",
        f"**Root cause:** {plan['cause']}",
        "**Evidence:**",
        f"- {alert['metric']} peaked at {peak[1]} at {peak[0]}.",
        *[f"- Log: `{l}`" for l in evidence_lines],
        f"**Action:** {status}",
        f"**Runbook:** {runbook.group(1).capitalize() if runbook else 'none found'} [1].",
    ]
    if memory:
        lines.append(f"**Similar past incident:** {memory}")
    return say("\n".join(lines))


def make_agent(approver=deny_all, llm=None, memory_path: str | None = None) -> IncidentResponder:
    memory = LongTermMemory(memory_path)
    for text, tags in PAST_INCIDENTS:
        memory.remember(text, tags)
    runbooks = HybridRetriever(load_markdown_dir(os.path.join(HERE, "data", "runbooks")), max_words=70)
    observability = MCPClient.python_module("agents.incident_responder.observability_server")
    operations = MCPClient.python_module("agents.incident_responder.ops_server")
    tools = ToolRegistry(observability.as_tools()
                         + operations.as_tools(requires_approval={"rollback_deploy": True, "scale_service": True, "restart_service": True})
                         + build_local_tools(runbooks, memory))
    return IncidentResponder([observability, operations], name="Incident Responder", system=SYSTEM,
                             llm=llm or choose_llm(offline_planner), tools=tools, max_steps=12,
                             approver=approver, long_term=memory)
