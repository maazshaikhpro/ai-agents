# Incident Responder (MCP, RAG, memory and approvals)

An on-call SRE agent. Given an alert, it gathers evidence, finds the root cause, follows the matching runbook and proposes a fix, but it never changes production without approval.

```bash
python3 -m agents.incident_responder "Investigate alert ALRT-101" --trace
python3 -m agents.incident_responder "Investigate alert ALRT-202" --approve
```

**What it uses**

- **Two MCP servers.** `observability_server.py` is read-only: alerts, metrics, logs and deploys. `ops_server.py` takes production actions: rollback, scale and restart. Every operations tool is marked as needing approval.
- **RAG over runbooks** in `data/runbooks`, cited in the report.
- **Long-term memory.** `recall_incidents` finds similar past incidents, and `remember_incident` saves a summary of this one for next time.

**Scenarios** (in `data/scenarios.json`)

| Alert | Service | Root cause | Proposed action |
|---|---|---|---|
| ALRT-101 | checkout | Bug in deploy v2.4.0 | Roll back to v2.3.9 |
| ALRT-202 | payments | Database connection pool exhausted by a traffic spike | Scale from 4 to 8 replicas |
| ALRT-303 | search | Memory leak in a new cache (OOMKilled) | Restart, then cap the cache |

The report gives a summary, root cause, evidence (metric peaks and log lines), the action and its approval status, the runbook it followed, and any similar past incident.
