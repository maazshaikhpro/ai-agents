# Data Analyst (MCP)

Answers business questions about a sales database (products, customers and 400 orders from January to June 2026) by writing SQL. The agent has no database connection of its own; it uses the `sales-db` MCP server.

```bash
python3 -m agents.data_analyst "What was total revenue in March?"
python3 -m agents.data_analyst "Revenue by region"
```

**How it works**

1. `make_agent` starts `server.py` as a subprocess and performs the MCP handshake (`initialize`).
2. It discovers the tools with `tools/list` and wraps each one so the harness can call it like a local tool.
3. The agent lists tables, writes a SELECT, runs it with `tools/call`, and answers with the numbers and the SQL it used.

**Least privilege:** the database connection is read-only (`PRAGMA query_only`), only single SELECT or WITH statements are accepted, and results are capped at 100 rows.

**Tools (over MCP):** `list_tables`, `describe_table`, `run_sql`, plus the resource `schema://sales`

**Use the server elsewhere:** `python3 -m agents.data_analyst.server` speaks MCP over stdio, so Claude Desktop or any MCP host can use it.
