"""Data Analyst: answers business questions by querying a database through MCP.

The agent never touches the database directly. It starts the sales MCP server
as a subprocess, discovers its tools with ``tools/list`` and calls them with
``tools/call``, so the same server could serve Claude Desktop or any MCP host.
"""

from __future__ import annotations

import re

from agentkit import Agent, ToolRegistry, choose_llm, deny_all
from agentkit.llm import LLMResponse, call, last_user_text, say, tool_results
from agentkit.mcp import MCPClient

SYSTEM = """You are a data analyst with read-only SQL access to a sales database through MCP tools.
- Start by listing tables and describing the ones you need. Revenue is orders.quantity * products.price.
- Write one SELECT at a time; if a query fails, read the error and fix it.
- Answer with the numbers, a one-line takeaway, and the SQL you used in a code block.
- Never guess numbers. If the data cannot answer the question, say so."""

REVENUE = "SUM(o.quantity * p.price)"
JOIN = "FROM orders o JOIN products p ON p.id = o.product_id JOIN customers c ON c.id = o.customer_id"
MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"], 1)}
REGIONS = ("north", "south", "east", "west")


def plan_sql(question: str) -> tuple[str, str] | None:
    """Offline stand-in for the model's SQL writing: question patterns to SQL.

    Returns (sql, what the result means), or None if no pattern fits.
    """
    q = question.lower()
    where = []
    region = next((r for r in REGIONS if re.search(rf"\b{r}\b", q)), None)
    if region:
        where.append(f"c.region = '{region.title()}'")
    month = next((n for m, n in MONTHS.items() if m in q), None)
    if month:
        where.append(f"substr(o.order_date, 6, 2) = '{month:02d}'")
    category = next((c for c in ("computers", "displays", "accessories", "audio", "wearables") if c in q), None)
    if category:
        where.append(f"p.category = '{category.title()}'")
    filters = f" WHERE {' AND '.join(where)}" if where else ""
    top = re.search(r"\btop\s+(\d+)", q)
    limit = int(top.group(1)) if top else 5

    if re.search(r"how many (orders|purchases)", q):
        return f"SELECT COUNT(*) AS orders {JOIN}{filters}", "number of orders"
    if re.search(r"how many customers", q):
        return ("SELECT COUNT(DISTINCT c.id) AS customers " + JOIN + filters if where else "SELECT COUNT(*) AS customers FROM customers"), "number of customers"
    if re.search(r"(revenue|sales) by (region)", q):
        return f"SELECT c.region, ROUND({REVENUE}, 2) AS revenue {JOIN}{filters} GROUP BY c.region ORDER BY revenue DESC", "revenue by region"
    if re.search(r"(revenue|sales) by (month)|monthly (revenue|sales)|trend", q):
        return (f"SELECT substr(o.order_date, 1, 7) AS month, ROUND({REVENUE}, 2) AS revenue {JOIN}{filters} GROUP BY month ORDER BY month",
                "revenue by month")
    if re.search(r"(revenue|sales) by categor", q):
        return f"SELECT p.category, ROUND({REVENUE}, 2) AS revenue {JOIN}{filters} GROUP BY p.category ORDER BY revenue DESC", "revenue by category"
    if re.search(r"(top|best)[\w\s]*customer", q):
        return (f"SELECT c.name, c.region, ROUND({REVENUE}, 2) AS revenue {JOIN}{filters} GROUP BY c.id ORDER BY revenue DESC LIMIT {limit if top else 1}",
                "top customers by revenue")
    if re.search(r"(top|best)[\w\s-]*(product|seller|item)", q):
        by_units = "unit" in q or "quantity" in q
        measure = "SUM(o.quantity) AS units" if by_units else f"ROUND({REVENUE}, 2) AS revenue"
        return (f"SELECT p.name, {measure} {JOIN}{filters} GROUP BY p.id ORDER BY {'units' if by_units else 'revenue'} DESC LIMIT {limit}",
                f"top products by {'units sold' if by_units else 'revenue'}")
    if re.search(r"average order value|aov", q):
        return f"SELECT ROUND({REVENUE} * 1.0 / COUNT(DISTINCT o.id), 2) AS average_order_value {JOIN}{filters}", "average order value"
    if re.search(r"(total )?(revenue|sales)", q):
        return f"SELECT ROUND({REVENUE}, 2) AS revenue {JOIN}{filters}", "total revenue"
    return None


def offline_planner(system: str, messages: list, tools: list) -> LLMResponse:
    question = last_user_text(messages)
    results = tool_results(messages)
    names = [r[0] for r in results]
    if "list_tables" not in names:
        return call("list_tables")
    planned = plan_sql(question)
    if not planned:
        return say("I can't answer that from the sales data in offline mode. The database has products, customers and orders, so try "
                   "questions about revenue, orders, customers, products, regions, categories or months. With ANTHROPIC_API_KEY set, "
                   "Claude writes the SQL for any question.")
    sql, meaning = planned
    if "run_sql" not in names:
        return call("run_sql", query=sql)
    content, error = next((c, e) for n, c, e in reversed(results) if n == "run_sql")
    if error:
        return say(f"The query failed: {content}")
    lines = [l for l in content.splitlines() if l and not l.startswith("---")]
    header, rows = lines[0].split(" | "), [l.split(" | ") for l in lines[1:]]
    if len(rows) == 1 and len(header) == 1:
        value = rows[0][0]
        money = "revenue" in meaning or "value" in meaning
        summary = f"The {meaning} is {'$' if money else ''}{value}."
    else:
        table = "\n".join(f"- {' · '.join(r)}" for r in rows)
        summary = f"Here are the {meaning}:\n{table}\n\nTop result: {rows[0][0].rstrip('.')}."
    return say(f"{summary}\n\n```sql\n{sql}\n```")


class DataAnalyst(Agent):
    """An agent that owns an MCP connection and closes it when done."""

    def __init__(self, client: MCPClient, **options):
        self.client = client
        super().__init__(tools=ToolRegistry(client.as_tools()), **options)

    def close(self) -> None:
        self.client.close()


def make_agent(approver=deny_all, llm=None) -> DataAnalyst:
    client = MCPClient.python_module("agents.data_analyst.server")
    return DataAnalyst(client, name="Data Analyst", system=SYSTEM + "\n\nServer notes: " + client.instructions,
                       llm=llm or choose_llm(offline_planner), max_steps=10, approver=approver)
