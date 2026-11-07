"""MCP server: read-only access to a sales database.

Run it on its own with ``python -m agents.data_analyst.server``. Any MCP client
(including Claude Desktop) can connect over stdio. Writes are impossible: the
connection is opened read-only, only single SELECT statements are accepted, and
results are capped, which is least privilege applied to an AI tool.
"""

from __future__ import annotations

import csv
import os
import re
import sqlite3

from agentkit.mcp import MCPServer

DATA = os.path.join(os.path.dirname(__file__), "data")
MAX_ROWS = 100

SCHEMA = {
    "products": "id INTEGER PRIMARY KEY, name TEXT, category TEXT, price REAL",
    "customers": "id INTEGER PRIMARY KEY, name TEXT, region TEXT",
    "orders": "id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers(id), product_id INTEGER REFERENCES products(id), quantity INTEGER, order_date TEXT",
}


def open_database() -> sqlite3.Connection:
    """Loads the CSVs into an in-memory database, then makes the connection read-only."""
    db = sqlite3.connect(":memory:", check_same_thread=False)
    for table, columns in SCHEMA.items():
        db.execute(f"CREATE TABLE {table} ({columns})")
        with open(os.path.join(DATA, f"{table}.csv"), newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        db.executemany(f"INSERT INTO {table} VALUES ({','.join('?' * len(rows[0]))})", rows[1:])
    db.commit()
    db.execute("PRAGMA query_only = ON")
    return db


def is_read_only(sql: str) -> bool:
    body = re.sub(r"--[^\n]*|/\*.*?\*/", " ", sql, flags=re.S).strip().rstrip(";").strip()
    return bool(re.match(r"(?is)^(select|with)\b", body)) and ";" not in body


def format_rows(cursor: sqlite3.Cursor, rows: list[tuple]) -> str:
    header = [d[0] for d in cursor.description]
    lines = [" | ".join(header), " | ".join("---" for _ in header)]
    for row in rows:
        lines.append(" | ".join(f"{v:,.2f}" if isinstance(v, float) else str(v) for v in row))
    return "\n".join(lines)


def build_server(db: sqlite3.Connection | None = None) -> MCPServer:
    db = db or open_database()
    server = MCPServer("sales-db", instructions="Read-only sales database. Revenue = orders.quantity * products.price. Dates are YYYY-MM-DD in 2026.")

    @server.tool
    def list_tables() -> str:
        """Lists the tables in the sales database with their row counts."""
        return "\n".join(f"{t}: {db.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]} rows" for t in SCHEMA)

    @server.tool
    def describe_table(table: str) -> str:
        """Shows a table's columns and three sample rows.

        Args:
            table: One of products, customers or orders.
        """
        if table not in SCHEMA:
            raise ValueError(f"Unknown table '{table}'. Tables: {', '.join(SCHEMA)}")
        cursor = db.execute(f"SELECT * FROM {table} LIMIT 3")
        return f"Columns: {SCHEMA[table]}\n\n{format_rows(cursor, cursor.fetchall())}"

    @server.tool
    def run_sql(query: str) -> str:
        """Runs one read-only SQL SELECT (SQLite dialect) and returns up to 100 rows as a table.

        Args:
            query: A single SELECT or WITH statement.
        """
        if not is_read_only(query):
            raise PermissionError("Only a single read-only SELECT statement is allowed")
        cursor = db.execute(query)
        rows = cursor.fetchmany(MAX_ROWS + 1)
        note = f"\n(showing the first {MAX_ROWS} rows)" if len(rows) > MAX_ROWS else ""
        return format_rows(cursor, rows[:MAX_ROWS]) + note

    @server.resource("schema://sales", "Sales database schema")
    def schema() -> str:
        return "\n".join(f"{t}({c})" for t, c in SCHEMA.items())

    return server


if __name__ == "__main__":
    build_server().run_stdio()
