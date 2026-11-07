import json
import unittest

from agentkit.mcp import PROTOCOL_VERSION, MCPClient, MCPError, MCPServer
from agents.data_analyst.server import build_server, is_read_only, open_database


class ServerProtocolTests(unittest.TestCase):
    def setUp(self):
        self.server = MCPServer("test")

        @self.server.tool
        def echo(text: str) -> str:
            """Echoes text."""
            return text

    def rpc(self, method, params=None, msg_id=1):
        return self.server.handle({"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params or {}})

    def test_initialize_and_list(self):
        info = self.rpc("initialize", {"protocolVersion": PROTOCOL_VERSION, "capabilities": {}, "clientInfo": {"name": "t"}})["result"]
        self.assertEqual(info["protocolVersion"], PROTOCOL_VERSION)
        self.assertIn("tools", info["capabilities"])
        tools = self.rpc("tools/list")["result"]["tools"]
        self.assertEqual(tools[0]["name"], "echo")
        self.assertEqual(tools[0]["inputSchema"]["required"], ["text"])

    def test_call_and_errors(self):
        result = self.rpc("tools/call", {"name": "echo", "arguments": {"text": "hi"}})["result"]
        self.assertEqual(result, {"content": [{"type": "text", "text": "hi"}], "isError": False})
        self.assertTrue(self.rpc("tools/call", {"name": "echo", "arguments": {}})["result"]["isError"])
        self.assertEqual(self.rpc("tools/call", {"name": "nope"})["error"]["code"], -32602)
        self.assertEqual(self.rpc("nope/nope")["error"]["code"], -32601)
        self.assertIsNone(self.server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}))


class StdioRoundTripTests(unittest.TestCase):
    def test_client_talks_to_a_real_server_process(self):
        with MCPClient.python_module("agents.data_analyst.server") as client:
            self.assertEqual(client.server_info["name"], "sales-db")
            names = [t["name"] for t in client.list_tools()]
            self.assertEqual(names, ["list_tables", "describe_table", "run_sql"])
            self.assertIn("orders: 400 rows", client.call_tool("list_tables", {}).content)
            self.assertIn("products(", client.read_resource("schema://sales"))
            tools = {t.name: t for t in client.as_tools()}
            self.assertIn("COUNT", tools["run_sql"].fn(query="SELECT COUNT(*) FROM products").content.upper())
            denied = client.call_tool("run_sql", {"query": "DELETE FROM orders"})
            self.assertTrue(denied.is_error)
            self.assertIn("read-only", denied.content)

    def test_dead_server_raises(self):
        with self.assertRaises(MCPError):
            MCPClient(["python3", "-c", "import sys; sys.exit(0)"], timeout=5)


class ReadOnlyTests(unittest.TestCase):
    def test_only_single_selects_pass(self):
        self.assertTrue(is_read_only("SELECT 1"))
        self.assertTrue(is_read_only("with t as (select 1) select * from t;"))
        for bad in ["DELETE FROM orders", "SELECT 1; DROP TABLE orders", "UPDATE products SET price = 0", "/* x */ INSERT INTO t VALUES (1)"]:
            self.assertFalse(is_read_only(bad), bad)

    def test_database_rejects_writes_even_if_parsing_were_fooled(self):
        db = open_database()
        self.addCleanup(db.close)
        db_call = build_server(db).registry.get("run_sql").fn
        with self.assertRaises(Exception):
            db_call(query="WITH x AS (SELECT 1) INSERT INTO products VALUES (99, 'x', 'y', 1)")


if __name__ == "__main__":
    unittest.main()
