import unittest
from typing import Literal

from agentkit.tools import ToolRegistry, tool, validate


@tool
def add(a: int, b: int = 2) -> int:
    """Adds two numbers.

    Args:
        a: The first number.
        b: The second number.
    """
    return a + b


@tool(requires_approval=lambda args: args["amount"] > 100)
def pay(amount: float, method: Literal["card", "bank"]) -> str:
    """Pays an amount."""
    return f"paid {amount} by {method}"


@tool
def boom() -> str:
    """Always fails."""
    raise RuntimeError("kaboom")


class ToolTests(unittest.TestCase):
    def test_schema_comes_from_signature_and_docstring(self):
        self.assertEqual(add.description, "Adds two numbers.")
        self.assertEqual(add.input_schema["properties"]["a"], {"type": "integer", "description": "The first number."})
        self.assertEqual(add.input_schema["required"], ["a"])
        self.assertEqual(add.input_schema["properties"]["b"]["default"], 2)
        self.assertEqual(pay.input_schema["properties"]["method"], {"type": "string", "enum": ["card", "bank"]})

    def test_registry_validates_before_running(self):
        registry = ToolRegistry([add, pay])
        self.assertEqual(registry.call("add", {"a": 1}).content, "3")
        self.assertIn("missing required argument 'a'", registry.call("add", {}).content)
        self.assertIn("should be integer", registry.call("add", {"a": "1"}).content)
        self.assertIn("must be one of", registry.call("pay", {"amount": 5.0, "method": "cash"}).content)
        self.assertIn("unknown argument", registry.call("add", {"a": 1, "c": 2}).content)
        self.assertTrue(registry.call("nope", {}).is_error)

    def test_exceptions_become_error_results(self):
        result = ToolRegistry([boom]).call("boom", {})
        self.assertTrue(result.is_error)
        self.assertIn("RuntimeError: kaboom", result.content)

    def test_approval_can_depend_on_arguments(self):
        self.assertFalse(pay.needs_approval({"amount": 50}))
        self.assertTrue(pay.needs_approval({"amount": 500}))

    def test_validate_bool_is_not_integer(self):
        self.assertIsNotNone(validate({"a": True}, add.input_schema))


if __name__ == "__main__":
    unittest.main()
