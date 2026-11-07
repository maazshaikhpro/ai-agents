import unittest

from agentkit import Agent, LongTermMemory, ScriptedLLM, ToolRegistry, tool
from agentkit.llm import LLMResponse, ToolCall, call, last_user_text, say, tool_results


@tool
def lookup(key: str) -> str:
    """Looks up a value."""
    return {"a": "apple"}.get(key) or (_ for _ in ()).throw(KeyError(key))


@tool(requires_approval=True)
def delete_everything() -> str:
    """Deletes everything."""
    return "deleted"


def lookup_then_answer(system, messages, tools):
    results = tool_results(messages)
    if not results:
        return call("lookup", key="a")
    return say(f"The answer is {results[-1][1]}.")


class HarnessTests(unittest.TestCase):
    def agent(self, policy, **options):
        return Agent("test", "be helpful", ScriptedLLM(policy), ToolRegistry([lookup, delete_everything]), **options)

    def test_loop_runs_tools_and_returns_the_answer(self):
        result = self.agent(lookup_then_answer).run("what is a?")
        self.assertEqual(result.answer, "The answer is apple.")
        self.assertEqual([c.name for c in result.tool_calls], ["lookup"])
        self.assertEqual(result.steps, 2)
        self.assertEqual(result.stop, "answered")
        self.assertEqual([e.kind for e in result.trace.events], ["run_start", "llm", "tool", "llm", "run_end"])

    def test_tool_errors_go_back_to_the_model(self):
        def policy(system, messages, tools):
            results = tool_results(messages)
            if not results:
                return call("lookup", key="zzz")
            name, content, error = results[-1]
            return say("recovered" if error and "KeyError" in content else "missed")
        self.assertEqual(self.agent(policy).run("x").answer, "recovered")

    def test_step_budget_stops_a_looping_agent(self):
        result = self.agent(lambda s, m, t: call("lookup", key="a"), max_steps=3).run("loop")
        self.assertEqual(result.stop, "max_steps")
        self.assertEqual(result.steps, 3)

    def test_tool_call_budget(self):
        many = lambda s, m, t: LLMResponse(tool_calls=[ToolCall("", "lookup", {"key": "a"}) for _ in range(5)])
        self.assertEqual(self.agent(many, max_tool_calls=4).run("x").stop, "max_tool_calls")

    def test_risky_tools_wait_for_approval(self):
        def policy(system, messages, tools):
            results = tool_results(messages)
            return call("delete_everything") if not results else say(results[-1][1])
        denied = self.agent(policy).run("clean up")
        self.assertIs(denied.tool_calls[0].approved, False)
        self.assertIn("needs human approval", denied.answer)
        approved = self.agent(policy, approver=lambda name, args: True).run("clean up")
        self.assertEqual(approved.answer, "deleted")
        self.assertIs(approved.tool_calls[0].approved, True)

    def test_long_term_memory_is_added_to_the_task(self):
        memory = LongTermMemory()
        memory.remember("The billing service owner is Ana.", ["billing"])
        seen = []
        agent = self.agent(lambda s, m, t: (seen.append(last_user_text(m)), say("ok"))[1], long_term=memory)
        agent.run("who owns the billing service?")
        self.assertIn("The billing service owner is Ana.", seen[0])

    def test_model_failures_are_reported_not_raised(self):
        def broken(system, messages, tools):
            raise RuntimeError("model down")
        result = self.agent(broken).run("x")
        self.assertEqual(result.stop, "error")
        self.assertIn("model down", result.answer)


if __name__ == "__main__":
    unittest.main()
