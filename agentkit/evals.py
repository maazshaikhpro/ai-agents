"""Evals: measure whether an agent does the right thing, not just whether it runs.

A suite is a list of cases. Each case has an input and checks on the outcome:
the final answer (must or must not mention things, must cite sources) and the
trajectory (which tools were called, whether risky actions waited for approval,
how many steps it took). The runner executes every case on a fresh agent and
reports a pass rate with the reason for each failure, so a change to a prompt,
tool or model can be judged by numbers instead of impressions.

Cases live in JSON files next to each agent:

    {"id": "refund-small", "input": "...", "checks": {"contains": ["refund"], "called": ["issue_refund"]}}
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .harness import Agent, RunResult


@dataclass
class CaseResult:
    id: str
    passed: bool
    failures: list[str]
    answer: str
    tools: list[str]
    steps: int
    seconds: float


@dataclass
class Report:
    suite: str
    model: str
    results: list[CaseResult] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(r.passed for r in self.results)

    @property
    def rate(self) -> float:
        return self.passed / len(self.results) if self.results else 0.0

    def to_markdown(self) -> str:
        lines = [f"### {self.suite}", "", f"Model: `{self.model}` · **{self.passed}/{len(self.results)} passed ({self.rate:.0%})**", "",
                 "| Case | Result | Tools called | Steps | Notes |", "|---|---|---|---|---|"]
        for r in self.results:
            notes = "; ".join(r.failures).replace("|", "\\|") or "-"
            lines.append(f"| {r.id} | {'pass' if r.passed else 'FAIL'} | {', '.join(r.tools) or '-'} | {r.steps} | {notes} |")
        return "\n".join(lines) + "\n"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def grade(run: RunResult, checks: dict[str, Any]) -> list[str]:
    """Applies each check to a run and returns what failed (empty means pass)."""
    failures: list[str] = []
    answer = _norm(run.answer)
    called = [c.name for c in run.tool_calls]
    for phrase in checks.get("contains", []):
        if _norm(phrase) not in answer:
            failures.append(f"answer is missing '{phrase}'")
    for options in checks.get("contains_any", []):
        if not any(_norm(o) in answer for o in options):
            failures.append(f"answer mentions none of {options}")
    for phrase in checks.get("not_contains", []):
        if _norm(phrase) in answer:
            failures.append(f"answer should not mention '{phrase}'")
    for name in checks.get("called", []):
        if name not in called:
            failures.append(f"did not call {name}")
    for name in checks.get("not_called", []):
        ran = [c for c in run.tool_calls if c.name == name and c.approved is not False and not c.is_error]
        if ran:
            failures.append(f"should not have run {name}")
    for name in checks.get("needs_approval", []):
        records = run.called(name)
        if not records or any(r.approved is None for r in records):
            failures.append(f"{name} should have asked for approval")
    if checks.get("cites") and not re.search(r"\[\d+\]", run.answer):
        failures.append("answer has no [n] citations")
    if "max_steps" in checks and run.steps > checks["max_steps"]:
        failures.append(f"took {run.steps} steps, limit {checks['max_steps']}")
    if checks.get("no_tool_errors") and any(c.is_error and c.approved is not False for c in run.tool_calls):
        failures.append("a tool call failed")
    if "matches" in checks and not re.search(checks["matches"], run.answer, re.I | re.S):
        failures.append(f"answer does not match /{checks['matches']}/")
    if run.stop != "answered" and not checks.get("allow_stop"):
        failures.append(f"stopped early: {run.stop}")
    return failures


def load_cases(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def run_suite(name: str, make_agent: Callable[[], Agent], cases: list[dict[str, Any]], on_case: Callable[[CaseResult], None] | None = None) -> Report:
    """Runs each case on a fresh agent, so cases cannot affect each other."""
    report: Report | None = None
    for case in cases:
        agent = make_agent()
        if report is None:
            report = Report(name, agent.llm.name)
        started = time.time()
        run = agent.run(case["input"])
        failures = grade(run, case.get("checks", {}))
        result = CaseResult(case["id"], not failures, failures, run.answer, [c.name for c in run.tool_calls], run.steps, round(time.time() - started, 3))
        report.results.append(result)
        if on_case:
            on_case(result)
        close = getattr(agent, "close", None)
        if callable(close):
            close()
    return report or Report(name, "none")
