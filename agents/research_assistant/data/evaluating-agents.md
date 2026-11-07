# Evaluating AI agents

An evaluation, or eval, measures whether an AI system does what it should. Without evals, changes to prompts, tools or models are judged by a few hand-picked examples, which hides regressions.

An eval suite is a set of test cases, each with an input and a way to grade the output. A golden dataset is a curated set of such cases with known correct answers, often built from real user requests and past failures.

There are three common kinds of grader. Code-based graders check things exactly, such as whether an answer contains a number or whether a tool was called. Model-based graders, often called LLM-as-judge, ask a language model to score qualities like helpfulness or faithfulness against a rubric. Human graders review samples where judgement matters most.

Agents need trajectory evals as well as answer evals. A trajectory eval checks the path: which tools were called, in what order, with what arguments, and whether risky actions waited for approval. An agent can reach a right answer by an unsafe path.

Evals should run on every change, like unit tests, and results should be tracked over time. A falling pass rate is a regression to investigate before release.
