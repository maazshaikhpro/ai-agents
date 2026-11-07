import sys

from agentkit.cli import main

from .agent import make_agent

sys.exit(main(make_agent, "Answer questions from the AI knowledge base, with citations.", "How does hybrid search combine keyword and vector results?"))
