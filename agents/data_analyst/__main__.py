import sys

from agentkit.cli import main

from .agent import make_agent

sys.exit(main(make_agent, "Answer questions about the sales database through MCP.", "What are the top 3 products by revenue?"))
