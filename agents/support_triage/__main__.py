import sys

from agentkit.cli import main

from .agent import make_agent

sys.exit(main(make_agent, "Resolve a customer support ticket.", "My headphones from order ORD-1001 arrived broken. Can I get a refund?"))
