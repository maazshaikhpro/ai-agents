import sys

from agentkit.cli import main

from .agent import make_agent

sys.exit(main(make_agent, "Investigate a production alert and propose a fix.", "Investigate alert ALRT-101"))
