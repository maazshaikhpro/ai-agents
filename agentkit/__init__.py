"""agentkit: a small, dependency-free toolkit for building AI agents.

Harness (agent loop), tool calling, RAG, memory, evals, tracing and MCP, in plain
Python so every part is readable.
"""

from .evals import Report, grade, load_cases, run_suite
from .harness import Agent, RunResult, ask_in_terminal, deny_all
from .llm import AnthropicLLM, LLMResponse, ScriptedLLM, ToolCall, choose_llm
from .memory import ConversationMemory, LongTermMemory
from .rag import Document, HybridRetriever, format_context, load_markdown_dir
from .tools import Tool, ToolRegistry, ToolResult, tool
from .tracing import Tracer

__version__ = "0.1.0"

__all__ = [
    "Agent", "AnthropicLLM", "ConversationMemory", "Document", "HybridRetriever", "LLMResponse", "LongTermMemory",
    "Report", "RunResult", "ScriptedLLM", "Tool", "ToolCall", "ToolRegistry", "ToolResult", "Tracer", "ask_in_terminal",
    "choose_llm", "deny_all", "format_context", "grade", "load_cases", "load_markdown_dir", "run_suite", "tool",
]
