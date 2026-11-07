"""Model Context Protocol (MCP): a server to expose tools and a client to use them."""

from .client import MCPClient, MCPError
from .server import PROTOCOL_VERSION, MCPServer

__all__ = ["MCPClient", "MCPError", "MCPServer", "PROTOCOL_VERSION"]
