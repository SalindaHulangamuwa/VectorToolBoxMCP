"""Compatibility shim across MCP Python SDK versions.

In SDK 2.x ``FastMCP`` was renamed to ``MCPServer``. The decorator API is the
same, so one alias covers both.
"""

from __future__ import annotations

try:  # mcp >= 2.0
    from mcp.server.mcpserver import MCPServer as MCPServerType
except ImportError:  # pragma: no cover - mcp 1.x
    from mcp.server.fastmcp import FastMCP as MCPServerType  # type: ignore[assignment]

__all__ = ["MCPServerType"]
