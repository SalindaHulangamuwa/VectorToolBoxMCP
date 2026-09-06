"""Vector Toolbox MCP - a pluggable MCP server for vector databases."""

from __future__ import annotations

__version__ = "0.1.0"

from .core.registry import get_backend, list_backends, register_backend

__all__ = ["__version__", "get_backend", "list_backends", "register_backend"]
