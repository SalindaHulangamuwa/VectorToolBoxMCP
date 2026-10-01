"""Vector Toolbox MCP - a pluggable MCP server for vector databases."""

from __future__ import annotations

try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _version

    __version__ = _version("vector-toolbox-mcp")
except PackageNotFoundError:  # running from a source tree without installing
    __version__ = "0.0.0+source"

from .core.registry import get_backend, list_backends, register_backend

__all__ = ["__version__", "get_backend", "list_backends", "register_backend"]
