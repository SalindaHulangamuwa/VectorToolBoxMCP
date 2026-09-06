"""Lazily-constructed Pinecone client and index-handle cache."""

from __future__ import annotations

from typing import Any

from ...config import get_settings
from ...errors import ConfigurationError

_CLIENT: Any = None
_INDEX_HANDLES: dict[str, Any] = {}
_HOSTS: dict[str, str] = {}


def get_client():
    global _CLIENT
    if _CLIENT is None:
        try:
            from pinecone import Pinecone
        except ImportError as exc:  # pragma: no cover
            raise ConfigurationError("The pinecone package is not installed.") from exc
        api_key = get_settings().pinecone_api_key
        if not api_key:
            raise ConfigurationError(
                "PINECONE_API_KEY is not set. Add it to your .env or the MCP server env block."
            )
        _CLIENT = Pinecone(api_key=api_key, source_tag="vector-toolbox-mcp")
    return _CLIENT


def get_index(name: str):
    """Return a cached index handle, resolving the host once per index."""
    if name not in _INDEX_HANDLES:
        pc = get_client()
        _INDEX_HANDLES[name] = pc.Index(name=name)
    return _INDEX_HANDLES[name]


def index_host(name: str) -> str | None:
    if name not in _HOSTS:
        try:
            _HOSTS[name] = get_client().describe_index(name=name).host
        except Exception:
            return None
    return _HOSTS.get(name)


def reset_client_cache() -> None:
    global _CLIENT
    _CLIENT = None
    _INDEX_HANDLES.clear()
    _HOSTS.clear()
