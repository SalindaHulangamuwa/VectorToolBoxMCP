"""Cross-backend MCP tools.

Anything that describes the server as a whole lives here, not in a backend's
tool module, so backends never import each other. Each backend contributes to
``vectortoolbox_status`` by calling ``register_status_provider`` from its own
``register(mcp)``; this module just collects whatever was registered.
"""

from __future__ import annotations

from typing import Any, Callable

from ._mcp_compat import MCPServerType
from .config import get_settings

StatusProvider = Callable[[], dict[str, Any]]

_STATUS_PROVIDERS: dict[str, StatusProvider] = {}


def register_status_provider(backend: str, provider: StatusProvider) -> None:
    """Let a backend add its own section to ``vectortoolbox_status``."""
    _STATUS_PROVIDERS[backend] = provider


def _backend_status(name: str, provider: StatusProvider) -> dict[str, Any]:
    # One broken backend (missing package, bad config) must not hide the rest.
    try:
        return provider()
    except Exception as exc:
        return {"error": type(exc).__name__, "message": str(exc)[:200]}


def register(mcp: MCPServerType) -> None:
    @mcp.tool()
    def vectortoolbox_status() -> dict[str, Any]:
        """Report configuration: registered backends, embedding providers, read-only mode,
        plus one section per backend (Pinecone key/region/TTL, Chroma clients/version)."""
        from .core.registry import list_backends
        from .embeddings import DENSE_PROVIDERS, SPARSE_PROVIDERS

        settings = get_settings()
        out: dict[str, Any] = {
            "backends": list_backends(),
            "default_backend": settings.default_backend,
            "read_only": settings.read_only,
            "dense_embedding_providers": sorted(DENSE_PROVIDERS),
            "sparse_embedding_providers": sorted(SPARSE_PROVIDERS),
            "default_dense_embedder": f"{settings.embed_provider}/{settings.embed_model}",
            "default_sparse_embedder": f"{settings.sparse_provider}/{settings.sparse_model}",
        }
        for name, provider in sorted(_STATUS_PROVIDERS.items()):
            out[name] = _backend_status(name, provider)
        return out
