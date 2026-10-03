"""Weaviate backend: named clients (local / custom / cloud / embedded), typed collections
with named vectors, tenants, objects and references, hybrid / semantic / keyword search,
aggregation and filters."""

from ...core.registry import register_backend
from .backend import WeaviateBackend

register_backend("weaviate", WeaviateBackend)


def get_weaviate_backend() -> WeaviateBackend:
    from ...core.registry import get_backend

    backend = get_backend("weaviate")
    assert isinstance(backend, WeaviateBackend)
    return backend


__all__ = ["WeaviateBackend", "get_weaviate_backend"]
