"""Chroma backend: named clients (ephemeral / persistent / http / cloud), collections,
records, conditional transactions, metadata filtering and full-text search."""

from ...core.registry import register_backend
from .backend import ChromaBackend

register_backend("chroma", ChromaBackend)


def get_chroma_backend() -> ChromaBackend:
    from ...core.registry import get_backend

    backend = get_backend("chroma")
    assert isinstance(backend, ChromaBackend)
    return backend


__all__ = ["ChromaBackend", "get_chroma_backend"]
