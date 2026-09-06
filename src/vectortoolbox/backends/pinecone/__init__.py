from ...core.registry import register_backend
from .backend import PineconeBackend

register_backend("pinecone", PineconeBackend)


def get_pinecone_backend() -> PineconeBackend:
    from ...core.registry import get_backend

    backend = get_backend("pinecone")
    assert isinstance(backend, PineconeBackend)
    return backend


__all__ = ["PineconeBackend", "get_pinecone_backend"]
