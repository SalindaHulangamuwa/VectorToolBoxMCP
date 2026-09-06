"""Embedding provider registry."""

from __future__ import annotations

from functools import lru_cache

from ..config import get_settings
from ..errors import ConfigurationError
from .base import DenseEmbedder, SparseEmbedder
from .providers import (
    CohereEmbedder,
    HuggingFaceEmbedder,
    OpenAIEmbedder,
    PineconeDenseEmbedder,
    PineconeSparseEmbedder,
)

DENSE_PROVIDERS = {
    "pinecone": PineconeDenseEmbedder,
    "openai": OpenAIEmbedder,
    "cohere": CohereEmbedder,
    "huggingface": HuggingFaceEmbedder,
}

SPARSE_PROVIDERS = {
    "pinecone": PineconeSparseEmbedder,
}


@lru_cache(maxsize=16)
def _build_dense(provider: str, model: str, dimension: int | None) -> DenseEmbedder:
    if provider not in DENSE_PROVIDERS:
        raise ConfigurationError(
            f"Unknown embedding provider {provider!r}. Available: {', '.join(DENSE_PROVIDERS)}."
        )
    return DENSE_PROVIDERS[provider](model=model, dimension=dimension)


@lru_cache(maxsize=8)
def _build_sparse(provider: str, model: str) -> SparseEmbedder:
    if provider not in SPARSE_PROVIDERS:
        raise ConfigurationError(
            f"Unknown sparse provider {provider!r}. Available: {', '.join(SPARSE_PROVIDERS)}."
        )
    return SPARSE_PROVIDERS[provider](model=model)


def get_dense_embedder(
    provider: str | None = None,
    model: str | None = None,
    dimension: int | None = None,
) -> DenseEmbedder:
    settings = get_settings()
    return _build_dense(
        provider or settings.embed_provider,
        model or settings.embed_model,
        dimension if dimension is not None else settings.embed_dimension,
    )


def get_sparse_embedder(provider: str | None = None, model: str | None = None) -> SparseEmbedder:
    settings = get_settings()
    provider = provider or settings.sparse_provider or "pinecone"
    model = model or settings.sparse_model or "pinecone-sparse-english-v0"
    return _build_sparse(provider, model)


def clear_embedder_cache() -> None:
    _build_dense.cache_clear()
    _build_sparse.cache_clear()


__all__ = [
    "DenseEmbedder",
    "SparseEmbedder",
    "DENSE_PROVIDERS",
    "SPARSE_PROVIDERS",
    "get_dense_embedder",
    "get_sparse_embedder",
    "clear_embedder_cache",
]
