"""Pluggable embedding providers.

Storing vectors is decoupled from producing them: a tool takes
``embed_provider`` / ``embed_model`` and this layer resolves it. Providers are
imported lazily so the server starts without openai/cohere/torch installed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence


class DenseEmbedder(ABC):
    provider: str
    model: str

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...

    @property
    def dimension(self) -> int | None:
        return getattr(self, "_dimension", None)

    def describe(self) -> dict[str, Any]:
        return {"provider": self.provider, "model": self.model, "dimension": self.dimension}


class SparseEmbedder(ABC):
    provider: str
    model: str

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[dict[str, list]]:
        """Return [{'indices': [...], 'values': [...]}, ...]."""

    @abstractmethod
    def embed_query(self, text: str) -> dict[str, list]: ...

    def describe(self) -> dict[str, Any]:
        return {"provider": self.provider, "model": self.model, "kind": "sparse"}
