"""The contract every vector-database backend implements.

The point of this file is the multi-database goal: Pinecone is the first
backend, but Qdrant / Weaviate / Milvus / pgvector should slot in behind the
same surface. Anything genuinely Pinecone-specific (Lucene query strings,
integrated inference, read capacity) is exposed as backend-specific tools
rather than being forced into this ABC.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .types import Hit, IndexCapabilities


class VectorStoreBackend(ABC):
    """Minimal capability surface shared by all vector stores."""

    #: Short registry name, e.g. "pinecone".
    name: str

    # -- index lifecycle ---------------------------------------------------
    @abstractmethod
    def list_indexes(self) -> list[dict[str, Any]]: ...

    @abstractmethod
    def describe_index(self, index: str) -> dict[str, Any]: ...

    @abstractmethod
    def capabilities(self, index: str) -> IndexCapabilities:
        """Report which search modes and fields this index supports."""

    @abstractmethod
    def delete_index(self, index: str) -> dict[str, Any]: ...

    # -- namespaces / partitions ------------------------------------------
    @abstractmethod
    def list_namespaces(self, index: str) -> list[dict[str, Any]]: ...

    @abstractmethod
    def delete_namespace(self, index: str, namespace: str) -> dict[str, Any]: ...

    # -- records -----------------------------------------------------------
    @abstractmethod
    def upsert(
        self, index: str, namespace: str, records: list[dict[str, Any]]
    ) -> dict[str, Any]: ...

    @abstractmethod
    def fetch(
        self, index: str, namespace: str, ids: list[str] | None = None
    ) -> list[dict[str, Any]]: ...

    @abstractmethod
    def delete(
        self,
        index: str,
        namespace: str,
        ids: list[str] | None = None,
        filter: dict[str, Any] | None = None,
        delete_all: bool = False,
    ) -> dict[str, Any]: ...

    # -- retrieval ---------------------------------------------------------
    @abstractmethod
    def search(
        self,
        index: str,
        namespace: str,
        *,
        mode: str,
        query: str | None = None,
        vector: list[float] | None = None,
        fields: list[str] | None = None,
        top_k: int = 10,
        filter: dict[str, Any] | None = None,
        include_fields: list[str] | None = None,
    ) -> list[Hit]: ...

    @abstractmethod
    def stats(self, index: str) -> dict[str, Any]: ...
