"""Concrete embedding providers."""

from __future__ import annotations

from typing import Sequence

from ..config import get_settings
from ..errors import ConfigurationError
from .base import DenseEmbedder, SparseEmbedder


# --------------------------------------------------------------------------
# Pinecone hosted inference
# --------------------------------------------------------------------------
class PineconeDenseEmbedder(DenseEmbedder):
    provider = "pinecone"

    def __init__(self, model: str = "llama-text-embed-v2", dimension: int | None = None):
        from ..backends.pinecone.client import get_client

        self.model = model
        self._dimension = dimension
        self._pc = get_client()

    def _embed(self, texts: Sequence[str], input_type: str) -> list[list[float]]:
        params = {"input_type": input_type, "truncate": "END"}
        if self._dimension:
            params["dimension"] = self._dimension
        result = self._pc.inference.embed(model=self.model, inputs=list(texts), parameters=params)
        vectors = [list(item["values"]) for item in result.data]
        if vectors and self._dimension is None:
            self._dimension = len(vectors[0])
        return vectors

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, "passage")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "query")[0]


class PineconeSparseEmbedder(SparseEmbedder):
    provider = "pinecone"

    def __init__(self, model: str = "pinecone-sparse-english-v0"):
        from ..backends.pinecone.client import get_client

        self.model = model
        self._pc = get_client()

    def _embed(self, texts: Sequence[str], input_type: str) -> list[dict[str, list]]:
        result = self._pc.inference.embed(
            model=self.model,
            inputs=list(texts),
            parameters={"input_type": input_type, "truncate": "END"},
        )
        out = []
        for item in result.data:
            out.append(
                {
                    "indices": list(item["sparse_indices"]),
                    "values": list(item["sparse_values"]),
                }
            )
        return out

    def embed_documents(self, texts: Sequence[str]) -> list[dict[str, list]]:
        return self._embed(texts, "passage")

    def embed_query(self, text: str) -> dict[str, list]:
        return self._embed([text], "query")[0]


# --------------------------------------------------------------------------
# OpenAI
# --------------------------------------------------------------------------
class OpenAIEmbedder(DenseEmbedder):
    provider = "openai"

    def __init__(self, model: str = "text-embedding-3-small", dimension: int | None = None):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ConfigurationError(
                "The openai package is not installed. Install with: pip install 'vector-toolbox-mcp[openai]'"
            ) from exc
        api_key = get_settings().openai_api_key
        if not api_key:
            raise ConfigurationError("OPENAI_API_KEY is not set.")
        self.model = model
        self._dimension = dimension
        self._client = OpenAI(api_key=api_key)

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        kwargs = {"model": self.model, "input": list(texts)}
        if self._dimension:
            kwargs["dimensions"] = self._dimension
        response = self._client.embeddings.create(**kwargs)
        vectors = [item.embedding for item in response.data]
        if vectors and self._dimension is None:
            self._dimension = len(vectors[0])
        return vectors

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


# --------------------------------------------------------------------------
# Cohere
# --------------------------------------------------------------------------
class CohereEmbedder(DenseEmbedder):
    provider = "cohere"

    def __init__(self, model: str = "embed-english-v3.0", dimension: int | None = None):
        try:
            import cohere
        except ImportError as exc:  # pragma: no cover
            raise ConfigurationError(
                "The cohere package is not installed. Install with: pip install 'vector-toolbox-mcp[cohere]'"
            ) from exc
        api_key = get_settings().cohere_api_key
        if not api_key:
            raise ConfigurationError("COHERE_API_KEY is not set.")
        self.model = model
        self._dimension = dimension
        self._client = cohere.ClientV2(api_key=api_key)

    def _embed(self, texts: Sequence[str], input_type: str) -> list[list[float]]:
        response = self._client.embed(
            texts=list(texts),
            model=self.model,
            input_type=input_type,
            embedding_types=["float"],
        )
        vectors = [list(v) for v in response.embeddings.float_]
        if vectors and self._dimension is None:
            self._dimension = len(vectors[0])
        return vectors

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, "search_document")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "search_query")[0]


# --------------------------------------------------------------------------
# Local sentence-transformers
# --------------------------------------------------------------------------
class HuggingFaceEmbedder(DenseEmbedder):
    provider = "huggingface"

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2", dimension=None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ConfigurationError(
                "sentence-transformers is not installed. Install with: "
                "pip install 'vector-toolbox-mcp[local]'"
            ) from exc
        self.model = model
        self._model = SentenceTransformer(model)
        self._dimension = dimension or self._model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._model.encode(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        return list(map(float, self._model.encode([text])[0]))
