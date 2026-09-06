"""Backend-neutral request/response shapes.

These pydantic models are what MCP tools take as arguments, so their field
descriptions become the JSON schema an LLM reads before calling a tool. Keep
the descriptions concrete - they are documentation for the caller, not for us.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Metric = Literal["cosine", "euclidean", "dotproduct"]


# --------------------------------------------------------------------------
# Schema declaration
# --------------------------------------------------------------------------
class DenseFieldSpec(BaseModel):
    """A dense vector field - semantic search lives here."""

    name: str = Field(description="Field name, e.g. 'embedding'.")
    dimension: int = Field(description="Vector width. Must match your embedding model.")
    metric: Metric = Field(default="cosine", description="Similarity metric.")
    description: str | None = None


class SparseFieldSpec(BaseModel):
    """A sparse vector field - learned lexical search (SPLADE-style) lives here."""

    name: str = Field(description="Field name, e.g. 'keywords'.")
    description: str | None = None


class TextFieldSpec(BaseModel):
    """A string field, optionally with BM25 full-text search enabled."""

    name: str = Field(description="Field name, e.g. 'body'.")
    full_text_search: bool = Field(
        default=True,
        description="Enable BM25 tokenisation + Lucene query syntax on this field.",
    )
    language: str = Field(default="en", description="Analyzer language code, e.g. 'en'.")
    stemming: bool | None = Field(default=None, description="Reduce words to stems.")
    stop_words: bool | None = Field(
        default=None,
        description="Strip stop words. Not supported for every language.",
    )
    ngram: dict[str, Any] | None = Field(
        default=None,
        description="Optional n-gram config for substring matching, e.g. {'min': 3, 'max': 5}.",
    )
    filterable: bool = Field(
        default=False,
        description="Also make the raw string filterable with $eq/$in.",
    )


class ScalarFieldSpec(BaseModel):
    """A filterable non-searchable field declared up front.

    Pinecone indexes metadata automatically the first time it appears, so this
    is only needed when you want the field guaranteed filterable from day one.
    """

    name: str
    type: Literal["integer", "float", "boolean", "string", "string_list"]
    filterable: bool = True


class SemanticTextFieldSpec(BaseModel):
    """A text field the backend embeds for you (integrated inference)."""

    name: str = Field(description="Field holding the raw text, e.g. 'chunk_text'.")
    model: str = Field(description="Hosted embedding model, e.g. 'llama-text-embed-v2'.")
    metric: Metric = "cosine"
    dimension: int | None = None


# --------------------------------------------------------------------------
# Capability description
# --------------------------------------------------------------------------
class IndexCapabilities(BaseModel):
    """What an existing index can actually answer.

    Every search tool validates against this before hitting the network, so a
    mismatched request fails with an explanation instead of a 400.
    """

    index: str
    api: Literal["documents", "vectors", "integrated"] = Field(
        description=(
            "'documents' = schema index read via the Documents API. "
            "'vectors' = legacy dimension/metric index read via the Vectors API. "
            "'integrated' = index with a hosted embedding model attached."
        )
    )
    dense_fields: list[str] = []
    sparse_fields: list[str] = []
    fts_fields: list[str] = []
    semantic_text_fields: list[str] = []
    filterable_fields: list[str] = []
    dimensions: dict[str, int] = Field(
        default_factory=dict, description="Dense field name -> dimension."
    )
    metrics: dict[str, str] = Field(default_factory=dict, description="Dense field name -> metric.")
    supported_search_modes: list[str] = []
    host: str | None = None
    deletion_protection: str | None = None
    notes: list[str] = []


SearchMode = Literal[
    "text",  # BM25 over one or more FTS fields
    "query_string",  # Lucene syntax over FTS fields
    "dense",  # semantic search over a dense vector field
    "sparse",  # learned lexical search over a sparse vector field
    "hybrid",  # multiple signals fused client-side (RRF or weighted)
    "auto",  # pick the best mode the index supports
]


class Hit(BaseModel):
    id: str
    score: float
    fields: dict[str, Any] = {}
    signals: dict[str, float] = Field(
        default_factory=dict,
        description="Per-signal scores when the result came from a fused hybrid search.",
    )
