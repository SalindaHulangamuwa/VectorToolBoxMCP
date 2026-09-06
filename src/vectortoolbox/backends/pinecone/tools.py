"""MCP tool definitions for the Pinecone backend.

Naming convention: ``pinecone_<verb>_<noun>``. When a second backend lands its
tools are prefixed with its own name, and the generic cross-backend tools live
in ``vectortoolbox.tools_common``.
"""

from __future__ import annotations

import functools
from typing import Any, Literal

from ..._mcp_compat import MCPServerType

from ...config import get_settings
from ...core.types import DenseFieldSpec, SparseFieldSpec, TextFieldSpec
from ...errors import ReadOnlyError, ToolboxError
from . import ttl as ttl_mod
from .backend import PineconeBackend

WRITE_TOOLS_NOTE = "Set VTB_READ_ONLY=false to enable write tools."


def _backend() -> PineconeBackend:
    from ...core.registry import get_backend

    return get_backend("pinecone")  # type: ignore[return-value]


def _safe(write: bool = False):
    """Wrap a tool so failures come back as readable JSON, not tracebacks."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                if write and get_settings().read_only:
                    raise ReadOnlyError(
                        f"{func.__name__} is a write operation and the server is running in "
                        f"read-only mode. {WRITE_TOOLS_NOTE}"
                    )
                return func(*args, **kwargs)
            except ToolboxError as exc:
                return {"error": type(exc).__name__, "message": str(exc)}
            except Exception as exc:  # pragma: no cover - network/SDK errors
                return {
                    "error": type(exc).__name__,
                    "message": str(exc)[:2000],
                    "hint": "This came from the Pinecone SDK or the network, not argument "
                    "validation. Check the index name, API key and region.",
                }

        return wrapper

    return decorator


def register(mcp: MCPServerType) -> None:
    # ======================================================================
    # Index lifecycle
    # ======================================================================
    @mcp.tool()
    @_safe(write=True)
    def pinecone_create_index(
        name: str,
        dense_fields: list[DenseFieldSpec] | None = None,
        sparse_fields: list[SparseFieldSpec] | None = None,
        text_fields: list[TextFieldSpec] | None = None,
        cloud: str | None = None,
        region: str | None = None,
        pod: dict[str, Any] | None = None,
        read_capacity: dict[str, Any] | None = None,
        deletion_protection: Literal["enabled", "disabled"] | None = None,
        tags: dict[str, str] | None = None,
        cmek_id: str | None = None,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Create a Pinecone index by declaring its searchable fields.

        The schema decides which searches the index can ever answer, and it
        cannot be changed afterwards - so decide up front:

        * ``dense_fields``   -> semantic search. Dimension must match the
          embedding model you will use.
        * ``sparse_fields``  -> learned lexical search (SPLADE-style).
        * ``text_fields``    -> BM25 full-text search and Lucene query strings.

        Recipes matching the five common setups:

        1. Keyword search only, one field:
           ``text_fields=[{"name": "body"}]``
        2. Multi-field FTS:
           ``text_fields=[{"name": "body"}, {"name": "summary"}]``
        3. Dense + FTS in one index:
           ``dense_fields=[{"name": "embedding", "dimension": 1024}]`` plus
           ``text_fields=[{"name": "body"}]``
        4. Multi-signal (dense + sparse + FTS): all three lists populated.
        5. Sparse + dense hybrid over the Vectors API: exactly one dense and
           one sparse field, no text fields.

        Limits enforced before the call is sent: **at most one** dense_vector
        field and **at most one** sparse_vector field per index, up to 100
        full-text string fields. Field names must be unique, at most 64 bytes,
        and must not start with ``_`` (reserved for ``_id`` / ``_score``) or
        ``$`` (reserved for filter operators).

        Only searchable fields go in the schema. Ordinary metadata is indexed
        for filtering automatically the first time it appears on a record;
        declaring it here is rejected by the API.

        Args:
            name: 1-45 chars, lowercase alphanumerics and hyphens.
            pod: Pod deployment instead of managed serverless, e.g.
                ``{"environment": "us-east-1-aws", "pod_type": "p1.x1",
                "replicas": 1, "shards": 1}``.
            read_capacity: ``{"mode": "OnDemand"}`` or
                ``{"mode": "Dedicated", "dedicated": {...}}``.
            deletion_protection: "enabled" blocks deletion until switched back.
            tags: Up to 20 key/value pairs.
            cmek_id: Customer-managed encryption key id.
            timeout: Seconds to wait for readiness; -1 returns immediately.
        """
        return _backend().create_index(
            name=name,
            dense_fields=dense_fields,
            sparse_fields=sparse_fields,
            text_fields=text_fields,
            cloud=cloud,
            region=region,
            pod=pod,
            read_capacity=read_capacity,
            deletion_protection=deletion_protection,
            tags=tags,
            cmek_id=cmek_id,
            timeout=timeout,
        )

    @mcp.tool()
    @_safe(write=True)
    def pinecone_create_index_for_model(
        name: str,
        model: str = "llama-text-embed-v2",
        text_field: str = "chunk_text",
        cloud: str | None = None,
        region: str | None = None,
        metric: Literal["cosine", "euclidean", "dotproduct"] | None = None,
        dimension: int | None = None,
        read_parameters: dict[str, Any] | None = None,
        write_parameters: dict[str, Any] | None = None,
        filterable_fields: dict[str, Any] | None = None,
        deletion_protection: Literal["enabled", "disabled"] | None = None,
        tags: dict[str, str] | None = None,
        read_capacity: dict[str, Any] | None = None,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Create an index with a hosted embedding model attached.

        Pinecone embeds ``text_field`` on write and embeds queries on read, so
        no embedding provider is needed on this side. Read it back with
        ``pinecone_search_records``. The model cannot be changed later.

        Args:
            model: Hosted model, e.g. "llama-text-embed-v2",
                "multilingual-e5-large", "pinecone-sparse-english-v0".
            text_field: Record field holding the raw text to embed.
            filterable_fields: e.g. ``{"genre": {"filterable": true}}``.
        """
        return _backend().create_index_for_model(
            name=name,
            model=model,
            field_map={"text": text_field},
            cloud=cloud,
            region=region,
            metric=metric,
            dimension=dimension,
            read_parameters=read_parameters,
            write_parameters=write_parameters,
            filterable_fields=filterable_fields,
            deletion_protection=deletion_protection,
            tags=tags,
            read_capacity=read_capacity,
            timeout=timeout,
        )

    @mcp.tool()
    @_safe()
    def pinecone_list_indexes() -> dict[str, Any]:
        """List every index in the project, with the search modes each supports."""
        indexes = _backend().list_indexes()
        return {"count": len(indexes), "indexes": indexes}

    @mcp.tool()
    @_safe()
    def pinecone_describe_index(index: str) -> dict[str, Any]:
        """Full server-side description of one index: schema, deployment, status, host."""
        return _backend().describe_index(index)

    @mcp.tool()
    @_safe()
    def pinecone_index_capabilities(index: str, refresh: bool = False) -> dict[str, Any]:
        """Report which searches an index can answer, and with which fields.

        Call this before searching an index you did not create in this session.
        It names the dense / sparse / full-text fields, the dimension each
        dense field expects, and the list of valid ``mode`` values for
        ``pinecone_search``.
        """
        return _backend().capabilities(index, refresh=refresh).model_dump()

    @mcp.tool()
    @_safe(write=True)
    def pinecone_configure_index(
        index: str,
        deletion_protection: Literal["enabled", "disabled"] | None = None,
        tags: dict[str, str] | None = None,
        read_capacity: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Change an index's deletion protection, tags or read capacity.

        The field schema is immutable - a new signal (a sparse field, an FTS
        field) means creating a new index and reindexing.
        """
        return _backend().configure_index(
            index,
            deletion_protection=deletion_protection,
            tags=tags,
            read_capacity=read_capacity,
        )

    @mcp.tool()
    @_safe(write=True)
    def pinecone_delete_index(index: str, confirm: bool = False) -> dict[str, Any]:
        """Delete an index and everything in it. Requires ``confirm=true``."""
        return _backend().delete_index(index, confirm=confirm)

    # ======================================================================
    # Namespaces
    # ======================================================================
    @mcp.tool()
    @_safe()
    def pinecone_list_namespaces(index: str) -> dict[str, Any]:
        """List the namespaces in an index, with record counts where available."""
        namespaces = _backend().list_namespaces(index)
        return {"index": index, "count": len(namespaces), "namespaces": namespaces}

    @mcp.tool()
    @_safe()
    def pinecone_describe_namespace(index: str, namespace: str) -> dict[str, Any]:
        """Describe one namespace: record count and metadata."""
        return _backend().describe_namespace(index, namespace)

    @mcp.tool()
    @_safe(write=True)
    def pinecone_create_namespace(index: str, namespace: str) -> dict[str, Any]:
        """Create an empty namespace. Upserting to a new namespace also creates it."""
        return _backend().create_namespace(index, namespace)

    @mcp.tool()
    @_safe(write=True)
    def pinecone_delete_namespace(
        index: str, namespace: str, confirm: bool = False
    ) -> dict[str, Any]:
        """Delete a namespace and every record in it. Requires ``confirm=true``."""
        return _backend().delete_namespace(index, namespace, confirm=confirm)

    @mcp.tool()
    @_safe()
    def pinecone_sample_metadata(
        index: str, namespace: str, sample_size: int = 20
    ) -> dict[str, Any]:
        """Sample records from a namespace and describe the metadata shape.

        Returns each field observed, its types, how many of the sampled
        records carried it, and up to three example values - enough to write a
        correct filter without dumping the namespace into the conversation.
        """
        return _backend().sample_metadata(index, namespace, sample_size=sample_size)

    @mcp.tool()
    @_safe()
    def pinecone_describe_index_stats(index: str, namespace: str | None = None) -> dict[str, Any]:
        """Record counts, dimension and per-namespace breakdown for an index."""
        return _backend().stats(index, namespace=namespace)

    # ======================================================================
    # Writing records
    # ======================================================================
    @mcp.tool()
    @_safe(write=True)
    def pinecone_upsert_documents(
        index: str,
        namespace: str,
        documents: list[dict[str, Any]],
        embed_source_field: str | None = None,
        dense_field: str | None = None,
        sparse_field: str | None = None,
        embed_provider: Literal["pinecone", "openai", "cohere", "huggingface"] | None = None,
        embed_model: str | None = None,
        embed_dimension: int | None = None,
        sparse_provider: str | None = None,
        sparse_model: str | None = None,
        ttl_seconds: int | None = None,
        batch_size: int = 50,
    ) -> dict[str, Any]:
        """Upsert records, optionally embedding them on the way in.

        Each document needs a unique ``_id``. Fields declared in the index
        schema are searched; every other field is stored and indexed for
        filtering automatically.

        Embedding is plug-and-play. Pass ``embed_source_field`` naming the text
        field to embed and the toolbox fills every dense and sparse field the
        schema declares, using ``embed_provider``/``embed_model`` (defaults come
        from VTB_EMBED_PROVIDER / VTB_EMBED_MODEL). A document that already
        carries a vector for a field is left alone, so you can mix pre-computed
        and generated vectors in one call. The vector width is checked against
        the schema before anything is sent.

        Validated before sending, because Pinecone fails an entire upsert if
        any one document is invalid: each document needs a unique ``_id`` and
        at least one schema field (a metadata-only document is rejected), and
        no field name may start with ``_`` or ``$``. Requests are split at
        1000 documents.

        TTL is implemented by this server, not by Pinecone: ``ttl_seconds``
        stamps a ``vtb_expires_at`` epoch on each record, searches exclude
        lapsed records by default, and ``pinecone_purge_expired`` reclaims the
        storage. Records written without a TTL carry no extra field and are
        never hidden by that filter.

        Args:
            documents: e.g. ``[{"_id": "d1", "body": "...", "category": "tech",
                "year": 2026}]``.
            embed_source_field: Field whose text becomes the vector(s).
            dense_field / sparse_field: Target schema fields when the index
                declares more than one.
            ttl_seconds: Lifetime in seconds. Omit for no expiry.
            batch_size: Documents per request when batching.
        """
        return _backend().upsert_documents(
            index,
            namespace,
            documents,
            embed_source_field=embed_source_field,
            dense_field=dense_field,
            sparse_field=sparse_field,
            embed_provider=embed_provider,
            embed_model=embed_model,
            embed_dimension=embed_dimension,
            sparse_provider=sparse_provider,
            sparse_model=sparse_model,
            ttl_seconds=ttl_seconds,
            batch_size=batch_size,
        )

    @mcp.tool()
    @_safe(write=True)
    def pinecone_upsert_vectors(
        index: str,
        namespace: str,
        vectors: list[dict[str, Any]],
        ttl_seconds: int | None = None,
        batch_size: int = 100,
    ) -> dict[str, Any]:
        """Upsert through the legacy Vectors API - raw values plus metadata.

        Use this for single-vector indexes where you want a dense and a sparse
        vector on the same record, which is what makes the one-request hybrid
        query in ``pinecone_query_vectors`` possible.

        Args:
            vectors: ``[{"id": "v1", "values": [...],
                "sparse_values": {"indices": [...], "values": [...]},
                "metadata": {"category": "tech"}}]``.
            ttl_seconds: Stamps ``_expires_at`` into each record's metadata.
        """
        return _backend().upsert_vectors(
            index, namespace, vectors, ttl_seconds=ttl_seconds, batch_size=batch_size
        )

    @mcp.tool()
    @_safe(write=True)
    def pinecone_update_documents(
        index: str,
        namespace: str,
        documents: list[dict[str, Any]] | None = None,
        filter: dict[str, Any] | None = None,
        set_fields: dict[str, Any] | None = None,
        remove_fields: list[str] | None = None,
    ) -> dict[str, Any]:
        """Partially update documents - by id, or by filter across many at once.

        Args:
            documents: Per-record updates, each with ``_id`` and the fields to
                change.
            filter: Update every document matching this filter instead.
            set_fields: Fields to set on all matched documents.
            remove_fields: Field names to strip.
        """
        return _backend().update_documents(
            index,
            namespace,
            documents=documents,
            filter=filter,
            set_fields=set_fields,
            remove_fields=remove_fields,
        )

    @mcp.tool()
    @_safe(write=True)
    def pinecone_update_vector(
        index: str,
        namespace: str,
        id: str,
        values: list[float] | None = None,
        sparse_values: dict[str, Any] | None = None,
        set_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Update one record's vector values or metadata via the Vectors API."""
        return _backend().update_vector(
            index,
            namespace,
            id=id,
            values=values,
            sparse_values=sparse_values,
            set_metadata=set_metadata,
        )

    @mcp.tool()
    @_safe(write=True)
    def pinecone_delete_records(
        index: str,
        namespace: str,
        ids: list[str] | None = None,
        filter: dict[str, Any] | None = None,
        delete_all: bool = False,
        confirm: bool = False,
    ) -> dict[str, Any]:
        """Delete records by id, by metadata filter, or clear a namespace.

        ``delete_all=true`` requires ``confirm=true``.
        """
        return _backend().delete_documents(
            index,
            namespace,
            ids=ids,
            filter=filter,
            delete_all=delete_all,
            confirm=confirm,
        )

    @mcp.tool()
    @_safe(write=True)
    def pinecone_purge_expired(
        index: str, namespace: str, confirm: bool = False
    ) -> dict[str, Any]:
        """Permanently delete records whose TTL has lapsed. Requires ``confirm=true``.

        Searches already hide expired records; this is what actually frees the
        storage. Run it on a schedule if you rely on TTL.
        """
        return _backend().purge_expired(index, namespace, confirm=confirm)

    # ======================================================================
    # Reading records
    # ======================================================================
    @mcp.tool()
    @_safe()
    def pinecone_fetch_records(
        index: str,
        namespace: str,
        ids: list[str] | None = None,
        filter: dict[str, Any] | None = None,
        include_fields: list[str] | None = None,
        limit: int | None = 20,
    ) -> dict[str, Any]:
        """Fetch records by id or filter, without ranking them.

        ``include_fields=["*"]`` returns every field; omitting it returns all
        available fields for a document fetch.
        """
        docs = _backend().fetch_documents(
            index,
            namespace,
            ids=ids,
            filter=filter,
            include_fields=include_fields,
            limit=limit,
        )
        return {"index": index, "namespace": namespace, "count": len(docs), "records": docs}

    @mcp.tool()
    @_safe()
    def pinecone_list_record_ids(
        index: str, namespace: str, prefix: str | None = None, limit: int = 100
    ) -> dict[str, Any]:
        """List record ids in a namespace, optionally filtered by id prefix."""
        ids = _backend().list_ids(index, namespace, prefix=prefix, limit=limit)
        return {"index": index, "namespace": namespace, "count": len(ids), "ids": ids}

    # ======================================================================
    # Search
    # ======================================================================
    @mcp.tool()
    @_safe()
    def pinecone_search(
        index: str,
        namespace: str,
        query: str | None = None,
        mode: Literal["auto", "text", "query_string", "dense", "sparse", "hybrid"] = "auto",
        fields: list[str] | None = None,
        field_queries: dict[str, str] | None = None,
        top_k: int = 10,
        filter: dict[str, Any] | None = None,
        include_fields: list[str] | None = None,
        vector: list[float] | None = None,
        sparse_vector: dict[str, Any] | None = None,
        dense_field: str | None = None,
        sparse_field: str | None = None,
        embed_provider: Literal["pinecone", "openai", "cohere", "huggingface"] | None = None,
        embed_model: str | None = None,
        embed_dimension: int | None = None,
        sparse_provider: str | None = None,
        sparse_model: str | None = None,
        fusion: Literal["rrf", "weighted"] = "rrf",
        weights: dict[str, float] | None = None,
        candidate_multiplier: int = 3,
        exclude_expired: bool = True,
    ) -> dict[str, Any]:
        """Search an index, validating the request against its schema first.

        Modes, and what each needs from the schema:

        * ``text`` - BM25. A text clause names exactly one field, so scoring
          across several fields sends one clause per field in a single request
          and Pinecone combines them with **equal weight** (there is no
          per-clause weight). Pass ``fields`` to choose them, or omit it to use
          every FTS field. ``field_queries`` gives each field its own query
          text, e.g. ``{"body": "disappointing", "summary": "Disappointing"}``.
        * ``query_string`` - Lucene syntax, which targets fields inside the
          query itself: ``title:(quantum) OR body:(machine AND learning)``.
          Supports ``AND``/``OR``/``NOT``, ``+``/``-``, grouping, phrases
          ``"..."``, phrase slop ``"..."~2``, boosting ``term^2``, phrase
          prefix ``"mach lear"*`` and regex ``body:/mach.*/``. Fuzzy matching
          (``~1`` on a single term) is **not** supported.
        * ``dense`` - semantic search. ``query`` is embedded with the
          configured provider unless you pass ``vector`` yourself.
        * ``sparse`` - learned lexical search over a sparse vector field.
        * ``hybrid`` - every signal the index has, run separately and fused
          client-side. A dense or sparse clause must be the only clause in its
          request, so this is one request per signal merged with Reciprocal
          Rank Fusion (``fusion="weighted"`` plus
          ``weights={"dense": 2, "text": 1}`` to bias one signal).
        * ``auto`` - the richest mode the index supports.

        ``filter`` narrows candidates before ranking and is deterministic, not
        a scoring signal. Metadata operators: ``$eq $ne $gt $gte $lt $lte $in
        $nin $exists $and $or $not``. On FTS-enabled string fields you also get
        ``$match_phrase``, ``$match_all`` and ``$match_any`` (at most 128
        tokens each) - which is how you rank by vector while *requiring* an
        exact term.

        ``include_fields`` defaults to every stored field. Pass a narrower list
        to keep responses small, or ``[]`` for ids and scores only.
        ``top_k`` may be 1-10000.

        Records whose TTL has lapsed are excluded by default; records written
        without a TTL are never hidden.
        """
        return _backend().search(
            index,
            namespace,
            mode=mode,
            query=query,
            field_queries=field_queries,
            vector=vector,
            sparse_vector=sparse_vector,
            fields=fields,
            dense_field=dense_field,
            sparse_field=sparse_field,
            top_k=top_k,
            filter=filter,
            include_fields=include_fields,
            embed_provider=embed_provider,
            embed_model=embed_model,
            embed_dimension=embed_dimension,
            sparse_provider=sparse_provider,
            sparse_model=sparse_model,
            fusion=fusion,
            weights=weights,
            candidate_multiplier=candidate_multiplier,
            exclude_expired=exclude_expired,
        )

    @mcp.tool()
    @_safe()
    def pinecone_search_records(
        index: str,
        namespace: str,
        query: str,
        top_k: int = 10,
        filter: dict[str, Any] | None = None,
        fields: list[str] | None = None,
        rerank_model: str | None = None,
        rerank_fields: list[str] | None = None,
        rerank_top_n: int | None = None,
        match_terms: dict[str, Any] | None = None,
        exclude_expired: bool = True,
    ) -> dict[str, Any]:
        """Search an integrated-inference index - Pinecone embeds the query.

        Only for indexes created with ``pinecone_create_index_for_model``.
        Optional server-side reranking: pass ``rerank_model`` (e.g.
        "bge-reranker-v2-m3") and ``rerank_fields``.

        ``match_terms`` constrains sparse retrieval to records containing
        specific terms, e.g. ``{"strategy": "all", "terms": ["refund"]}``
        (sparse indexes on pinecone-sparse-english-v0 only).
        """
        rerank = None
        if rerank_model:
            rerank = {"model": rerank_model, "rank_fields": rerank_fields or ["text"]}
            if rerank_top_n:
                rerank["top_n"] = rerank_top_n
        return _backend().search_records(
            index,
            namespace,
            query=query,
            top_k=top_k,
            filter=filter,
            fields=fields,
            rerank=rerank,
            match_terms=match_terms,
            exclude_expired=exclude_expired,
        )

    @mcp.tool()
    @_safe()
    def pinecone_query_vectors(
        index: str,
        namespace: str,
        vector: list[float] | None = None,
        sparse_vector: dict[str, Any] | None = None,
        query: str | None = None,
        id: str | None = None,
        top_k: int = 10,
        filter: dict[str, Any] | None = None,
        include_metadata: bool = True,
        include_values: bool = False,
        embed_provider: str | None = None,
        embed_model: str | None = None,
        embed_dimension: int | None = None,
        sparse_provider: str | None = None,
        sparse_model: str | None = None,
        exclude_expired: bool = True,
    ) -> dict[str, Any]:
        """Vectors API query - the true single-request dense+sparse hybrid.

        On a single-vector index holding both a dense and a sparse vector per
        record, passing both here has Pinecone do the hybrid scoring server
        side, rather than the client-side fusion ``pinecone_search`` uses for
        schema indexes.

        Pass ``query`` instead of vectors to have them embedded here first, or
        ``id`` to search by an existing record.
        """
        backend = _backend()
        if query and vector is None:
            from ...embeddings import get_dense_embedder

            vector = get_dense_embedder(embed_provider, embed_model, embed_dimension).embed_query(
                query
            )
        if query and sparse_vector is None and sparse_model:
            from ...embeddings import get_sparse_embedder

            sparse_vector = get_sparse_embedder(sparse_provider, sparse_model).embed_query(query)
        return backend.query_vectors(
            index,
            namespace,
            vector=vector,
            sparse_vector=sparse_vector,
            id=id,
            top_k=top_k,
            filter=filter,
            include_metadata=include_metadata,
            include_values=include_values,
            exclude_expired=exclude_expired,
        )

    @mcp.tool()
    @_safe()
    def pinecone_rerank(
        query: str,
        documents: list[dict[str, Any]],
        model: str = "bge-reranker-v2-m3",
        rank_fields: list[str] | None = None,
        top_n: int | None = None,
        return_documents: bool = True,
    ) -> dict[str, Any]:
        """Rerank a candidate list with a hosted cross-encoder.

        Use it as a second stage: retrieve widely with ``pinecone_search``
        (top_k 50-100), then rerank down to the handful you actually want.

        Args:
            documents: ``[{"id": "d1", "text": "..."}]``.
            rank_fields: Which field the reranker reads, default ``["text"]``.
        """
        return _backend().rerank(
            query=query,
            documents=documents,
            model=model,
            top_n=top_n,
            rank_fields=rank_fields or ["text"],
            return_documents=return_documents,
        )

    # ======================================================================
    # Utilities
    # ======================================================================
    @mcp.tool()
    @_safe()
    def pinecone_embed(
        texts: list[str],
        provider: Literal["pinecone", "openai", "cohere", "huggingface"] | None = None,
        model: str | None = None,
        dimension: int | None = None,
        input_type: Literal["passage", "query"] = "passage",
        kind: Literal["dense", "sparse"] = "dense",
    ) -> dict[str, Any]:
        """Generate embeddings without storing them - useful for dimension checks."""
        if kind == "sparse":
            from ...embeddings import get_sparse_embedder

            embedder = get_sparse_embedder(provider, model)
            vectors = (
                embedder.embed_documents(texts)
                if input_type == "passage"
                else [embedder.embed_query(t) for t in texts]
            )
            return {"model": embedder.describe(), "count": len(vectors), "vectors": vectors}

        from ...embeddings import get_dense_embedder

        embedder = get_dense_embedder(provider, model, dimension)
        vectors = (
            embedder.embed_documents(texts)
            if input_type == "passage"
            else [embedder.embed_query(t) for t in texts]
        )
        return {
            "model": embedder.describe(),
            "count": len(vectors),
            "dimension": len(vectors[0]) if vectors else None,
            "vectors": vectors,
        }

    @mcp.tool()
    @_safe()
    def pinecone_list_models(
        model_type: Literal["embed", "rerank"] | None = None,
    ) -> dict[str, Any]:
        """List the hosted embedding and reranking models Pinecone offers."""
        return {"models": _backend().list_models(model_type=model_type)}

    @mcp.tool()
    @_safe()
    def vectortoolbox_status() -> dict[str, Any]:
        """Report configuration: backends, default embedding provider, read-only mode."""
        from ...core.registry import list_backends
        from ...embeddings import DENSE_PROVIDERS, SPARSE_PROVIDERS

        settings = get_settings()
        return {
            "backends": list_backends(),
            "default_backend": settings.default_backend,
            "pinecone_api_key_set": bool(settings.pinecone_api_key),
            "pinecone_region": f"{settings.pinecone_cloud}/{settings.pinecone_region}",
            "dense_embedding_providers": sorted(DENSE_PROVIDERS),
            "sparse_embedding_providers": sorted(SPARSE_PROVIDERS),
            "default_dense_embedder": f"{settings.embed_provider}/{settings.embed_model}",
            "default_sparse_embedder": f"{settings.sparse_provider}/{settings.sparse_model}",
            "read_only": settings.read_only,
            "ttl": {
                "implementation": "client-side",
                "field": ttl_mod.TTL_FIELD,
                "note": "Pinecone has no server-side record expiry; this server stamps and "
                "filters on an epoch field and purges on demand.",
            },
        }
