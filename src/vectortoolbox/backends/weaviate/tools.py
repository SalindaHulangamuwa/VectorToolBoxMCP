"""MCP tool definitions for the Weaviate backend.

Naming: ``weaviate_<verb>[_<noun>]``. Every tool takes ``client`` (default
``"default"``, built from VTB_WEAVIATE_* settings). Tools covering the same
ground as Weaviate's own MCP server:

* ``weaviate-collections-get-config`` -> ``weaviate_get_collection_config``
* ``weaviate-tenants-list``           -> ``weaviate_list_tenants``
* ``weaviate-query-hybrid``           -> ``weaviate_hybrid_search``
* ``weaviate-objects-upsert``         -> ``weaviate_upsert_objects``

plus the rest of the lifecycle: clients, collection create/update/delete,
tenants, references, filtered fetch, semantic and keyword search, aggregation.
"""

from __future__ import annotations

import functools
from typing import Any, Literal

from ..._mcp_compat import MCPServerType
from ...config import get_settings
from ...errors import ReadOnlyError, ToolboxError
from ...tools_common import register_status_provider
from . import client as client_mod
from .backend import METADATA_FIELDS, WeaviateBackend

ClientKindArg = Literal["local", "custom", "cloud", "embedded"]
EmbedProviderArg = Literal["pinecone", "openai", "cohere", "huggingface"]
MetadataArg = Literal["distance", "certainty", "score", "explain_score", "creation_time", "last_update_time", "is_consistent"]

FILTER_DOC = """
        **filters** - Weaviate's native ``where`` format (as in Weaviate's own
        MCP server)::

            {"operator": "And", "operands": [
              {"path": ["year"], "operator": "GreaterThanEqual", "valueInt": 2020},
              {"path": ["tags"], "operator": "ContainsAny", "valueTextArray": ["ai"]}]}

        or the Mongo-style form shared with the Chroma/Pinecone tools:
        ``{"year": {"$gte": 2020}, "tags": {"$in": ["ai"]}}``. Operators:
        Equal, NotEqual, GreaterThan(Equal), LessThan(Equal), Like (``*``/``?``
        wildcards), ContainsAny, ContainsAll, ContainsNone, IsNull,
        WithinGeoRange; And / Or / Not. Paths: a property, ``id``,
        ``_creationTimeUnix``, ``_lastUpdateTimeUnix``, ``len(prop)``,
        ``count(ref)``, or ``["ref", "Target", "prop"]`` through a reference.
        Filters apply before scoring and are checked against the property types.
"""

RETURN_DOC = """
        Response shape: ``return_properties`` (default all), ``return_references``
        (reference properties to expand), ``return_metadata`` (any of
        distance, certainty, score, explain_score, creation_time,
        last_update_time, is_consistent), ``include_vector``. ``group_by``
        ``{"property", "objects_per_group", "number_of_groups"}`` returns groups;
        ``rerank`` ``{"property", "query"}`` reorders with the collection's
        reranker module. ``auto_limit`` (autocut) stops at the n-th jump in scores.
        Multi-tenant collections need ``tenant``.
"""


def _with_doc(extra: str):
    def decorator(func):
        func.__doc__ = (func.__doc__ or "").rstrip() + "\n" + extra
        return func

    return decorator


def _backend() -> WeaviateBackend:
    from ...core.registry import get_backend

    return get_backend("weaviate")  # type: ignore[return-value]


def _safe(write: bool = False):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                if write and get_settings().read_only:
                    raise ReadOnlyError(
                        f"{func.__name__} is a write operation and the server is running in "
                        "read-only mode. Set VTB_READ_ONLY=false to enable write tools."
                    )
                return func(*args, **kwargs)
            except ToolboxError as exc:
                return {"error": type(exc).__name__, "message": str(exc)}
            except Exception as exc:  # SDK / network errors
                text = f"{type(exc).__name__}: {exc}"
                hint = ("This came from the Weaviate client or server, not argument validation. Check the "
                        "client (weaviate_heartbeat) and the collection config (weaviate_get_collection_config).")
                low = text.lower()
                if "connect" in low or "unavailable" in low or "refused" in low:
                    hint = ("Weaviate is not reachable. Check it is running and the client's host/ports "
                            "(HTTP 8080 and gRPC 50051 by default) - weaviate_list_clients shows them.")
                elif "api key" in low or "apikey" in low or "401" in low or "403" in low:
                    hint = ("Authentication failed - either the Weaviate API key, or a model provider key a "
                            "vectorizer module needs (sent as X-<Provider>-Api-Key headers from OPENAI_API_KEY etc.).")
                elif "vectorizer" in low or "module" in low:
                    hint = "A vectorizer/generative module failed or is not enabled. weaviate_list_modules shows what this server has."
                return {"error": type(exc).__name__, "message": str(exc)[:2000], "hint": hint}

        return wrapper

    return decorator


def status() -> dict[str, Any]:
    """Weaviate section of ``vectortoolbox_status``."""
    settings = get_settings()
    try:
        import weaviate

        version = weaviate.__version__
    except Exception:
        version = None
    return {
        "installed": version is not None,
        "client_version": version,
        "default_client_kind": settings.weaviate_client,
        "clients": client_mod.list_clients(),
        "cloud_api_key_set": bool(settings.weaviate_api_key),
        "provider_headers_available": client_mod.provider_headers()[1],
    }


def register(mcp: MCPServerType) -> None:
    register_status_provider("weaviate", status)

    # ======================================================================
    # Clients and server
    # ======================================================================
    @mcp.tool()
    @_safe()
    def weaviate_create_client(
        name: str,
        kind: ClientKindArg,
        host: str | None = None,
        port: int | None = None,
        secure: bool | None = None,
        grpc_host: str | None = None,
        grpc_port: int | None = None,
        grpc_secure: bool | None = None,
        url: str | None = None,
        api_key_env_var: str | None = None,
        header_env_vars: dict[str, str] | None = None,
        path: str | None = None,
        version: str | None = None,
        timeout_seconds: int | None = None,
        replace: bool = False,
    ) -> dict[str, Any]:
        """Connect a named Weaviate client that other weaviate_* tools can target.

        * ``local`` - server on this machine: ``host`` (localhost), ``port``
          (8080), ``grpc_port`` (50051). Docker:
          ``docker run -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate``.
        * ``custom`` - any self-hosted server: HTTP ``host``/``port``/``secure``
          and gRPC ``grpc_host``/``grpc_port``/``grpc_secure`` separately.
        * ``cloud`` - Weaviate Cloud: ``url`` (or WEAVIATE_URL) and the API key
          from WEAVIATE_API_KEY or the variable named by ``api_key_env_var``.
        * ``embedded`` - the client downloads and runs Weaviate itself, data in
          ``path`` (default ~/.vector-toolbox/weaviate); ``version`` pins it.
          Linux and macOS. Stops when this server stops.

        Model-provider keys found in the environment (OPENAI_API_KEY,
        COHERE_API_KEY, VOYAGEAI_API_KEY, ...) are forwarded as the headers
        Weaviate's vectorizer modules read. ``header_env_vars`` adds others:
        ``{"X-Custom-Api-Key": "MY_ENV_VAR"}``. Raw keys are never accepted as
        arguments. A client named ``default`` is built from VTB_WEAVIATE_*
        settings on first use; ``replace=true`` reconnects a name.
        """
        entry = client_mod.register_client(
            name, kind, replace=replace, host=host, port=port, secure=secure, grpc_host=grpc_host,
            grpc_port=grpc_port, grpc_secure=grpc_secure, url=url, api_key_env_var=api_key_env_var,
            header_env_vars=header_env_vars, path=path, version=version, timeout_seconds=timeout_seconds,
        )
        return {"created": entry.describe(), "ready": entry.client.is_ready()}

    @mcp.tool()
    @_safe()
    def weaviate_list_clients() -> dict[str, Any]:
        """List Weaviate clients connected in this session (the default appears once used)."""
        s = get_settings()
        return {"clients": client_mod.list_clients(),
                "default_settings": {"kind": s.weaviate_client,
                                     "http": f"{s.weaviate_http_host}:{s.weaviate_http_port}",
                                     "grpc_port": s.weaviate_grpc_port,
                                     "cloud_url": s.weaviate_url, "api_key_set": bool(s.weaviate_api_key)}}

    @mcp.tool()
    @_safe()
    def weaviate_remove_client(name: str) -> dict[str, Any]:
        """Close and forget a named client. Server data is untouched (an embedded server stops)."""
        return client_mod.remove_client(name)

    @mcp.tool()
    @_safe()
    def weaviate_heartbeat(client: str = "default") -> dict[str, Any]:
        """Check a client: ready/live, server version, cluster nodes, module count."""
        return _backend().heartbeat(client)

    @mcp.tool()
    @_safe()
    def weaviate_list_modules(client: str = "default") -> dict[str, Any]:
        """Modules enabled on this server, grouped (text2vec, multi2vec, generative, reranker, ...).

        A collection can only use a vectorizer, generative or reranker module
        the server has enabled. Without one, use vectorizer ``none`` and let
        this server embed with ``embed_provider``.
        """
        return _backend().list_modules(client)

    # ======================================================================
    # Collections
    # ======================================================================
    @mcp.tool()
    @_safe()
    def weaviate_list_collections(client: str = "default", include_counts: bool = False) -> dict[str, Any]:
        """List collections with their vectors (name -> vectorizer), property count and
        multi-tenancy. ``include_counts`` adds object counts (one aggregate each)."""
        return _backend().list_collections(client, include_counts=include_counts)

    @mcp.tool()
    @_safe()
    def weaviate_get_collection_config(
        collection_name: str | None = None, client: str = "default", raw: bool = False
    ) -> dict[str, Any]:
        """Schema and configuration of one collection, or of all when ``collection_name`` is omitted.

        Compact view: properties (data type, tokenization, filterable /
        searchable / range-filter indexes), references, named vectors
        (vectorizer, model, source properties, index type, distance,
        quantizer), multi-tenancy, replication, BM25, stopwords, generative and
        reranker modules. ``raw=true`` returns Weaviate's full class JSON.
        """
        return _backend().get_config(client, collection_name, raw=raw)

    @mcp.tool()
    @_safe()
    def weaviate_collection_capabilities(collection: str, client: str = "default") -> dict[str, Any]:
        """Which searches a collection supports and how: vectors and their vectorizers,
        BM25-searchable properties, filterable and range-indexed properties, tenancy."""
        return _backend().capabilities(collection, client=client).model_dump()

    @mcp.tool()
    @_safe(write=True)
    def weaviate_create_collection(
        name: str,
        client: str = "default",
        description: str | None = None,
        properties: list[dict[str, Any]] | None = None,
        references: list[dict[str, Any]] | None = None,
        vectors: list[dict[str, Any]] | None = None,
        multi_tenancy: dict[str, Any] | bool | None = None,
        replication_factor: int | None = None,
        bm25: dict[str, float] | None = None,
        stopwords_preset: Literal["en", "none"] | None = None,
        index_timestamps: bool | None = None,
        index_null_state: bool | None = None,
        index_property_length: bool | None = None,
        generative: dict[str, Any] | None = None,
        reranker: dict[str, Any] | None = None,
        raw_config: dict[str, Any] | None = None,
        if_not_exists: bool = False,
    ) -> dict[str, Any]:
        """Create a collection: typed properties, named vectors, indexes, tenancy.

        * ``name`` - starts with a capital letter (``Article``).
        * ``properties`` - ``[{"name": "title", "data_type": "text",
          "tokenization": "word", "filterable": true, "searchable": true},
          {"name": "year", "data_type": "int", "range_filters": true}]``.
          Types: text, int, number, boolean, date, uuid, geoCoordinates,
          phoneNumber, blob, object (with ``nested_properties``), and arrays
          (``text[]``, ``int[]``, ...). ``searchable`` = in BM25 (text only);
          ``range_filters`` speeds up >/< on int/number/date.
          ``skip_vectorization`` keeps a property out of the vectorizer input.
        * ``references`` - ``[{"name": "author", "target": "Person"}]``.
        * ``vectors`` - one entry per named vector (default: one self-provided
          vector called ``default``)::

              [{"name": "content", "vectorizer": "text2vec-openai",
                "model": "text-embedding-3-small", "source_properties": ["title", "body"],
                "index_type": "hnsw", "distance": "cosine", "quantizer": "rq"},
               {"name": "custom", "vectorizer": "none", "index_type": "flat"}]

          ``vectorizer`` is a module enabled on the server (weaviate_list_modules)
          or ``none`` to send your own vectors (``embed_provider`` on the write
          and search tools embeds them here). Index types: hnsw (default),
          flat (small or per-tenant data), dynamic (flat -> hnsw as it grows).
          Distances: cosine, dot, l2-squared, hamming, manhattan. Quantizers:
          pq, bq, sq, rq (compress vectors; flat supports bq/rq). Extra
          settings go in ``vectorizer_config`` / ``index_config`` / ``quantizer_config``.
        * ``multi_tenancy`` - ``true`` or ``{"auto_tenant_creation": true,
          "auto_tenant_activation": true}``. Only settable at creation.
        * ``bm25`` - ``{"b": 0.75, "k1": 1.2}``; ``stopwords_preset``.
        * ``index_timestamps`` / ``index_null_state`` / ``index_property_length``
          enable filtering by creation time, IsNull and ``len(prop)``.
        * ``generative`` / ``reranker`` - ``{"module": "generative-openai", "model": ...}``.

        Fixed after creation: property data types and tokenization, vectorizer
        and its source properties, index type, distance, multi-tenancy on/off.
        ``raw_config`` takes Weaviate's REST class JSON directly instead.
        """
        return _backend().create_collection(
            client, name, raw_config=raw_config, if_not_exists=if_not_exists,
            description=description, properties=properties, references=references, vectors=vectors,
            multi_tenancy=multi_tenancy, replication_factor=replication_factor, bm25=bm25,
            stopwords_preset=stopwords_preset, index_timestamps=index_timestamps,
            index_null_state=index_null_state, index_property_length=index_property_length,
            generative=generative, reranker=reranker,
        )

    @mcp.tool()
    @_safe(write=True)
    def weaviate_update_collection(
        name: str,
        client: str = "default",
        description: str | None = None,
        property_descriptions: dict[str, str] | None = None,
        bm25: dict[str, float] | None = None,
        stopwords_preset: Literal["en", "none"] | None = None,
        stopwords_additions: list[str] | None = None,
        stopwords_removals: list[str] | None = None,
        auto_tenant_creation: bool | None = None,
        auto_tenant_activation: bool | None = None,
        replication_factor: int | None = None,
        async_replication: bool | None = None,
        vector_index: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Change the mutable settings of a collection; returns what changed.

        * descriptions, BM25 ``{"b", "k1"}``, stopwords.
        * multi-tenant collections: ``auto_tenant_creation``, ``auto_tenant_activation``.
        * ``replication_factor``, ``async_replication``.
        * ``vector_index`` per named vector, e.g. ``{"default": {"ef": 128,
          "filter_strategy": "acorn", "quantizer": "rq"}}``. HNSW: ef,
          dynamic_ef_min/max/factor, flat_search_cutoff, filter_strategy
          (sweeping | acorn), vector_cache_max_objects. Flat:
          vector_cache_max_objects. Dynamic: threshold. Turning on a
          ``quantizer`` (pq/bq/sq/rq, or ``{"type": "pq", "segments": 96}``)
          compresses existing vectors in the background.

        Not changeable: property types, vectorizers, index type, distance,
        turning multi-tenancy on. Add properties/vectors with weaviate_add_to_collection.
        """
        return _backend().update_collection(
            client, name, description=description, property_descriptions=property_descriptions, bm25=bm25,
            stopwords_preset=stopwords_preset, stopwords_additions=stopwords_additions,
            stopwords_removals=stopwords_removals, auto_tenant_creation=auto_tenant_creation,
            auto_tenant_activation=auto_tenant_activation, replication_factor=replication_factor,
            async_replication=async_replication, vector_index=vector_index,
        )

    @mcp.tool()
    @_safe(write=True)
    def weaviate_add_to_collection(
        name: str,
        client: str = "default",
        property: dict[str, Any] | None = None,
        reference: dict[str, Any] | None = None,
        vector: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add one property, cross-reference or named vector to an existing collection.

        * ``property`` - same shape as in weaviate_create_collection.
        * ``reference`` - ``{"name": "author", "target": "Person"}``.
        * ``vector`` - ``{"name": "summary_vec"}`` (a self-provided vector).

        Existing objects get no value for it; nothing is re-vectorized.
        """
        return _backend().add_to_collection(client, name, property=property, reference=reference, vector=vector)

    @mcp.tool()
    @_safe(write=True)
    def weaviate_delete_collection(name: str, client: str = "default", confirm: bool = False) -> dict[str, Any]:
        """Delete a collection, all its objects and tenants. Requires ``confirm=true``; without it the
        call reports what would be lost."""
        return _backend().delete_collection(client, name, confirm=confirm)

    # ======================================================================
    # Tenants
    # ======================================================================
    @mcp.tool()
    @_safe()
    def weaviate_list_tenants(collection_name: str, client: str = "default") -> dict[str, Any]:
        """Tenants of a multi-tenant collection with their activity status (ACTIVE / INACTIVE /
        OFFLOADED, or transitional OFFLOADING / ONLOADING) and a count per status."""
        return _backend().list_tenants(client, collection_name)

    @mcp.tool()
    @_safe(write=True)
    def weaviate_create_tenants(
        collection_name: str, tenants: list[Any], client: str = "default"
    ) -> dict[str, Any]:
        """Create tenants: ``["acme", "globex"]`` or ``[{"name": "acme", "activity_status": "INACTIVE"}]``.
        Existing names are skipped and reported."""
        return _backend().create_tenants(client, collection_name, tenants)

    @mcp.tool()
    @_safe(write=True)
    def weaviate_update_tenants(
        collection_name: str, tenants: list[dict[str, Any]], client: str = "default"
    ) -> dict[str, Any]:
        """Change tenant activity: ``[{"name": "acme", "activity_status": "INACTIVE"}]``.

        ACTIVE = loaded and queryable; INACTIVE = on disk, not in memory (cheap,
        not queryable); OFFLOADED = moved to cloud storage (needs an offload
        module). Data is kept in every state.
        """
        return _backend().update_tenants(client, collection_name, tenants)

    @mcp.tool()
    @_safe(write=True)
    def weaviate_delete_tenants(
        collection_name: str, tenants: list[str], client: str = "default", confirm: bool = False
    ) -> dict[str, Any]:
        """Delete tenants and all their objects. Requires ``confirm=true``."""
        return _backend().delete_tenants(client, collection_name, tenants, confirm=confirm)

    # ======================================================================
    # Objects
    # ======================================================================
    write_doc = """
        Each object: ``{"uuid"?: ..., "properties": {...}, "vector"?: [...] or
        {"vector_name": [...]}, "references"?: {"author": "<uuid>" or [uuids]}}``.
        Without a uuid Weaviate assigns one; ``id_from="sku"`` derives a stable
        UUID from that property so re-running the same data never duplicates it.

        Vectors: collections with a vectorizer module embed for you. For
        self-provided vectors (vectorizer ``none``) pass them, or set
        ``embed_source`` to the text property to embed here (with
        ``embed_provider``/``embed_model``, default VTB_EMBED_*; ``embed_vectors``
        picks which named vectors to fill). Objects left without a vector are
        reported - vector search will never find them.

        Property names are checked against the schema: Weaviate's auto-schema
        would otherwise add a misspelt property silently. ``allow_new_properties``
        lets new ones through. Per-object failures are returned, not raised.
        Multi-tenant collections need ``tenant``.
"""

    @mcp.tool()
    @_with_doc(write_doc)
    @_safe(write=True)
    def weaviate_insert_objects(
        collection: str,
        objects: list[dict[str, Any]],
        client: str = "default",
        tenant: str | None = None,
        id_from: str | None = None,
        embed_source: str | None = None,
        embed_vectors: list[str] | None = None,
        embed_provider: EmbedProviderArg | None = None,
        embed_model: str | None = None,
        embed_dimension: int | None = None,
        allow_new_properties: bool = False,
    ) -> dict[str, Any]:
        """Insert new objects (batched). Objects whose uuid already exists are skipped and
        listed under ``skipped_existing`` - use weaviate_upsert_objects to overwrite."""
        return _backend().insert_objects(
            client, collection, objects, tenant=tenant, upsert=False, id_from=id_from,
            embed_source=embed_source, embed_vectors=embed_vectors, embed_provider=embed_provider,
            embed_model=embed_model, embed_dimension=embed_dimension, allow_new_properties=allow_new_properties,
        )

    @mcp.tool()
    @_with_doc(write_doc)
    @_safe(write=True)
    def weaviate_upsert_objects(
        collection: str,
        objects: list[dict[str, Any]],
        client: str = "default",
        tenant: str | None = None,
        merge: bool = False,
        id_from: str | None = None,
        embed_source: str | None = None,
        embed_vectors: list[str] | None = None,
        embed_provider: EmbedProviderArg | None = None,
        embed_model: str | None = None,
        embed_dimension: int | None = None,
        allow_new_properties: bool = False,
    ) -> dict[str, Any]:
        """Insert objects, or overwrite those whose uuid exists. Reports inserted vs replaced.

        ``merge=false`` (default) replaces an existing object entirely;
        ``merge=true`` changes only the properties given and keeps the rest.
        """
        return _backend().insert_objects(
            client, collection, objects, tenant=tenant, upsert=True, merge=merge, id_from=id_from,
            embed_source=embed_source, embed_vectors=embed_vectors, embed_provider=embed_provider,
            embed_model=embed_model, embed_dimension=embed_dimension, allow_new_properties=allow_new_properties,
        )

    @mcp.tool()
    @_safe(write=True)
    def weaviate_update_object(
        collection: str,
        uuid: str,
        client: str = "default",
        properties: dict[str, Any] | None = None,
        vector: list[float] | dict[str, list[float]] | None = None,
        references: dict[str, Any] | None = None,
        replace: bool = False,
        tenant: str | None = None,
        allow_new_properties: bool = False,
        embed_source: str | None = None,
        embed_provider: EmbedProviderArg | None = None,
        embed_model: str | None = None,
    ) -> dict[str, Any]:
        """Update one object by uuid. Default merges the given properties into the object;
        ``replace=true`` overwrites it. A vectorizer module re-embeds changed text itself;
        for self-provided vectors pass ``vector`` or ``embed_source``."""
        return _backend().update_object(
            client, collection, uuid, properties=properties, vector=vector, references=references, replace=replace,
            tenant=tenant, allow_new_properties=allow_new_properties, embed_source=embed_source,
            embed_provider=embed_provider, embed_model=embed_model,
        )

    @mcp.tool()
    @_with_doc(FILTER_DOC)
    @_safe(write=True)
    def weaviate_delete_objects(
        collection: str,
        client: str = "default",
        ids: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        delete_all: bool = False,
        dry_run: bool = False,
        confirm: bool = False,
        tenant: str | None = None,
    ) -> dict[str, Any]:
        """Delete objects by uuid, by filter, or all of them.

        ``ids`` and ``filters`` combine with AND. ``dry_run=true`` returns the
        number that would be deleted. ``delete_all=true`` empties the
        collection (it is kept) and requires ``confirm=true``. Large deletes run
        in pages of 10,000.
        """
        return _backend().delete_objects(client, collection, ids=ids, filters=filters, delete_all=delete_all,
                                         dry_run=dry_run, confirm=confirm, tenant=tenant)

    @mcp.tool()
    @_safe(write=True)
    def weaviate_add_references(
        collection: str, references: list[dict[str, Any]], client: str = "default", tenant: str | None = None
    ) -> dict[str, Any]:
        """Link objects through a reference property:
        ``[{"from_uuid": "...", "from_property": "author", "to": "<uuid>" or ["<uuid>", ...]}]``."""
        return _backend().add_references(client, collection, references, tenant=tenant)

    # ======================================================================
    # Reading and search
    # ======================================================================
    @mcp.tool()
    @_with_doc(FILTER_DOC)
    @_safe()
    def weaviate_fetch_objects(
        collection: str,
        client: str = "default",
        ids: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        limit: int = 25,
        offset: int | None = None,
        after: str | None = None,
        sort: list[dict[str, Any]] | None = None,
        return_properties: list[str] | None = None,
        return_references: list[str] | None = None,
        return_metadata: list[MetadataArg] | None = None,
        include_vector: bool = False,
        tenant: str | None = None,
    ) -> dict[str, Any]:
        """Fetch objects by uuid and/or filter - no ranking.

        Paging: ``limit`` + ``offset``, or ``after=<last uuid>`` (cursor,
        whole collection only, returned as ``next_after``) to walk everything.
        ``sort``: ``[{"property": "year", "ascending": false}]`` (also
        ``_creationTimeUnix`` / ``_lastUpdateTimeUnix``).
        """
        return _backend().fetch_objects(
            client, collection, ids=ids, filters=filters, limit=limit, offset=offset, after=after, sort=sort,
            return_properties=return_properties, return_references=return_references,
            return_metadata=return_metadata, include_vector=include_vector, tenant=tenant,
        )

    @mcp.tool()
    @_with_doc(FILTER_DOC + RETURN_DOC)
    @_safe()
    def weaviate_hybrid_search(
        collection: str,
        query: str,
        client: str = "default",
        alpha: float = 0.75,
        vector: list[float] | dict[str, list[float]] | None = None,
        query_properties: list[str] | None = None,
        fusion_type: Literal["relative_score", "ranked"] = "relative_score",
        max_vector_distance: float | None = None,
        bm25_operator: Literal["and", "or"] | None = None,
        minimum_match: int | None = None,
        target_vector: str | list[str] | None = None,
        target_vector_combination: Literal["average", "minimum", "sum", "relative_score", "manual_weights"] | None = None,
        target_vector_weights: dict[str, float] | None = None,
        filters: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int | None = None,
        auto_limit: int | None = None,
        group_by: dict[str, Any] | None = None,
        rerank: dict[str, Any] | None = None,
        return_properties: list[str] | None = None,
        return_references: list[str] | None = None,
        return_metadata: list[MetadataArg] | None = None,
        include_vector: bool = False,
        tenant: str | None = None,
        embed_provider: EmbedProviderArg | None = None,
        embed_model: str | None = None,
        embed_dimension: int | None = None,
    ) -> dict[str, Any]:
        """Hybrid search - vector similarity and BM25 keyword scores fused in one server-side query.

        * ``alpha`` - 0.0 pure keyword, 1.0 pure vector, default 0.75.
        * ``fusion_type`` - ``relative_score`` (normalises scores, default) or
          ``ranked`` (reciprocal rank).
        * ``query_properties`` - limit BM25 to these properties; boost with
          ``"title^2"``. ``bm25_operator`` ``and`` = every term must match;
          ``or`` with ``minimum_match``.
        * ``max_vector_distance`` - drop vector candidates farther than this.
        * ``target_vector`` - which named vector(s); several are combined by
          ``target_vector_combination`` (or ``target_vector_weights``).
        * The vector side comes from the collection's vectorizer; for a
          self-provided vector the query is embedded here with ``embed_provider``
          (default VTB_EMBED_*), or pass ``vector``.
        """
        return _backend().hybrid(
            client, collection, query, alpha=alpha, vector=vector, query_properties=query_properties,
            fusion_type=fusion_type, max_vector_distance=max_vector_distance, bm25_operator=bm25_operator,
            minimum_match=minimum_match, target_vector=target_vector,
            target_vector_combination=target_vector_combination, target_vector_weights=target_vector_weights,
            filters=filters, limit=limit, offset=offset, auto_limit=auto_limit, group_by=group_by, rerank=rerank,
            return_properties=return_properties, return_references=return_references,
            return_metadata=return_metadata, include_vector=include_vector, tenant=tenant,
            embed_provider=embed_provider, embed_model=embed_model, embed_dimension=embed_dimension,
        )

    @mcp.tool()
    @_with_doc(FILTER_DOC + RETURN_DOC)
    @_safe()
    def weaviate_semantic_search(
        collection: str,
        client: str = "default",
        query: str | None = None,
        vector: list[float] | dict[str, list[float]] | None = None,
        near_object: str | None = None,
        target_vector: str | list[str] | None = None,
        target_vector_combination: Literal["average", "minimum", "sum", "relative_score", "manual_weights"] | None = None,
        target_vector_weights: dict[str, float] | None = None,
        distance: float | None = None,
        certainty: float | None = None,
        filters: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int | None = None,
        auto_limit: int | None = None,
        group_by: dict[str, Any] | None = None,
        rerank: dict[str, Any] | None = None,
        return_properties: list[str] | None = None,
        return_references: list[str] | None = None,
        return_metadata: list[MetadataArg] | None = None,
        include_vector: bool = False,
        tenant: str | None = None,
        embed_provider: EmbedProviderArg | None = None,
        embed_model: str | None = None,
        embed_dimension: int | None = None,
    ) -> dict[str, Any]:
        """Vector similarity search - one of: ``query`` (text), ``vector``, or ``near_object`` (a uuid:
        "more like this").

        Text is embedded by the target vector's vectorizer module (near_text),
        or - for self-provided vectors - here with ``embed_provider`` and sent
        as a vector. ``distance`` / ``certainty`` cut off weak matches (lower
        distance is closer). ``target_vector`` picks a named vector.
        """
        return _backend().semantic(
            client, collection, query=query, vector=vector, near_object=near_object, target_vector=target_vector,
            target_vector_combination=target_vector_combination, target_vector_weights=target_vector_weights,
            distance=distance, certainty=certainty, filters=filters, limit=limit, offset=offset,
            auto_limit=auto_limit, group_by=group_by, rerank=rerank, return_properties=return_properties,
            return_references=return_references, return_metadata=return_metadata,
            include_vector=include_vector, tenant=tenant, embed_provider=embed_provider,
            embed_model=embed_model, embed_dimension=embed_dimension,
        )

    @mcp.tool()
    @_with_doc(FILTER_DOC + RETURN_DOC)
    @_safe()
    def weaviate_keyword_search(
        collection: str,
        query: str,
        client: str = "default",
        query_properties: list[str] | None = None,
        operator: Literal["and", "or"] | None = None,
        minimum_match: int | None = None,
        filters: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int | None = None,
        auto_limit: int | None = None,
        group_by: dict[str, Any] | None = None,
        rerank: dict[str, Any] | None = None,
        return_properties: list[str] | None = None,
        return_references: list[str] | None = None,
        return_metadata: list[MetadataArg] | None = None,
        include_vector: bool = False,
        tenant: str | None = None,
    ) -> dict[str, Any]:
        """BM25 keyword search over the collection's searchable text properties.

        ``query_properties`` limits and boosts fields (``["title^3", "body"]``).
        ``operator="and"`` requires every term; ``"or"`` with ``minimum_match``
        requires at least that many. Tokenization of each property (word,
        lowercase, field, trigram, ...) decides what counts as a term.
        """
        return _backend().keyword(
            client, collection, query, query_properties=query_properties, operator=operator,
            minimum_match=minimum_match, filters=filters, limit=limit, offset=offset, auto_limit=auto_limit,
            group_by=group_by, rerank=rerank, return_properties=return_properties,
            return_references=return_references, return_metadata=return_metadata,
            include_vector=include_vector, tenant=tenant,
        )

    @mcp.tool()
    @_with_doc(FILTER_DOC)
    @_safe()
    def weaviate_aggregate(
        collection: str,
        client: str = "default",
        metrics: list[dict[str, Any]] | None = None,
        filters: dict[str, Any] | None = None,
        group_by: dict[str, Any] | None = None,
        tenant: str | None = None,
    ) -> dict[str, Any]:
        """Counts and statistics without returning objects.

        Always returns the total count (after ``filters``). ``metrics``:
        ``[{"property": "price", "metrics": ["mean", "minimum", "maximum"]},
        {"property": "category", "top_occurrences": 5}]`` - numbers: count,
        minimum, maximum, mean, median, mode, sum; text: top occurrences;
        boolean: total_true, percentage_true, ...; date: minimum, maximum,
        median, mode. ``group_by`` ``{"property": "category", "limit": 10}``
        returns the count (and metrics) per value.
        """
        return _backend().aggregate(client, collection, metrics=metrics, filters=filters,
                                    group_by=group_by, tenant=tenant)

    @mcp.tool()
    @_with_doc(FILTER_DOC)
    @_safe()
    def weaviate_validate_filter(
        filters: dict[str, Any], collection: str | None = None, client: str = "default"
    ) -> dict[str, Any]:
        """Check a filter without running it and return its normalised native form. With
        ``collection``, property names and value types are checked against the schema."""
        return _backend().validate_filter(client, filters, collection)


__all__ = ["register", "status", "METADATA_FIELDS"]
