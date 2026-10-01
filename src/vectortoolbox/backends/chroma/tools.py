"""MCP tool definitions for the Chroma backend.

Naming convention: ``chroma_<verb>[_<noun>]``. Every tool takes ``client``
(default ``"default"``) naming a client from ``chroma_create_client``; the
default client is built from VTB_CHROMA_* settings on first use.
"""

from __future__ import annotations

import functools
from typing import Any, Literal

from ..._mcp_compat import MCPServerType
from ...config import get_settings
from ...errors import ReadOnlyError, ToolboxError
from ...tools_common import register_status_provider
from . import client as client_mod
from .backend import ChromaBackend
from .filters import normalize_where, normalize_where_document

ClientKindArg = Literal["ephemeral", "persistent", "http", "cloud"]
EmbedProviderArg = Literal["pinecone", "openai", "cohere", "huggingface"]
IncludeArg = Literal["documents", "metadatas", "embeddings", "uris", "distances"]
OutputArg = Literal["rows", "columns", "both"]


WRITE_DOC = """

        Give records either as Chroma's parallel columns - ``ids`` plus any of
        ``documents``, ``embeddings``, ``metadatas``, ``uris`` (same length,
        same order) - or as ``records=[{"id", "document", "embedding",
        "metadata", "uri"}]``, which cannot get out of step.

        Embeddings: pass them, or pass documents and let the collection's
        embedding route produce them (Chroma's embedding function, or the
        toolbox embedder recorded on the collection; ``embed_provider`` /
        ``embed_model`` force one for this call and fill in only the records
        that lack a vector). Width must match the collection's existing
        vectors.

        Metadata is flat: str, int, float, bool, or a non-empty array of one
        of those types (``{"genres": ["action", "drama"]}``) - arrays are
        what ``$contains`` filters on. No nested objects, no keys starting
        with ``#`` or ``$``.
        """


def _with_doc(extra: str):
    """Append shared documentation to a tool docstring before it is registered."""

    def decorator(func):
        func.__doc__ = (func.__doc__ or "").rstrip() + "\n" + extra
        return func

    return decorator


def _backend() -> ChromaBackend:
    from ...core.registry import get_backend

    return get_backend("chroma")  # type: ignore[return-value]


def _safe(write: bool = False):
    """Return failures as readable JSON. Write tools respect VTB_READ_ONLY."""

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
                text = f"{type(exc).__name__} {exc}".lower()
                hint = (
                    "This came from the Chroma SDK or server, not argument validation. "
                    "Check the client (chroma_heartbeat), the collection name and the data shape."
                )
                if any(s in text for s in ("onnx", "download", "forbidden", "proxy")):
                    hint = (
                        "Embedding failed while fetching a model. Chroma's default embedding "
                        "function downloads all-MiniLM-L6-v2 (ONNX, ~80 MB) on first use. Allow "
                        "that download, pass embeddings yourself, or create the collection with "
                        "toolbox_embedder / another embedding_function."
                    )
                elif "dimension" in text:
                    hint = (
                        "Vector width does not match the collection. Every embedding in a "
                        "collection must have the dimension of the first one written."
                    )
                return {"error": type(exc).__name__, "message": str(exc)[:2000], "hint": hint}

        return wrapper

    return decorator


def register(mcp: MCPServerType) -> None:
    register_status_provider("chroma", status)

    # ======================================================================
    # Clients
    # ======================================================================
    @mcp.tool()
    @_safe()
    def chroma_create_client(
        name: str,
        kind: ClientKindArg,
        path: str | None = None,
        host: str | None = None,
        port: int | None = None,
        ssl: bool | None = None,
        headers: dict[str, str] | None = None,
        tenant: str | None = None,
        database: str | None = None,
        api_key_env_var: str | None = None,
        replace: bool = False,
    ) -> dict[str, Any]:
        """Create a named Chroma client that other chroma_* tools can target.

        Kinds:

        * ``ephemeral`` - in-memory. Fast, nothing written to disk, gone when
          the server stops. All ephemeral clients in one process share a store.
        * ``persistent`` - local directory (``path``, default
          VTB_CHROMA_PATH = ~/.vector-toolbox/chroma). Survives restarts.
        * ``http`` - a self-hosted Chroma server (``chroma run``, Docker):
          ``host``/``port``/``ssl``, optional ``headers`` for auth proxies.
        * ``cloud`` - Chroma Cloud. Reads the key from CHROMA_API_KEY, or from
          the variable named by ``api_key_env_var``. ``tenant``/``database``
          default to CHROMA_TENANT / CHROMA_DATABASE, else to what the key
          implies. Raw keys are never accepted as arguments.

        A client called ``default`` is built automatically from VTB_CHROMA_*
        settings if you never create one; ``replace=true`` rebuilds a name.
        """
        entry = client_mod.register_client(
            name, kind, replace=replace, path=path, host=host, port=port, ssl=ssl,
            headers=headers, tenant=tenant, database=database, api_key_env_var=api_key_env_var,
        )
        return {"created": entry.describe()}

    @mcp.tool()
    @_safe()
    def chroma_list_clients() -> dict[str, Any]:
        """List the Chroma clients created in this session (the default one appears once used)."""
        settings = get_settings()
        return {
            "clients": client_mod.list_clients(),
            "default_settings": {
                "kind": settings.chroma_client,
                "path": settings.chroma_path if settings.chroma_client == "persistent" else None,
                "host": f"{settings.chroma_host}:{settings.chroma_port}"
                if settings.chroma_client == "http" else None,
                "cloud_api_key_set": bool(settings.chroma_api_key),
            },
        }

    @mcp.tool()
    @_safe()
    def chroma_remove_client(name: str) -> dict[str, Any]:
        """Forget a named client. Persistent/http/cloud data is untouched; an ephemeral store
        lives on while any other ephemeral client exists."""
        return client_mod.remove_client(name)

    @mcp.tool()
    @_safe()
    def chroma_heartbeat(client: str = "default") -> dict[str, Any]:
        """Check a client is reachable: heartbeat, server version, tenant, database, max batch size."""
        return _backend().heartbeat(client)

    # ======================================================================
    # Collections
    # ======================================================================
    @mcp.tool()
    @_safe()
    def chroma_list_embedding_functions() -> dict[str, Any]:
        """List the embedding options for a collection.

        Two routes, chosen per collection at creation:

        * **Chroma embedding functions** (``embedding_function`` on
          chroma_create_collection): Chroma embeds documents and query_texts
          itself, and persists the function's config with the collection so
          every client rebuilds it. Keys come from env vars
          (``api_key_env_var``), never from arguments.
        * **Toolbox embedders** (``toolbox_embedder``): this server embeds
          with the same providers the Pinecone tools use (VTB_EMBED_*), sends
          plain vectors, and records the provider on the collection metadata.

        With neither, Chroma's ``default`` function (all-MiniLM-L6-v2 via
        ONNX, 384 dims, downloaded on first use) embeds documents.
        """
        from ...embeddings import DENSE_PROVIDERS
        from .backend import known_embedding_functions

        dense, sparse = known_embedding_functions()
        settings = get_settings()
        return {
            "chroma_embedding_functions": sorted(dense),
            "chroma_sparse_embedding_functions": sorted(sparse),
            "sparse_note": "Sparse functions feed Chroma Cloud's Schema/Search API, not "
            "a collection's embedding_function.",
            "toolbox_providers": sorted(DENSE_PROVIDERS),
            "toolbox_default": f"{settings.embed_provider}/{settings.embed_model}",
            "examples": {
                "chroma_openai": {"name": "openai", "kwargs": {
                    "model_name": "text-embedding-3-small", "api_key_env_var": "OPENAI_API_KEY"}},
                "chroma_local": {"name": "sentence_transformer", "kwargs": {
                    "model_name": "all-MiniLM-L6-v2"}},
                "toolbox_openai": {"provider": "openai", "model": "text-embedding-3-small",
                                   "dimension": 1536},
            },
        }

    @mcp.tool()
    @_safe(write=True)
    def chroma_create_collection(
        name: str,
        client: str = "default",
        metadata: dict[str, Any] | None = None,
        embedding_function: dict[str, Any] | None = None,
        toolbox_embedder: dict[str, Any] | None = None,
        hnsw: dict[str, Any] | None = None,
        spann: dict[str, Any] | None = None,
        get_or_create: bool = False,
    ) -> dict[str, Any]:
        """Create a collection: name, metadata, embedding route and vector-index config.

        * ``name``: 3-512 chars of ``[a-zA-Z0-9._-]``, starting and ending
          alphanumeric. Unique per database.
        * ``metadata``: free-form, flat (str/int/float/bool) - e.g.
          ``{"owner": "search-team", "source": "docs"}``. Not for index
          settings: the old ``hnsw:space`` keys are rejected in favour of
          ``hnsw={"space": ...}``.
        * ``embedding_function``: a Chroma function, e.g.
          ``{"name": "openai", "kwargs": {"model_name": "text-embedding-3-small",
          "api_key_env_var": "OPENAI_API_KEY"}}``. See
          chroma_list_embedding_functions.
        * ``toolbox_embedder``: ``{"provider": "openai", "model": ...,
          "dimension": ...}`` - this server embeds instead of Chroma.
        * ``hnsw`` (single-node: ephemeral, persistent, most self-hosted):
          ``space`` (l2 default | ip | cosine), ``ef_construction`` (100),
          ``max_neighbors`` (16), ``ef_search`` (100), ``num_threads``,
          ``batch_size``, ``sync_threshold``, ``resize_factor``.
        * ``spann`` (Chroma Cloud / distributed): ``space``,
          ``search_nprobe`` / ``write_nprobe`` (64 default, max 128),
          ``ef_construction``, ``ef_search``, ``max_neighbors``,
          ``reassign_neighbor_count``, ``split_threshold``, ``merge_threshold``.

        ``space`` and ``ef_construction``/``max_neighbors`` are fixed once
        created - pick ``cosine`` for most text embedding models.
        ``get_or_create=true`` returns an existing collection unchanged.
        """
        return _backend().create_collection(
            client, name, metadata=metadata, embedding_function=embedding_function,
            toolbox_embedder=toolbox_embedder, hnsw=hnsw, spann=spann, get_or_create=get_or_create,
        )

    @mcp.tool()
    @_safe()
    def chroma_get_collection(name: str, client: str = "default") -> dict[str, Any]:
        """Describe one collection: id, record count, metadata, index configuration
        (HNSW/SPANN with every parameter), distance space and embedding route."""
        return _backend().describe_collection(client, name)

    @mcp.tool()
    @_safe()
    def chroma_list_collections(
        client: str = "default",
        limit: int | None = None,
        offset: int | None = None,
        include_counts: bool = False,
    ) -> dict[str, Any]:
        """List collections on a client (paged with limit/offset). ``include_counts`` adds
        a record count per collection at one extra call each."""
        return _backend().list_collections(client, limit=limit, offset=offset,
                                           include_counts=include_counts)

    @mcp.tool()
    @_safe(write=True)
    def chroma_modify_collection(
        name: str,
        client: str = "default",
        new_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        merge_metadata: bool = True,
    ) -> dict[str, Any]:
        """Rename a collection and/or change its metadata.

        ``merge_metadata=true`` (default) merges into the current metadata, and
        a key set to null is removed; ``false`` replaces the metadata wholesale
        (the toolbox's own ``vtb_embed_*`` keys are kept either way). For index
        settings use chroma_configure_collection.
        """
        return _backend().modify_collection(client, name, new_name=new_name, metadata=metadata,
                                            merge_metadata=merge_metadata)

    @mcp.tool()
    @_safe(write=True)
    def chroma_configure_collection(
        name: str,
        client: str = "default",
        hnsw: dict[str, Any] | None = None,
        spann: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Tune an existing collection's vector index. Returns the before/after config.

        Changeable after creation:

        * HNSW: ``ef_search`` (query-time recall vs speed), ``num_threads``,
          ``batch_size``, ``sync_threshold``, ``resize_factor``.
        * SPANN: ``search_nprobe`` (1-128), ``ef_search``.

        Everything else - ``space``, ``ef_construction``, ``max_neighbors``,
        ``write_nprobe`` - is fixed at creation; changing it means a new
        collection and a re-add.
        """
        return _backend().configure_collection(client, name, hnsw=hnsw, spann=spann)

    @mcp.tool()
    @_safe(write=True)
    def chroma_delete_collection(
        name: str, client: str = "default", confirm: bool = False
    ) -> dict[str, Any]:
        """Delete a collection and all its records. Requires ``confirm=true``; without it the
        call reports how many records would be lost."""
        return _backend().delete_collection(client, name, confirm=confirm)

    @mcp.tool()
    @_safe()
    def chroma_count(collection: str, client: str = "default") -> dict[str, Any]:
        """Number of records in a collection."""
        return _backend().count(client, collection)

    @mcp.tool()
    @_safe()
    def chroma_peek(collection: str, client: str = "default", limit: int = 10) -> dict[str, Any]:
        """The first ``limit`` records (documents, metadata, embeddings) - a quick look at the data."""
        return _backend().peek(client, collection, limit=limit)

    @mcp.tool()
    @_safe(write=True)
    def chroma_fork_collection(
        collection: str, new_name: str, client: str = "default"
    ) -> dict[str, Any]:
        """Copy-on-write fork of a collection under a new name (Chroma Cloud clients only)."""
        return _backend().fork(client, collection, new_name)

    # ======================================================================
    # Writing records
    # ======================================================================

    @mcp.tool()
    @_with_doc(WRITE_DOC)
    @_safe(write=True)
    def chroma_add(
        collection: str,
        client: str = "default",
        ids: list[str] | None = None,
        documents: list[str | None] | None = None,
        embeddings: list[list[float] | None] | None = None,
        metadatas: list[dict[str, Any] | None] | None = None,
        uris: list[str | None] | None = None,
        records: list[dict[str, Any]] | None = None,
        embed_provider: EmbedProviderArg | None = None,
        embed_model: str | None = None,
        embed_dimension: int | None = None,
        on_conflict: Literal["skip", "error"] = "skip",
    ) -> dict[str, Any]:
        """Add new records. Ids that already exist are never overwritten.

        Chroma itself skips existing ids silently; this tool reports them as
        ``skipped_existing`` (or fails with ``on_conflict="error"``). Use
        chroma_upsert to overwrite.
        """
        return _backend().write(
            client, collection, "add", ids=ids, documents=documents, embeddings=embeddings,
            metadatas=metadatas, uris=uris, records=records, embed_provider=embed_provider,
            embed_model=embed_model, embed_dimension=embed_dimension, on_conflict=on_conflict,
        )


    @mcp.tool()
    @_with_doc(WRITE_DOC)
    @_safe(write=True)
    def chroma_update(
        collection: str,
        client: str = "default",
        ids: list[str] | None = None,
        documents: list[str | None] | None = None,
        embeddings: list[list[float] | None] | None = None,
        metadatas: list[dict[str, Any] | None] | None = None,
        uris: list[str | None] | None = None,
        records: list[dict[str, Any]] | None = None,
        embed_provider: EmbedProviderArg | None = None,
        embed_model: str | None = None,
        embed_dimension: int | None = None,
        on_conflict: Literal["skip", "error"] = "skip",
    ) -> dict[str, Any]:
        """Update existing records - only the parts you pass change.

        * Metadata **merges**: listed keys are set, a key set to null is
          removed, unlisted keys are kept.
        * A new document is re-embedded by the collection's embedding route
          unless you also pass its embedding.
        * Ids that do not exist are reported as ``skipped_missing`` (Chroma
          ignores them silently); ``on_conflict="error"`` fails instead.
        """
        return _backend().write(
            client, collection, "update", ids=ids, documents=documents, embeddings=embeddings,
            metadatas=metadatas, uris=uris, records=records, embed_provider=embed_provider,
            embed_model=embed_model, embed_dimension=embed_dimension, on_conflict=on_conflict,
        )


    @mcp.tool()
    @_with_doc(WRITE_DOC)
    @_safe(write=True)
    def chroma_upsert(
        collection: str,
        client: str = "default",
        ids: list[str] | None = None,
        documents: list[str | None] | None = None,
        embeddings: list[list[float] | None] | None = None,
        metadatas: list[dict[str, Any] | None] | None = None,
        uris: list[str | None] | None = None,
        records: list[dict[str, Any]] | None = None,
        embed_provider: EmbedProviderArg | None = None,
        embed_model: str | None = None,
        embed_dimension: int | None = None,
    ) -> dict[str, Any]:
        """Insert records that are new and update those that exist; reports created vs updated."""
        return _backend().write(
            client, collection, "upsert", ids=ids, documents=documents, embeddings=embeddings,
            metadatas=metadatas, uris=uris, records=records, embed_provider=embed_provider,
            embed_model=embed_model, embed_dimension=embed_dimension,
        )


    @mcp.tool()
    @_safe(write=True)
    def chroma_conditional_transaction(
        collection: str,
        writes: list[dict[str, Any]],
        checks: list[dict[str, Any]] | None = None,
        client: str = "default",
        max_retries: int = 3,
        allow_non_atomic: bool = False,
        embed_provider: EmbedProviderArg | None = None,
        embed_model: str | None = None,
        embed_dimension: int | None = None,
    ) -> dict[str, Any]:
        """Read-check-write in one optimistic transaction: writes commit only if every check
        passes AND the records read are unchanged at commit time.

        ``checks`` (all must pass; any failure means nothing is written):

        * ``{"id": "doc-1"}`` / ``{"id": "doc-1", "exists": false}`` - must
          (not) exist.
        * ``{"id": "doc-1", "metadata": {"status": "draft", "version": 3}}`` -
          those keys must currently hold those values (compare-and-swap).
        * ``{"id": "doc-1", "document": "exact text"}``.
        * ``{"where": {...}, "where_document": {...}, "min_count": 1,
          "max_count": 10}`` - count of matching records. Only the ids it
          returns are protected against concurrent change.

        ``writes``: ``[{"op": "add"|"update"|"upsert", "ids": [...],
        "documents"/"embeddings"/"metadatas"/"uris": [...]}]`` (or
        ``"records": [...]``), and ``{"op": "delete", "ids": [...]}``.

        Chroma's rules, checked before anything runs: one collection, at
        most one write per id, deletes need explicit ids (no filter deletes),
        reads are ``get`` only (no similarity query). On a concurrency
        conflict the whole read-check-write is retried up to ``max_retries``.

        Example - publish a draft only if nobody else has:
        ``checks=[{"id": "post-7", "metadata": {"status": "draft"}}]``,
        ``writes=[{"op": "update", "ids": ["post-7"],
        "metadatas": [{"status": "published"}]}]``.

        Needs a chromadb release with ``Collection.conditional()``. Without
        one this refuses, unless ``allow_non_atomic=true`` (checks then
        writes, no isolation - reported as ``atomic: false``).
        """
        return _backend().conditional(
            client, collection, checks=checks, writes=writes, max_retries=max_retries,
            allow_non_atomic=allow_non_atomic, embed_provider=embed_provider,
            embed_model=embed_model, embed_dimension=embed_dimension,
        )

    @mcp.tool()
    @_safe(write=True)
    def chroma_delete(
        collection: str,
        client: str = "default",
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
        limit: int | None = None,
        delete_all: bool = False,
        dry_run: bool = False,
        confirm: bool = False,
    ) -> dict[str, Any]:
        """Delete records by id, by metadata filter, by document filter, or all of them.

        * ``ids`` and filters combine (AND): only listed ids that also match.
        * ``limit`` caps a filter delete.
        * ``dry_run=true`` reports what would be deleted without deleting.
        * ``delete_all=true`` empties the collection (kept, unlike
          chroma_delete_collection) and requires ``confirm=true``.

        Filters use the same grammar as chroma_get - see chroma_validate_filter.
        """
        return _backend().delete_records(
            client, collection, ids=ids, where=where, where_document=where_document, limit=limit,
            delete_all=delete_all, dry_run=dry_run, confirm=confirm,
        )

    # ======================================================================
    # Query and get
    # ======================================================================
    @mcp.tool()
    @_safe()
    def chroma_query(
        collection: str,
        client: str = "default",
        query_texts: list[str] | None = None,
        query_embeddings: list[list[float]] | None = None,
        n_results: int = 10,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
        ids: list[str] | None = None,
        include: list[IncludeArg] | None = None,
        embed_provider: EmbedProviderArg | None = None,
        embed_model: str | None = None,
        embed_dimension: int | None = None,
        output: OutputArg = "rows",
    ) -> dict[str, Any]:
        """Nearest-neighbour search. A batch API: several query texts or embeddings at once,
        each getting its own ``n_results`` list.

        * ``query_texts`` are embedded by the collection's embedding route;
          ``query_embeddings`` skip embedding. One or the other.
        * ``where`` (metadata), ``where_document`` (full text) and ``ids``
          restrict the candidates for every query in the batch.

        **Choosing which data is returned** - ``include`` any of
        ``documents``, ``metadatas``, ``embeddings``, ``uris``, ``distances``
        (default documents + metadatas + distances). Ids always come back.
        Leave out embeddings unless you need them; they are large.

        **Results shape** - ``output="rows"`` (default) gives
        ``queries: [{query, results: [{id, distance, similarity?, document,
        metadata}]}]``, nearest first. ``"columns"`` gives Chroma's native
        shape: one outer list per included field, one inner list per query
        (``ids[q][i]``, ``distances[q][i]``...). ``"both"`` returns both.
        Distance is lower-is-closer; for cosine/ip collections
        ``similarity = 1 - distance`` is added.
        """
        return _backend().query(
            client, collection, query_texts=query_texts, query_embeddings=query_embeddings,
            n_results=n_results, where=where, where_document=where_document, ids=ids,
            include=include, embed_provider=embed_provider, embed_model=embed_model,
            embed_dimension=embed_dimension, output=output,
        )

    @mcp.tool()
    @_safe()
    def chroma_get(
        collection: str,
        client: str = "default",
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
        limit: int | None = 100,
        offset: int | None = None,
        include: list[Literal["documents", "metadatas", "embeddings", "uris"]] | None = None,
        output: OutputArg = "rows",
    ) -> dict[str, Any]:
        """Fetch records by id and/or filter - no ranking, no embedding needed.

        Page with ``limit``/``offset`` (a full page returns ``next_offset``).
        ``include`` defaults to documents + metadatas; ``distances`` does not
        exist here. Results shape: ``output="rows"`` gives ``records: [{id,
        document, metadata}]``; ``"columns"`` gives Chroma's flat parallel
        lists (``ids[i]``, ``documents[i]``...) - one level shallower than a
        query result, because there is no per-query batch.
        """
        return _backend().get(
            client, collection, ids=ids, where=where, where_document=where_document,
            limit=limit, offset=offset, include=include, output=output,
        )

    # ======================================================================
    # Metadata filtering + full-text search
    # ======================================================================
    @mcp.tool()
    @_safe()
    def chroma_validate_filter(
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Check and normalise a filter without running it; returns the exact form Chroma gets.

        **where** (metadata):

        * Comparison: ``{"year": 2024}`` (= ``$eq``), ``$ne``, and on numbers
          ``$gt $gte $lt $lte`` - ``{"year": {"$gte": 2020}}``.
        * Inclusion: ``{"genre": {"$in": ["sci-fi", "drama"]}}``, ``$nin``.
        * Array metadata: ``{"genres": {"$contains": "action"}}``,
          ``$not_contains`` - the value must match the array's element type.
        * Logical: ``{"$and": [...]}``, ``{"$or": [...]}``, nestable.

        **where_document** (full text, on the document string):
        ``{"$contains": "refund"}``, ``$not_contains``, ``{"$regex":
        "^Invoice \\\\d+"}``, ``$not_regex``, combined with ``$and``/``$or``.

        Forgiving input accepted and rewritten: ``{}`` = no filter, several
        keys in one object = ``$and``, several operators on one field =
        ``$and``, a one-item ``$and`` = the item. Rejected with a reason: a
        bare list (use ``$in`` or ``$contains``), null, ``$exists``,
        ``$not``, regex look-around/backreferences.
        """
        return {
            "where": normalize_where(where),
            "where_document": normalize_where_document(where_document),
            "valid": True,
        }

    @mcp.tool()
    @_safe()
    def chroma_sample_metadata(
        collection: str, client: str = "default", sample_size: int = 50
    ) -> dict[str, Any]:
        """Sample records and describe the metadata shape: each key, its types (``str[]`` marks
        array metadata), how often it appears, examples, and the operators that fit. Run this
        before writing a where filter on a collection you did not build."""
        return _backend().sample_metadata(client, collection, sample_size=sample_size)

    @mcp.tool()
    @_safe()
    def chroma_full_text_search(
        collection: str,
        client: str = "default",
        contains: list[str] | None = None,
        not_contains: list[str] | None = None,
        regex: list[str] | None = None,
        not_regex: list[str] | None = None,
        match: Literal["all", "any"] = "all",
        where_document: dict[str, Any] | None = None,
        where: dict[str, Any] | None = None,
        query_text: str | None = None,
        query_embedding: list[float] | None = None,
        limit: int = 20,
        offset: int | None = None,
        include: list[IncludeArg] | None = None,
        embed_provider: EmbedProviderArg | None = None,
        embed_model: str | None = None,
    ) -> dict[str, Any]:
        """Find records by the text of their documents, optionally ranked by meaning.

        Builds the ``where_document`` filter for you:

        * ``contains=["refund", "invoice"]`` - substring match,
          case-sensitive. ``match="all"`` (default) needs every term,
          ``"any"`` needs one.
        * ``not_contains`` / ``not_regex`` - always exclusions.
        * ``regex=["(?i)refund"]`` - patterns; ``(?i)`` for case-insensitive.
          No look-around or backreferences.
        * ``where_document`` - a raw expression ANDed with the above.
        * ``where`` - metadata filter on top.

        Without a query this is a pure filter: matching records come back
        unranked, in storage order, paged by ``limit``/``offset``.

        **Combining with document search**: pass ``query_text`` (or
        ``query_embedding``) and the text filter restricts a vector query -
        semantic ranking over only the documents that contain your terms.
        """
        return _backend().full_text_search(
            client, collection, contains=contains, not_contains=not_contains, regex=regex,
            not_regex=not_regex, match=match, where_document=where_document, where=where,
            query_text=query_text, query_embedding=query_embedding, limit=limit, offset=offset,
            include=include, embed_provider=embed_provider, embed_model=embed_model,
        )


def status() -> dict[str, Any]:
    """Chroma section of ``vectortoolbox_status``."""
    settings = get_settings()
    try:
        import chromadb

        version = chromadb.__version__
        has_txn = hasattr(__import__("chromadb.api.models.Collection", fromlist=["Collection"]).Collection,
                          "conditional")
    except Exception:
        version, has_txn = None, False
    return {
        "installed": version is not None,
        "chromadb_version": version,
        "conditional_transactions": has_txn,
        "default_client_kind": settings.chroma_client,
        "clients": client_mod.list_clients(),
        "cloud_api_key_set": bool(settings.chroma_api_key),
    }
