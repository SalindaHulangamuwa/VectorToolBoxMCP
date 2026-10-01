"""Chroma backend.

Chroma's model is simpler than Pinecone's: a *collection* is one embedding
space plus, per record, an optional document string, a flat metadata dict and
an optional URI. There is no field schema to declare - instead a collection
carries an **embedding function** and an **index configuration** (HNSW on
single-node Chroma, SPANN on Chroma Cloud / distributed).

What this module adds on top of the SDK:

* named clients (``client.py``) so tools can target ephemeral, persistent,
  self-hosted and cloud Chroma in one session;
* filter validation (``filters.py``) that turns Chroma's strict grammar into
  readable errors before a request is sent;
* the toolbox's pluggable embedding providers as an alternative to Chroma's
  own embedding functions - the choice is remembered on the collection;
* write pre-checks, because Chroma's ``add`` silently skips existing ids and
  ``update`` silently skips missing ones;
* row-shaped results next to Chroma's native column-major shape.
"""

from __future__ import annotations

import inspect
import re
from collections import Counter, defaultdict
from typing import Any, Callable, Sequence

from ...core.base import VectorStoreBackend
from ...core.types import Hit, IndexCapabilities
from ...errors import (
    CapabilityError,
    ConfigurationError,
    ConfirmationRequired,
    NotFound,
    ToolboxError,
)
from . import client as client_mod
from .filters import (
    FilterError,
    build_text_filter,
    normalize_where,
    normalize_where_document,
)

# --------------------------------------------------------------------------
# Configuration vocabulary
# --------------------------------------------------------------------------
SPACES = ("l2", "ip", "cosine")
HNSW_CREATE_KEYS = {
    "space", "ef_construction", "max_neighbors", "ef_search",
    "num_threads", "batch_size", "sync_threshold", "resize_factor",
}
HNSW_UPDATE_KEYS = {"ef_search", "num_threads", "batch_size", "sync_threshold", "resize_factor"}
SPANN_CREATE_KEYS = {
    "space", "search_nprobe", "write_nprobe", "ef_construction", "ef_search",
    "max_neighbors", "reassign_neighbor_count", "split_threshold", "merge_threshold",
}
SPANN_UPDATE_KEYS = {"search_nprobe", "ef_search"}
SPANN_MAX_NPROBE = 128

GET_INCLUDE = ("documents", "metadatas", "embeddings", "uris")
QUERY_INCLUDE = GET_INCLUDE + ("distances",)

# Collection-metadata keys this server owns. A collection created with a
# toolbox embedder records it here, so later writes and queries embed with the
# same provider without the caller repeating it.
META_PROVIDER = "vtb_embed_provider"
META_MODEL = "vtb_embed_model"
META_DIMENSION = "vtb_embed_dimension"
VTB_META_KEYS = (META_PROVIDER, META_MODEL, META_DIMENSION)

WRITE_OPS = ("add", "update", "upsert", "delete")
_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{1,510}[a-zA-Z0-9]$")


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def jsonable(value: Any) -> Any:
    """Make SDK results JSON-safe (numpy arrays, numpy scalars, objects)."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return jsonable(tolist())
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:  # pragma: no cover
            pass
    if hasattr(value, "name") and callable(value.name):
        return describe_embedding_function(value)
    return str(value)


def describe_embedding_function(ef: Any) -> dict[str, Any] | None:
    if ef is None:
        return None
    out: dict[str, Any] = {"class": type(ef).__name__}
    try:
        out["name"] = ef.name()
    except Exception:
        pass
    try:
        config = ef.get_config()
        out["config"] = {k: v for k, v in (config or {}).items() if "key" not in k.lower() or k.endswith("env_var")}
    except Exception:
        pass
    return out


def _check_name(name: str) -> None:
    if not isinstance(name, str) or not _NAME_RE.match(name) or ".." in name:
        raise ToolboxError(
            f"Invalid collection name {name!r}: use 3-512 characters from [a-zA-Z0-9._-], "
            "starting and ending with a letter or digit, without '..'."
        )


def _check_include(include: Sequence[str] | None, allowed: tuple[str, ...], default: list[str]) -> list[str]:
    if include is None:
        return list(default)
    bad = [i for i in include if i not in allowed]
    if bad:
        hint = " (distances only exist for query results)" if "distances" in bad else ""
        raise ToolboxError(
            f"include has unknown value(s) {bad}{hint}. Allowed: {', '.join(allowed)}. "
            "ids are always returned."
        )
    return list(dict.fromkeys(include))


def _similarity(distance: float | None, space: str | None) -> float | None:
    # Chroma reports cosine and ip as 1 - similarity; l2 has no bounded form.
    if distance is None or space not in {"cosine", "ip"}:
        return None
    return round(1.0 - float(distance), 6)


def _validate_metadata(meta: Any, where: str, *, allow_none_values: bool) -> dict[str, Any] | None:
    if meta is None or meta == {}:
        return None
    if not isinstance(meta, dict):
        raise ToolboxError(f"{where} must be an object, got {type(meta).__name__}.")
    for key, value in meta.items():
        if not isinstance(key, str) or not key:
            raise ToolboxError(f"{where} has an empty or non-string key.")
        if key.startswith("#") or key.startswith("$"):
            raise ToolboxError(f"{where}.{key}: keys starting with '#' or '$' are reserved by Chroma.")
        if value is None:
            if not allow_none_values:
                raise ToolboxError(
                    f"{where}.{key} is null. Null removes a key only in update/upsert; "
                    "omit the key when adding."
                )
            continue
        if isinstance(value, dict):
            raise ToolboxError(f"{where}.{key} is an object. Chroma metadata is flat - no nesting.")
        if isinstance(value, list):
            if not value:
                raise ToolboxError(f"{where}.{key} is an empty array; Chroma rejects empty arrays.")
            kinds = {type(v) for v in value}
            if len(kinds) > 1 or not kinds <= {str, int, float, bool}:
                raise ToolboxError(
                    f"{where}.{key} must hold values of one type (all str, int, float or bool), "
                    f"got {[type(v).__name__ for v in value]}."
                )
        elif not isinstance(value, (str, int, float, bool)):
            raise ToolboxError(f"{where}.{key} has unsupported type {type(value).__name__}.")
    return meta


# --------------------------------------------------------------------------
# Embedding functions (Chroma-native)
# --------------------------------------------------------------------------
def known_embedding_functions() -> tuple[dict[str, Any], dict[str, Any]]:
    client_mod.import_chromadb()
    from chromadb.utils import embedding_functions as efs

    dense = dict(getattr(efs, "known_embedding_functions", {}))
    sparse = dict(getattr(efs, "sparse_known_embedding_functions", {}))
    return dense, sparse


def build_embedding_function(spec: dict[str, Any] | None) -> Any:
    """Build a Chroma embedding function from ``{"name": ..., "kwargs": {...}}``.

    Raw API keys are refused: they would sit in the conversation transcript.
    Every Chroma provider takes ``api_key_env_var`` naming the variable that
    holds the key, and that name - not the key - is what Chroma persists.
    """
    if spec is None:
        return None
    if not isinstance(spec, dict) or "name" not in spec:
        raise ConfigurationError(
            "embedding_function must look like {'name': 'openai', 'kwargs': {'model_name': "
            "'text-embedding-3-small', 'api_key_env_var': 'OPENAI_API_KEY'}}."
        )
    name = spec["name"]
    kwargs = dict(spec.get("kwargs") or {})
    config = spec.get("config")
    if any(k == "api_key" or k.endswith("_api_key") for k in kwargs):
        raise ConfigurationError(
            "Pass api_key_env_var (the NAME of an environment variable) instead of a raw "
            "api_key - keys passed as tool arguments end up in the conversation log."
        )
    dense, sparse = known_embedding_functions()
    if name in sparse and name not in dense:
        raise CapabilityError(
            f"{name!r} is a sparse embedding function. Sparse vectors are configured through "
            "a collection Schema on Chroma Cloud, not as a collection's embedding_function."
        )
    if name not in dense:
        raise ConfigurationError(
            f"Unknown Chroma embedding function {name!r}. Known: {', '.join(sorted(dense))}. "
            "See chroma_list_embedding_functions."
        )
    cls = dense[name]
    try:
        if config is not None:
            return cls.build_from_config(config)
        return cls(**kwargs)
    except TypeError as exc:
        try:
            params = [p for p in inspect.signature(cls.__init__).parameters if p != "self"]
        except (TypeError, ValueError):  # pragma: no cover
            params = []
        raise ConfigurationError(
            f"Could not build {name!r}: {exc}. Accepted kwargs: {', '.join(params) or 'none'}."
        ) from exc
    except Exception as exc:
        raise ConfigurationError(f"Could not build embedding function {name!r}: {exc}") from exc


def _validate_index_config(
    hnsw: dict[str, Any] | None,
    spann: dict[str, Any] | None,
    *,
    creating: bool,
    client_kind: str,
) -> dict[str, Any] | None:
    if hnsw and spann:
        raise CapabilityError(
            "Pass hnsw or spann, not both. A collection has one vector index: HNSW on "
            "single-node Chroma (ephemeral, persistent, most self-hosted servers), SPANN on "
            "Chroma Cloud and distributed deployments."
        )
    if not hnsw and not spann:
        return None
    kind, cfg = ("hnsw", hnsw) if hnsw else ("spann", spann)
    allowed = (
        (HNSW_CREATE_KEYS if kind == "hnsw" else SPANN_CREATE_KEYS)
        if creating
        else (HNSW_UPDATE_KEYS if kind == "hnsw" else SPANN_UPDATE_KEYS)
    )
    unknown = sorted(set(cfg) - allowed)
    if unknown:
        when = "at creation" if creating else "after creation (the rest are fixed at creation)"
        raise CapabilityError(
            f"{kind} parameter(s) {unknown} cannot be set {when}. Allowed: {', '.join(sorted(allowed))}."
        )
    if "space" in cfg and cfg["space"] not in SPACES:
        raise CapabilityError(f"space must be one of {', '.join(SPACES)}, got {cfg['space']!r}.")
    for key in ("search_nprobe", "write_nprobe"):
        if key in cfg and not (1 <= int(cfg[key]) <= SPANN_MAX_NPROBE):
            raise CapabilityError(f"spann.{key} must be 1-{SPANN_MAX_NPROBE} (64 or 128 recommended).")
    for key, value in cfg.items():
        if key != "space" and (not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0):
            raise CapabilityError(f"{kind}.{key} must be a positive number, got {value!r}.")
    if kind == "spann" and client_kind in {"ephemeral", "persistent"}:
        raise CapabilityError(
            f"SPANN is the Chroma Cloud / distributed index. A {client_kind} client runs "
            "single-node Chroma, which only has HNSW and would silently ignore a SPANN "
            "config. Pass hnsw=... here, or create the collection through a cloud client."
        )
    return {kind: dict(cfg)}


# --------------------------------------------------------------------------
# The backend
# --------------------------------------------------------------------------
class ChromaBackend(VectorStoreBackend):
    name = "chroma"

    # -- plumbing ----------------------------------------------------------
    @staticmethod
    def _entry(client: str | None):
        return client_mod.get_entry(client)

    def _collection(self, client: str | None, name: str, embedding_function: Any = None):
        entry = self._entry(client)
        chromadb = client_mod.import_chromadb()
        not_found = getattr(getattr(chromadb, "errors", None), "NotFoundError", ())
        kwargs: dict[str, Any] = {"name": name}
        if embedding_function is not None:
            kwargs["embedding_function"] = embedding_function
        try:
            return entry.client.get_collection(**kwargs)
        except Exception as exc:
            missing = (not_found and isinstance(exc, not_found)) or "does not exist" in str(exc)
            if not missing:
                raise
            try:
                names = [c.name for c in entry.client.list_collections(limit=25)]
            except Exception:  # pragma: no cover
                names = []
            raise NotFound(
                f"Collection {name!r} does not exist on client {entry.name!r} ({entry.kind}). "
                f"Existing: {', '.join(names) or 'none'}."
            ) from exc

    @staticmethod
    def _space(col: Any) -> str | None:
        try:
            cfg = col.configuration or {}
        except Exception:  # pragma: no cover
            return None
        for kind in ("hnsw", "spann"):
            block = cfg.get(kind) or {}
            if block.get("space"):
                return block["space"]
        return None

    @staticmethod
    def _batch_size(entry) -> int:
        try:
            return int(entry.client.get_max_batch_size())
        except Exception:
            return 1000

    @staticmethod
    def _toolbox_embedder(
        col: Any,
        provider: str | None,
        model: str | None,
        dimension: int | None,
    ):
        meta = col.metadata or {}
        provider = provider or meta.get(META_PROVIDER)
        if not provider:
            return None
        from ...embeddings import get_dense_embedder

        if model is None and provider == meta.get(META_PROVIDER):
            model = meta.get(META_MODEL)
        if dimension is None and provider == meta.get(META_PROVIDER):
            dimension = meta.get(META_DIMENSION)
        return get_dense_embedder(provider, model, dimension)

    # ======================================================================
    # Clients
    # ======================================================================
    def heartbeat(self, client: str | None) -> dict[str, Any]:
        entry = self._entry(client)
        out = entry.describe()
        out["heartbeat_ns"] = entry.client.heartbeat()
        try:
            out["server_version"] = entry.client.get_version()
        except Exception:  # pragma: no cover
            pass
        try:
            out["max_batch_size"] = entry.client.get_max_batch_size()
        except Exception:
            pass
        return out

    # ======================================================================
    # Collections
    # ======================================================================
    def list_collections(
        self, client: str | None = None, limit: int | None = None, offset: int | None = None,
        include_counts: bool = False,
    ) -> dict[str, Any]:
        entry = self._entry(client)
        cols = entry.client.list_collections(limit=limit, offset=offset)
        out = []
        for col in cols:
            item = {"name": col.name, "id": str(col.id), "metadata": jsonable(col.metadata)}
            if include_counts:
                item["count"] = col.count()
            out.append(item)
        try:
            total = entry.client.count_collections()
        except Exception:  # pragma: no cover
            total = None
        return {"client": entry.name, "total": total, "count": len(out), "collections": out}

    def create_collection(
        self,
        client: str | None,
        name: str,
        *,
        metadata: dict[str, Any] | None = None,
        embedding_function: dict[str, Any] | None = None,
        toolbox_embedder: dict[str, Any] | None = None,
        hnsw: dict[str, Any] | None = None,
        spann: dict[str, Any] | None = None,
        get_or_create: bool = False,
    ) -> dict[str, Any]:
        entry = self._entry(client)
        _check_name(name)
        if embedding_function and toolbox_embedder:
            raise ConfigurationError(
                "Choose one embedding route: embedding_function (Chroma embeds documents and "
                "query_texts itself) or toolbox_embedder (this server embeds them and sends "
                "vectors). Not both."
            )
        metadata = dict(metadata or {})
        legacy = [k for k in metadata if k.startswith("hnsw:")]
        if legacy:
            raise CapabilityError(
                f"Metadata keys {legacy} are the pre-1.0 way to configure HNSW. Pass them as "
                "hnsw={...} (e.g. hnsw={'space': 'cosine'}) instead."
            )
        clash = [k for k in VTB_META_KEYS if k in metadata]
        if clash:
            raise ConfigurationError(f"Metadata keys {clash} are reserved; use toolbox_embedder.")
        _validate_metadata(metadata, "metadata", allow_none_values=False)

        configuration = _validate_index_config(hnsw, spann, creating=True, client_kind=entry.kind) or {}

        ef = None
        if embedding_function:
            ef = build_embedding_function(embedding_function)
            space = (hnsw or spann or {}).get("space")
            if space:
                try:
                    supported = list(ef.supported_spaces())
                except Exception:
                    supported = []
                if supported and space not in supported:
                    raise CapabilityError(
                        f"{embedding_function['name']!r} supports spaces {supported}; "
                        f"{space!r} would give meaningless distances."
                    )

        if toolbox_embedder:
            provider = toolbox_embedder.get("provider")
            if not provider:
                raise ConfigurationError("toolbox_embedder needs at least {'provider': ...}.")
            from ...embeddings import DENSE_PROVIDERS

            if provider not in DENSE_PROVIDERS:
                raise ConfigurationError(
                    f"Unknown toolbox provider {provider!r}. Available: {', '.join(DENSE_PROVIDERS)}."
                )
            metadata[META_PROVIDER] = provider
            if toolbox_embedder.get("model"):
                metadata[META_MODEL] = toolbox_embedder["model"]
            if toolbox_embedder.get("dimension"):
                metadata[META_DIMENSION] = int(toolbox_embedder["dimension"])

        kwargs: dict[str, Any] = {"name": name}
        if metadata:
            kwargs["metadata"] = metadata
        if configuration:
            kwargs["configuration"] = configuration
        if ef is not None:
            kwargs["embedding_function"] = ef

        existed = False
        if get_or_create:
            try:
                existed = name in {c.name for c in entry.client.list_collections()}
            except Exception:  # pragma: no cover
                existed = False
            col = entry.client.get_or_create_collection(**kwargs)
        else:
            col = entry.client.create_collection(**kwargs)

        out = self._describe(col, entry)
        out["created"] = not existed
        if existed:
            out["note"] = (
                "Collection already existed; get_or_create returned it unchanged. Metadata and "
                "configuration passed here were NOT applied - use chroma_modify_collection / "
                "chroma_configure_collection."
            )
        return out

    def _describe(self, col: Any, entry) -> dict[str, Any]:
        metadata = dict(col.metadata or {})
        toolbox = {k.replace("vtb_embed_", ""): metadata[k] for k in VTB_META_KEYS if k in metadata}
        try:
            configuration = jsonable(col.configuration)
        except Exception:  # pragma: no cover
            configuration = None
        embedding = (
            {"route": "toolbox", **toolbox}
            if toolbox
            else {
                "route": "chroma",
                "embedding_function": describe_embedding_function(getattr(col, "_embedding_function", None)),
            }
        )
        return {
            "client": entry.name,
            "client_kind": entry.kind,
            "name": col.name,
            "id": str(col.id),
            "count": col.count(),
            "metadata": jsonable(metadata),
            "configuration": configuration,
            "space": self._space(col),
            "embedding": embedding,
        }

    def describe_collection(self, client: str | None, name: str) -> dict[str, Any]:
        entry = self._entry(client)
        return self._describe(self._collection(client, name), entry)

    def modify_collection(
        self,
        client: str | None,
        name: str,
        *,
        new_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        merge_metadata: bool = True,
    ) -> dict[str, Any]:
        entry = self._entry(client)
        col = self._collection(client, name)
        new_metadata = None
        if metadata is not None:
            current = dict(col.metadata or {})
            if merge_metadata:
                merged = {**current, **metadata}
                new_metadata = {k: v for k, v in merged.items() if v is not None}
            else:
                # Replacing must not drop the toolbox's own embedder record.
                keep = {k: current[k] for k in VTB_META_KEYS if k in current}
                new_metadata = {**keep, **{k: v for k, v in metadata.items() if v is not None}}
            _validate_metadata(new_metadata, "metadata", allow_none_values=False)
        if new_name is not None:
            _check_name(new_name)
        if new_name is None and new_metadata is None:
            raise ToolboxError("Nothing to change: pass new_name and/or metadata.")
        col.modify(name=new_name, metadata=new_metadata)
        return self._describe(self._collection(client, new_name or name), entry)

    def configure_collection(
        self,
        client: str | None,
        name: str,
        *,
        hnsw: dict[str, Any] | None = None,
        spann: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        entry = self._entry(client)
        configuration = _validate_index_config(hnsw, spann, creating=False, client_kind=entry.kind)
        if not configuration:
            raise ToolboxError("Pass hnsw={...} or spann={...} with the parameters to change.")
        col = self._collection(client, name)
        current = col.configuration or {}
        kind = next(iter(configuration))
        if current.get(kind) is None and current.get("hnsw" if kind == "spann" else "spann"):
            other = "hnsw" if kind == "spann" else "spann"
            raise CapabilityError(
                f"Collection {name!r} uses a {other.upper()} index; {kind} settings do not apply."
            )
        before = jsonable(current.get(kind))
        col.modify(configuration=configuration)
        after = jsonable((self._collection(client, name).configuration or {}).get(kind))
        return {"collection": name, "index": kind, "before": before, "after": after}

    def delete_collection(self, client: str | None, name: str, confirm: bool = False) -> dict[str, Any]:
        entry = self._entry(client)
        col = self._collection(client, name)
        count = col.count()
        if not confirm:
            raise ConfirmationRequired(
                f"Deleting collection {name!r} removes all {count} records on client "
                f"{entry.name!r} ({entry.kind}) and cannot be undone. Call again with confirm=true."
            )
        entry.client.delete_collection(name=name)
        return {"deleted": name, "records_removed": count, "client": entry.name}

    def count(self, client: str | None, name: str) -> dict[str, Any]:
        return {"collection": name, "count": self._collection(client, name).count()}

    def peek(self, client: str | None, name: str, limit: int = 10) -> dict[str, Any]:
        col = self._collection(client, name)
        res = col.peek(limit=limit)
        include = [k for k in GET_INCLUDE if res.get(k) is not None]
        return {"collection": name, "records": _rows_from_get(res, include)}

    def fork(self, client: str | None, name: str, new_name: str) -> dict[str, Any]:
        entry = self._entry(client)
        if entry.kind != "cloud":
            raise CapabilityError(
                f"Forking is a Chroma Cloud feature; client {entry.name!r} is {entry.kind}. "
                "Copy with chroma_get (include embeddings) + chroma_add instead."
            )
        col = self._collection(client, name)
        forked = col.fork(new_name=new_name)
        return {"source": name, "fork": forked.name, "id": str(forked.id), "count": forked.count()}

    # ======================================================================
    # Writing
    # ======================================================================
    @staticmethod
    def _columns(
        *,
        ids: list[str] | None,
        documents: list[str | None] | None,
        embeddings: list[list[float] | None] | None,
        metadatas: list[dict[str, Any] | None] | None,
        uris: list[str | None] | None,
        records: list[dict[str, Any]] | None,
    ) -> dict[str, list]:
        if records is not None:
            if any(x is not None for x in (ids, documents, embeddings, metadatas, uris)):
                raise ToolboxError("Pass records=[...] or the column lists (ids, documents, ...), not both.")
            unknown = {k for r in records for k in r} - {"id", "document", "embedding", "metadata", "uri"}
            if unknown:
                raise ToolboxError(
                    f"records have unknown keys {sorted(unknown)}. Each record: "
                    "{'id', 'document'?, 'embedding'?, 'metadata'?, 'uri'?}."
                )
            ids = [r.get("id") for r in records]
            documents = [r.get("document") for r in records]
            embeddings = [r.get("embedding") for r in records]
            metadatas = [r.get("metadata") for r in records]
            uris = [r.get("uri") for r in records]
        if not ids:
            raise ToolboxError("No records given: pass ids (+ documents/embeddings/metadatas) or records.")
        if any(not isinstance(i, str) or not i for i in ids):
            raise ToolboxError("Every id must be a non-empty string.")
        dupes = [i for i, n in Counter(ids).items() if n > 1]
        if dupes:
            raise ToolboxError(f"Duplicate ids in one request: {dupes[:10]}.")
        cols = {"documents": documents, "embeddings": embeddings, "metadatas": metadatas, "uris": uris}
        for key, value in cols.items():
            if value is not None and len(value) != len(ids):
                raise ToolboxError(f"{key} has {len(value)} entries but ids has {len(ids)}; they must match.")
        return {"ids": list(ids), **{k: (list(v) if v is not None else None) for k, v in cols.items()}}

    def _prepare_write(
        self,
        col: Any,
        op: str,
        cols: dict[str, list],
        *,
        embed_provider: str | None,
        embed_model: str | None,
        embed_dimension: int | None,
    ) -> dict[str, Any]:
        """Validate a columnar write and resolve embeddings. Returns SDK kwargs."""
        n = len(cols["ids"])
        docs, embs, metas, uris = cols["documents"], cols["embeddings"], cols["metadatas"], cols["uris"]

        if metas is not None:
            metas = [
                _validate_metadata(m, f"metadatas[{i}]", allow_none_values=op != "add")
                for i, m in enumerate(metas)
            ]
            if all(m is None for m in metas):
                metas = None

        has_doc = [d is not None for d in docs] if docs is not None else [False] * n
        has_emb = [e is not None for e in embs] if embs is not None else [False] * n

        embedder = self._toolbox_embedder(col, embed_provider, embed_model, embed_dimension)
        if embedder is not None:
            missing = [i for i in range(n) if not has_emb[i] and has_doc[i]]
            if missing:
                vectors = embedder.embed_documents([docs[i] for i in missing])
                embs = list(embs) if embs is not None else [None] * n
                for i, vec in zip(missing, vectors):
                    embs[i] = vec
                has_emb = [e is not None for e in embs]
            if op != "update" and not all(has_emb):
                bad = [cols["ids"][i] for i in range(n) if not has_emb[i]][:10]
                raise ToolboxError(
                    f"Records {bad} have neither an embedding nor a document to embed."
                )
        elif embs is not None and any(has_emb) and not all(has_emb):
            raise ToolboxError(
                "Some records carry an embedding and some do not. Chroma needs all or none in one "
                "call - split the call, or set embed_provider so this server fills the gaps."
            )
        elif op in {"add", "upsert"} and not any(has_emb):
            has_uri = [u is not None for u in uris] if uris is not None else [False] * n
            bad = [cols["ids"][i] for i in range(n) if not (has_doc[i] or has_uri[i])][:10]
            if bad:
                raise ToolboxError(
                    f"Records {bad} have no embedding and no document, so there is nothing to "
                    "embed. Pass embeddings, or documents for the collection's embedding function."
                )

        if embs is not None and not any(e is not None for e in embs):
            embs = None
        if docs is not None and not any(has_doc):
            docs = None
        if uris is not None and all(u is None for u in uris):
            uris = None
        if op == "update" and docs is None and embs is None and metas is None and uris is None:
            raise ToolboxError("update needs at least one of documents, embeddings, metadatas or uris.")

        kwargs: dict[str, Any] = {"ids": cols["ids"]}
        for key, value in (("documents", docs), ("embeddings", embs), ("metadatas", metas), ("uris", uris)):
            if value is not None:
                kwargs[key] = value
        return kwargs

    @staticmethod
    def _existing_ids(col: Any, ids: list[str]) -> set[str]:
        found: set[str] = set()
        for start in range(0, len(ids), 1000):
            found.update(col.get(ids=ids[start : start + 1000], include=[])["ids"])
        return found

    @staticmethod
    def _subset(kwargs: dict[str, Any], keep: Callable[[str], bool]) -> dict[str, Any]:
        mask = [keep(i) for i in kwargs["ids"]]
        return {k: [v for v, m in zip(vals, mask) if m] for k, vals in kwargs.items()}

    def write(
        self,
        client: str | None,
        collection: str,
        op: str,
        *,
        ids: list[str] | None = None,
        documents: list[str | None] | None = None,
        embeddings: list[list[float] | None] | None = None,
        metadatas: list[dict[str, Any] | None] | None = None,
        uris: list[str | None] | None = None,
        records: list[dict[str, Any]] | None = None,
        embed_provider: str | None = None,
        embed_model: str | None = None,
        embed_dimension: int | None = None,
        on_conflict: str = "skip",
    ) -> dict[str, Any]:
        if op not in {"add", "update", "upsert"}:
            raise ToolboxError(f"Unknown write op {op!r}.")
        if on_conflict not in {"skip", "error"}:
            raise ToolboxError("on_conflict must be 'skip' or 'error'.")
        entry = self._entry(client)
        col = self._collection(client, collection)
        cols = self._columns(
            ids=ids, documents=documents, embeddings=embeddings,
            metadatas=metadatas, uris=uris, records=records,
        )
        existing = self._existing_ids(col, cols["ids"])
        report: dict[str, Any] = {"collection": collection, "op": op}

        # Chroma's add silently ignores ids that exist; update silently ignores
        # ids that do not. Surface both instead of reporting a false success.
        if op == "add" and existing:
            if on_conflict == "error":
                raise ToolboxError(
                    f"{len(existing)} id(s) already exist: {sorted(existing)[:10]}. add never "
                    "overwrites - use chroma_upsert or chroma_update."
                )
            report["skipped_existing"] = sorted(existing)
            keep = lambda i: i not in existing  # noqa: E731
        elif op == "update":
            missing = [i for i in cols["ids"] if i not in existing]
            if missing and on_conflict == "error":
                raise ToolboxError(
                    f"{len(missing)} id(s) do not exist: {missing[:10]}. update never creates - "
                    "use chroma_upsert or chroma_add."
                )
            if missing:
                report["skipped_missing"] = missing
            keep = lambda i: i in existing  # noqa: E731
        else:
            keep = lambda i: True  # noqa: E731
            report["updated"] = len([i for i in cols["ids"] if i in existing])
            report["created"] = len(cols["ids"]) - report["updated"]

        cols = {k: ([v for v, i in zip(vals, cols["ids"]) if keep(i)] if vals is not None else None)
                for k, vals in cols.items()}
        if not cols["ids"]:
            report["written"] = 0
            return report

        kwargs = self._prepare_write(
            col, op, cols,
            embed_provider=embed_provider, embed_model=embed_model, embed_dimension=embed_dimension,
        )
        batch = self._batch_size(entry)
        method = getattr(col, op)
        written = 0
        for start in range(0, len(kwargs["ids"]), batch):
            chunk = {k: v[start : start + batch] for k, v in kwargs.items()}
            method(**chunk)
            written += len(chunk["ids"])
        report["written"] = written
        if "embeddings" not in kwargs and "documents" in kwargs:
            report["embedded_by"] = "chroma-embedding-function"
        elif "embeddings" in kwargs:
            embedder = self._toolbox_embedder(col, embed_provider, embed_model, embed_dimension)
            report["embedded_by"] = (
                f"toolbox:{embedder.provider}/{embedder.model}" if embedder else "provided"
            )
        return report

    def delete_records(
        self,
        client: str | None,
        collection: str,
        *,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
        limit: int | None = None,
        delete_all: bool = False,
        dry_run: bool = False,
        confirm: bool = False,
    ) -> dict[str, Any]:
        col = self._collection(client, collection)
        where = normalize_where(where)
        where_document = normalize_where_document(where_document)

        if delete_all:
            if ids or where or where_document:
                raise ToolboxError("delete_all=true deletes everything; do not combine it with ids or filters.")
            total = col.count()
            if dry_run:
                return {"collection": collection, "dry_run": True, "would_delete": total}
            if not confirm:
                raise ConfirmationRequired(
                    f"delete_all removes all {total} records from {collection!r}. Call again with "
                    "confirm=true (the collection itself is kept)."
                )
            deleted = 0
            while True:
                page = col.get(limit=1000, include=[])["ids"]
                if not page:
                    break
                col.delete(ids=page)
                deleted += len(page)
            return {"collection": collection, "deleted": deleted}

        if not ids and not where and not where_document:
            raise ToolboxError(
                "Say what to delete: ids, where, where_document - or delete_all=true with confirm=true."
            )
        if limit is not None and not (where or where_document):
            raise ToolboxError("limit only applies to filter deletes (where / where_document).")

        if dry_run:
            res = col.get(ids=ids, where=where, where_document=where_document, limit=limit, include=[])
            return {"collection": collection, "dry_run": True, "would_delete": len(res["ids"]),
                    "ids": res["ids"][:100]}

        kwargs: dict[str, Any] = {"ids": ids, "where": where, "where_document": where_document}
        if limit is not None:
            kwargs["limit"] = limit
        try:
            result = col.delete(**kwargs)
        except TypeError:  # pragma: no cover - older SDKs without limit
            kwargs.pop("limit", None)
            result = col.delete(**kwargs)
        out = {"collection": collection}
        if isinstance(result, dict) and "deleted" in result:
            out["deleted"] = result["deleted"]
        else:
            out["result"] = jsonable(result)
        return out

    # ======================================================================
    # Conditional transactions
    # ======================================================================
    def conditional(
        self,
        client: str | None,
        collection: str,
        *,
        writes: list[dict[str, Any]],
        checks: list[dict[str, Any]] | None = None,
        max_retries: int = 3,
        allow_non_atomic: bool = False,
        embed_provider: str | None = None,
        embed_model: str | None = None,
        embed_dimension: int | None = None,
    ) -> dict[str, Any]:
        col = self._collection(client, collection)
        checks = list(checks or [])
        if not writes:
            raise ToolboxError("A transaction needs at least one write.")

        # -- validate checks -------------------------------------------------
        id_checks, filter_checks = [], []
        for i, chk in enumerate(checks):
            if "id" in chk:
                extra = set(chk) - {"id", "exists", "metadata", "document"}
                if extra:
                    raise ToolboxError(f"checks[{i}] has unknown keys {sorted(extra)}.")
                id_checks.append(chk)
            elif "where" in chk or "where_document" in chk:
                extra = set(chk) - {"where", "where_document", "min_count", "max_count"}
                if extra:
                    raise ToolboxError(f"checks[{i}] has unknown keys {sorted(extra)}.")
                if "min_count" not in chk and "max_count" not in chk:
                    raise ToolboxError(f"checks[{i}] is a filter check; give min_count and/or max_count.")
                filter_checks.append({
                    **chk,
                    "where": normalize_where(chk.get("where")),
                    "where_document": normalize_where_document(chk.get("where_document")),
                })
            else:
                raise ToolboxError(
                    f"checks[{i}] must name an 'id' (with exists / metadata / document "
                    "expectations) or a 'where' filter (with min_count / max_count)."
                )

        # -- validate + prepare writes (embedding happens before the txn) ---
        prepared: list[tuple[str, dict[str, Any]]] = []
        seen: Counter = Counter()
        for i, w in enumerate(writes):
            op = w.get("op")
            if op not in WRITE_OPS:
                raise ToolboxError(f"writes[{i}].op must be one of {', '.join(WRITE_OPS)}.")
            if op == "delete":
                extra = set(w) - {"op", "ids"}
                if extra:
                    raise ToolboxError(
                        f"writes[{i}]: transactional deletes take explicit ids only (no where); "
                        f"got {sorted(extra)}."
                    )
                if not w.get("ids"):
                    raise ToolboxError(f"writes[{i}]: delete needs ids.")
                kwargs = {"ids": list(w["ids"])}
            else:
                cols = self._columns(
                    ids=w.get("ids"), documents=w.get("documents"), embeddings=w.get("embeddings"),
                    metadatas=w.get("metadatas"), uris=w.get("uris"), records=w.get("records"),
                )
                kwargs = self._prepare_write(
                    col, op, cols, embed_provider=embed_provider,
                    embed_model=embed_model, embed_dimension=embed_dimension,
                )
            seen.update(kwargs["ids"])
            prepared.append((op, kwargs))
        twice = [i for i, n in seen.items() if n > 1]
        if twice:
            raise ToolboxError(
                f"ids {twice[:10]} are written more than once. A transaction buffers at most one "
                "write per id - merge them into a single write."
            )

        state: dict[str, Any] = {}

        def body(txn: Any) -> None:
            state["attempts"] = state.get("attempts", 0) + 1
            results: list[dict[str, Any]] = []
            failed = False
            if id_checks:
                wanted = list(dict.fromkeys(c["id"] for c in id_checks))
                got = _txn_get(txn, ids=wanted, include=["metadatas", "documents"])
                found = {
                    rid: {
                        "metadata": (got.get("metadatas") or [None] * len(got["ids"]))[k],
                        "document": (got.get("documents") or [None] * len(got["ids"]))[k],
                    }
                    for k, rid in enumerate(got["ids"])
                }
                for chk in id_checks:
                    ok, reason = _evaluate_id_check(chk, found.get(chk["id"]))
                    results.append({"check": chk, "passed": ok, **({"reason": reason} if reason else {})})
                    failed = failed or not ok
            for chk in filter_checks:
                got = _txn_get(txn, where=chk["where"], where_document=chk["where_document"], include=[])
                n = len(got["ids"])
                ok = (chk.get("min_count") is None or n >= chk["min_count"]) and (
                    chk.get("max_count") is None or n <= chk["max_count"]
                )
                results.append({"check": {k: v for k, v in chk.items() if v is not None},
                                "passed": ok, "matched": n})
                failed = failed or not ok
            state["checks"] = results
            state["failed"] = failed
            if failed:
                return  # nothing buffered -> nothing committed
            for op, kwargs in prepared:
                getattr(txn, op)(**kwargs)

        summary = [{"op": op, "ids": kw["ids"][:20], "count": len(kw["ids"])} for op, kw in prepared]
        factory = getattr(col, "conditional", None)
        if factory is None:
            if not allow_non_atomic:
                version = getattr(client_mod.import_chromadb(), "__version__", "?")
                raise CapabilityError(
                    f"The installed chromadb ({version}) has no conditional transactions "
                    "(Collection.conditional). Upgrade with `pip install -U chromadb`, or pass "
                    "allow_non_atomic=true to run the same checks then writes WITHOUT isolation - "
                    "a concurrent writer could slip in between."
                )
            body(_DirectTxn(col))
            return {
                "collection": collection, "atomic": False,
                "committed": not state["failed"], "checks": state["checks"],
                "writes": summary if not state["failed"] else [],
                "note": "Ran without isolation (allow_non_atomic=true).",
            }

        txn = factory()
        try:
            if hasattr(txn, "run"):
                result = txn.run(body, max_retries=max_retries)
            else:  # pragma: no cover - manual-only API
                body(txn)
                result = txn.commit()
        except Exception as exc:
            text = f"{type(exc).__name__}: {exc}"
            if "conflict" in text.lower():
                return {
                    "collection": collection, "atomic": True, "committed": False,
                    "error": "conflict",
                    "message": f"Records changed under the transaction on every attempt "
                    f"({state.get('attempts', 0)} tries). {text[:500]}",
                    "checks": state.get("checks", []),
                }
            raise
        return {
            "collection": collection, "atomic": True,
            "committed": not state.get("failed", False),
            "attempts": state.get("attempts"),
            "checks": state.get("checks", []),
            "writes": summary if not state.get("failed") else [],
            "result": jsonable(result),
        }

    # ======================================================================
    # Reading
    # ======================================================================
    def get(
        self,
        client: str | None,
        collection: str,
        *,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
        limit: int | None = None,
        offset: int | None = None,
        include: list[str] | None = None,
        output: str = "rows",
    ) -> dict[str, Any]:
        col = self._collection(client, collection)
        include = _check_include(include, GET_INCLUDE, ["documents", "metadatas"])
        res = col.get(
            ids=ids, where=normalize_where(where), where_document=normalize_where_document(where_document),
            limit=limit, offset=offset, include=include,
        )
        out: dict[str, Any] = {"collection": collection, "count": len(res["ids"]), "include": include}
        if output in {"rows", "both"}:
            out["records"] = _rows_from_get(res, include)
        if output in {"columns", "both"}:
            out["columns"] = _columns_view(res, include)
        if output not in {"rows", "columns", "both"}:
            raise ToolboxError("output must be 'rows', 'columns' or 'both'.")
        if limit is not None and len(res["ids"]) == limit:
            out["next_offset"] = (offset or 0) + limit
        return out

    def query(
        self,
        client: str | None,
        collection: str,
        *,
        query_texts: list[str] | None = None,
        query_embeddings: list[list[float]] | None = None,
        n_results: int = 10,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
        ids: list[str] | None = None,
        include: list[str] | None = None,
        embed_provider: str | None = None,
        embed_model: str | None = None,
        embed_dimension: int | None = None,
        output: str = "rows",
    ) -> dict[str, Any]:
        if bool(query_texts) == bool(query_embeddings):
            raise ToolboxError("Pass exactly one of query_texts or query_embeddings.")
        if n_results < 1:
            raise ToolboxError("n_results must be at least 1.")
        if output not in {"rows", "columns", "both"}:
            raise ToolboxError("output must be 'rows', 'columns' or 'both'.")
        col = self._collection(client, collection)
        include = _check_include(include, QUERY_INCLUDE, ["documents", "metadatas", "distances"])

        embedded_by = "provided"
        if query_texts:
            embedder = self._toolbox_embedder(col, embed_provider, embed_model, embed_dimension)
            if embedder is not None:
                query_embeddings = [embedder.embed_query(t) for t in query_texts]
                embedded_by = f"toolbox:{embedder.provider}/{embedder.model}"
            else:
                embedded_by = "chroma-embedding-function"

        kwargs: dict[str, Any] = {
            "n_results": n_results,
            "where": normalize_where(where),
            "where_document": normalize_where_document(where_document),
            "include": include,
        }
        if ids:
            kwargs["ids"] = ids
        if query_embeddings is not None:
            kwargs["query_embeddings"] = query_embeddings
        else:
            kwargs["query_texts"] = query_texts
        res = col.query(**kwargs)

        space = self._space(col)
        labels = query_texts or [f"embedding[{i}]" for i in range(len(query_embeddings or []))]
        out: dict[str, Any] = {
            "collection": collection, "space": space, "include": include, "embedded_by": embedded_by,
            "distance_note": "Lower distance = closer."
            + (" similarity = 1 - distance." if space in {"cosine", "ip"} else ""),
        }
        if output in {"rows", "both"}:
            out["queries"] = [
                {"query": label, "results": _rows_from_query(res, qi, include, space)}
                for qi, label in enumerate(labels)
            ]
        if output in {"columns", "both"}:
            out["columns"] = _columns_view(res, include)
        return out

    def full_text_search(
        self,
        client: str | None,
        collection: str,
        *,
        contains: list[str] | None = None,
        not_contains: list[str] | None = None,
        regex: list[str] | None = None,
        not_regex: list[str] | None = None,
        match: str = "all",
        where_document: dict[str, Any] | None = None,
        where: dict[str, Any] | None = None,
        query_text: str | None = None,
        query_embedding: list[float] | None = None,
        limit: int = 20,
        offset: int | None = None,
        include: list[str] | None = None,
        embed_provider: str | None = None,
        embed_model: str | None = None,
    ) -> dict[str, Any]:
        built = build_text_filter(
            contains=contains, not_contains=not_contains, regex=regex, not_regex=not_regex, match=match
        )
        extra = normalize_where_document(where_document)
        clauses = [c for c in (built, extra) if c]
        if not clauses:
            raise FilterError(
                "Give at least one of contains, not_contains, regex, not_regex or where_document."
            )
        combined = clauses[0] if len(clauses) == 1 else {"$and": clauses}

        if query_text or query_embedding:
            res = self.query(
                client, collection,
                query_texts=[query_text] if query_text else None,
                query_embeddings=[query_embedding] if query_embedding else None,
                n_results=limit, where=where, where_document=combined, include=include,
                embed_provider=embed_provider, embed_model=embed_model,
            )
            res["where_document"] = combined
            res["ranking"] = "vector similarity, restricted to documents matching the text filter"
            return res
        res = self.get(
            client, collection, where=where, where_document=combined,
            limit=limit, offset=offset, include=include,
        )
        res["where_document"] = combined
        res["ranking"] = "none - text filters match or not; results come back in storage order"
        return res

    def sample_metadata(self, client: str | None, collection: str, sample_size: int = 50) -> dict[str, Any]:
        col = self._collection(client, collection)
        res = col.get(limit=sample_size, include=["metadatas", "documents"])
        metas = res.get("metadatas") or []
        fields: dict[str, dict[str, Any]] = defaultdict(lambda: {"types": Counter(), "present_in": 0, "examples": []})
        for meta in metas:
            for key, value in (meta or {}).items():
                info = fields[key]
                info["present_in"] += 1
                if isinstance(value, list):
                    elem = type(value[0]).__name__ if value else "?"
                    info["types"][f"{elem}[]"] += 1
                else:
                    info["types"][type(value).__name__] += 1
                if len(info["examples"]) < 3 and value not in info["examples"]:
                    info["examples"].append(value)
        described = {}
        for key, info in sorted(fields.items()):
            types = sorted(info["types"])
            if any(t.endswith("[]") for t in types):
                ops = ["$contains", "$not_contains"]
            elif set(types) <= {"int", "float"}:
                ops = ["$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin"]
            else:
                ops = ["$eq", "$ne", "$in", "$nin"]
            described[key] = {
                "types": types, "present_in": info["present_in"],
                "examples": jsonable(info["examples"]), "operators": ops,
            }
        docs = res.get("documents") or []
        return {
            "collection": collection,
            "sampled": len(res["ids"]),
            "total": col.count(),
            "records_with_documents": sum(1 for d in docs if d),
            "fields": described,
        }

    # ======================================================================
    # VectorStoreBackend contract (index == collection, namespace == database)
    # ======================================================================
    def list_indexes(self) -> list[dict[str, Any]]:
        return self.list_collections(None)["collections"]

    def describe_index(self, index: str) -> dict[str, Any]:
        return self.describe_collection(None, index)

    def capabilities(self, index: str, client: str | None = None) -> IndexCapabilities:
        col = self._collection(client, index)
        sample = self.sample_metadata(client, index, sample_size=20)
        space = self._space(col)
        cfg = col.configuration or {}
        notes = [
            "Chroma has no field schema: one embedding per record, plus a document string and "
            "flat metadata. Any metadata key is filterable as soon as it is written.",
            "Full-text search is filter-only ($contains / $regex on documents); it narrows results "
            "but does not score them. Combine with a vector query for ranking.",
        ]
        if cfg.get("spann"):
            notes.append("Index: SPANN (Chroma Cloud / distributed).")
        elif cfg.get("hnsw"):
            notes.append("Index: HNSW (single-node).")
        return IndexCapabilities(
            index=index,
            api="collection",
            dense_fields=["embedding"],
            fts_fields=["document"],
            filterable_fields=sorted(sample["fields"]),
            metrics={"embedding": space} if space else {},
            supported_search_modes=["dense", "text", "regex", "dense+text"],
            notes=notes,
        )

    def delete_index(self, index: str, confirm: bool = False) -> dict[str, Any]:
        return self.delete_collection(None, index, confirm=confirm)

    def list_namespaces(self, index: str) -> list[dict[str, Any]]:
        entry = self._entry(None)
        return [{"name": getattr(entry.client, "database", None),
                 "note": "Chroma partitions by tenant/database, not namespaces."}]

    def delete_namespace(self, index: str, namespace: str) -> dict[str, Any]:
        raise CapabilityError("Chroma has no namespaces inside a collection; delete records by filter instead.")

    def upsert(self, index: str, namespace: str, records: list[dict[str, Any]]) -> dict[str, Any]:
        return self.write(None, index, "upsert", records=records)

    def fetch(self, index: str, namespace: str, ids: list[str] | None = None) -> list[dict[str, Any]]:
        return self.get(None, index, ids=ids)["records"]

    def delete(
        self, index: str, namespace: str, ids: list[str] | None = None,
        filter: dict[str, Any] | None = None, delete_all: bool = False,
    ) -> dict[str, Any]:
        return self.delete_records(None, index, ids=ids, where=filter, delete_all=delete_all,
                                   confirm=delete_all)

    def search(
        self, index: str, namespace: str, *, mode: str, query: str | None = None,
        vector: list[float] | None = None, fields: list[str] | None = None, top_k: int = 10,
        filter: dict[str, Any] | None = None, include_fields: list[str] | None = None,
    ) -> list[Hit]:
        if mode == "text":
            res = self.full_text_search(None, index, contains=[query] if query else None,
                                        where=filter, limit=top_k)
            return [Hit(id=r["id"], score=1.0, fields=r) for r in res["records"]]
        if mode in {"dense", "auto"}:
            res = self.query(None, index, query_texts=[query] if query and vector is None else None,
                             query_embeddings=[vector] if vector is not None else None,
                             n_results=top_k, where=filter)
            return [
                Hit(id=r["id"], score=r.get("similarity", -r.get("distance", 0.0)), fields=r)
                for r in res["queries"][0]["results"]
            ]
        raise CapabilityError(f"Chroma collections support modes dense and text, not {mode!r}.")

    def stats(self, index: str) -> dict[str, Any]:
        return self.describe_collection(None, index)


# --------------------------------------------------------------------------
# Result shaping
# --------------------------------------------------------------------------
_FIELD_NAMES = {"documents": "document", "metadatas": "metadata", "embeddings": "embedding",
                "uris": "uri", "distances": "distance"}


def _rows_from_get(res: dict[str, Any], include: list[str]) -> list[dict[str, Any]]:
    rows = []
    for i, rid in enumerate(res["ids"]):
        row: dict[str, Any] = {"id": rid}
        for key in include:
            values = res.get(key)
            if values is not None:
                row[_FIELD_NAMES[key]] = jsonable(values[i])
        rows.append(row)
    return rows


def _rows_from_query(res: dict[str, Any], qi: int, include: list[str], space: str | None) -> list[dict[str, Any]]:
    rows = []
    for i, rid in enumerate(res["ids"][qi]):
        row: dict[str, Any] = {"id": rid}
        for key in include:
            values = res.get(key)
            if values is not None and values[qi] is not None:
                row[_FIELD_NAMES[key]] = jsonable(values[qi][i])
        sim = _similarity(row.get("distance"), space)
        if sim is not None:
            row["similarity"] = sim
        rows.append(row)
    return rows


def _columns_view(res: dict[str, Any], include: list[str]) -> dict[str, Any]:
    """Chroma's native column-major shape, JSON-safe, only the included keys."""
    out = {"ids": jsonable(res["ids"])}
    for key in include:
        if res.get(key) is not None:
            out[key] = jsonable(res[key])
    return out


# --------------------------------------------------------------------------
# Transaction helpers
# --------------------------------------------------------------------------
def _txn_get(txn: Any, **kwargs: Any) -> dict[str, Any]:
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    try:
        return txn.get(**kwargs)
    except TypeError:
        kwargs.pop("include", None)
        return txn.get(**kwargs)


def _evaluate_id_check(chk: dict[str, Any], record: dict[str, Any] | None) -> tuple[bool, str | None]:
    want_exists = chk.get("exists", True if ("metadata" in chk or "document" in chk) else None)
    if want_exists is None:
        want_exists = True
    if not want_exists:
        return (record is None, None if record is None else "record exists")
    if record is None:
        return False, "record does not exist"
    if "metadata" in chk:
        meta = record.get("metadata") or {}
        diff = {k: meta.get(k) for k, v in chk["metadata"].items() if meta.get(k) != v}
        if diff:
            return False, f"metadata differs: currently {jsonable(diff)}"
    if "document" in chk and record.get("document") != chk["document"]:
        return False, "document differs"
    return True, None


class _DirectTxn:
    """Stand-in for a transaction when the SDK has none (allow_non_atomic)."""

    def __init__(self, col: Any):
        self._col = col

    def get(self, **kwargs: Any):
        return self._col.get(**kwargs)

    def add(self, **kwargs: Any):
        return self._col.add(**kwargs)

    def update(self, **kwargs: Any):
        return self._col.update(**kwargs)

    def upsert(self, **kwargs: Any):
        return self._col.upsert(**kwargs)

    def delete(self, **kwargs: Any):
        return self._col.delete(**kwargs)
