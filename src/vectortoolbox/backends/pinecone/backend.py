"""Pinecone backend.

Implements the shared :class:`VectorStoreBackend` contract plus the
Pinecone-specific operations the MCP tools expose. Everything that talks to
the network lives here; the tool layer is argument validation and formatting.
"""

from __future__ import annotations

import time
from typing import Any, Iterable, Sequence

from ...core.base import VectorStoreBackend
from ...core.fusion import fuse
from ...core.types import (
    DenseFieldSpec,
    Hit,
    IndexCapabilities,
    SparseFieldSpec,
    TextFieldSpec,
)
from ...embeddings import get_dense_embedder, get_sparse_embedder
from ...errors import CapabilityError, ConfirmationRequired
from . import ttl as ttl_mod
from .client import get_client, get_index, reset_client_cache
from .schema import (
    MAX_DOCUMENTS_PER_REQUEST,
    MAX_SCORE_BY_CLAUSES,
    _to_builtins,
    build_schema,
    capabilities_from_index_model,
    require_mode,
    resolve_field,
    validate_documents,
    validate_field_name,
    validate_top_k,
)

_CAPS_CACHE: dict[str, tuple[float, IndexCapabilities]] = {}
_CAPS_TTL_SECONDS = 60.0


class PineconeBackend(VectorStoreBackend):
    name = "pinecone"

    # ------------------------------------------------------------------
    # Index lifecycle
    # ------------------------------------------------------------------
    def create_index(
        self,
        *,
        name: str,
        dense_fields: Sequence[DenseFieldSpec] | None = None,
        sparse_fields: Sequence[SparseFieldSpec] | None = None,
        text_fields: Sequence[TextFieldSpec] | None = None,
        cloud: str | None = None,
        region: str | None = None,
        pod: dict[str, Any] | None = None,
        read_capacity: dict[str, Any] | None = None,
        deletion_protection: str | None = None,
        tags: dict[str, str] | None = None,
        cmek_id: str | None = None,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        from ...config import get_settings

        settings = get_settings()
        schema = build_schema(dense_fields, sparse_fields, text_fields)

        if pod:
            deployment = {"deployment_type": "pod", **pod}
        else:
            deployment = {
                "deployment_type": "managed",
                "cloud": cloud or settings.pinecone_cloud,
                "region": region or settings.pinecone_region,
            }

        kwargs: dict[str, Any] = {
            "name": name,
            "schema": schema,
            "deployment": deployment,
        }
        if read_capacity:
            kwargs["read_capacity"] = read_capacity
        if deletion_protection:
            kwargs["deletion_protection"] = deletion_protection
        if tags:
            kwargs["tags"] = tags
        if cmek_id:
            kwargs["cmek_id"] = cmek_id
        if timeout is not None:
            kwargs["timeout"] = timeout

        model = get_client().indexes.create(**kwargs)
        _CAPS_CACHE.pop(name, None)
        return _to_builtins(model)

    def create_index_for_model(
        self,
        *,
        name: str,
        model: str,
        field_map: dict[str, str],
        cloud: str | None = None,
        region: str | None = None,
        metric: str | None = None,
        dimension: int | None = None,
        read_parameters: dict[str, Any] | None = None,
        write_parameters: dict[str, Any] | None = None,
        filterable_fields: dict[str, Any] | None = None,
        deletion_protection: str | None = None,
        tags: dict[str, str] | None = None,
        read_capacity: dict[str, Any] | None = None,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        from ...config import get_settings

        settings = get_settings()
        embed: dict[str, Any] = {"model": model, "field_map": field_map}
        if metric:
            embed["metric"] = metric
        if dimension:
            embed["dimension"] = dimension
        if read_parameters:
            embed["read_parameters"] = read_parameters
        if write_parameters:
            embed["write_parameters"] = write_parameters

        kwargs: dict[str, Any] = {
            "name": name,
            "cloud": cloud or settings.pinecone_cloud,
            "region": region or settings.pinecone_region,
            "embed": embed,
        }
        if filterable_fields:
            kwargs["schema"] = {"fields": filterable_fields}
        if deletion_protection:
            kwargs["deletion_protection"] = deletion_protection
        if tags:
            kwargs["tags"] = tags
        if read_capacity:
            kwargs["read_capacity"] = read_capacity
        if timeout is not None:
            kwargs["timeout"] = timeout

        created = get_client().indexes.create_for_model(**kwargs)
        _CAPS_CACHE.pop(name, None)
        return _to_builtins(created)

    def list_indexes(self) -> list[dict[str, Any]]:
        out = []
        for model in get_client().indexes.list():
            data = _to_builtins(model)
            caps = capabilities_from_index_model(model)
            out.append(
                {
                    "name": data.get("name"),
                    "host": data.get("host"),
                    "ready": (data.get("status") or {}).get("ready"),
                    "deletion_protection": data.get("deletion_protection"),
                    "tags": data.get("tags"),
                    "search_modes": caps.supported_search_modes,
                    "dense_fields": caps.dense_fields,
                    "sparse_fields": caps.sparse_fields,
                    "fts_fields": caps.fts_fields,
                    "semantic_text_fields": caps.semantic_text_fields,
                }
            )
        return out

    def describe_index(self, index: str) -> dict[str, Any]:
        return _to_builtins(get_client().indexes.describe(name=index))

    def capabilities(self, index: str, *, refresh: bool = False) -> IndexCapabilities:
        cached = _CAPS_CACHE.get(index)
        if cached and not refresh and (time.monotonic() - cached[0]) < _CAPS_TTL_SECONDS:
            return cached[1]
        model = get_client().indexes.describe(name=index)
        caps = capabilities_from_index_model(model)
        _CAPS_CACHE[index] = (time.monotonic(), caps)
        return caps

    def configure_index(
        self,
        index: str,
        *,
        deletion_protection: str | None = None,
        tags: dict[str, str] | None = None,
        read_capacity: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"name": index}
        if deletion_protection is not None:
            kwargs["deletion_protection"] = deletion_protection
        if tags is not None:
            kwargs["tags"] = tags
        if read_capacity is not None:
            kwargs["read_capacity"] = read_capacity
        result = get_client().indexes.configure(**kwargs)
        _CAPS_CACHE.pop(index, None)
        return _to_builtins(result)

    def delete_index(self, index: str, *, confirm: bool = False) -> dict[str, Any]:
        if not confirm:
            raise ConfirmationRequired(
                f"Deleting index {index!r} destroys every record in it. Call again with "
                "confirm=true if that is what you intend."
            )
        get_client().indexes.delete(name=index)
        _CAPS_CACHE.pop(index, None)
        reset_client_cache()
        return {"deleted": index}

    # ------------------------------------------------------------------
    # Namespaces
    # ------------------------------------------------------------------
    def list_namespaces(self, index: str) -> list[dict[str, Any]]:
        out = []
        for page in get_index(index).list_namespaces():
            data = _to_builtins(page)
            if isinstance(data, dict) and "namespaces" in data:
                out.extend(data["namespaces"])
            else:
                out.append(data)
        return out

    def describe_namespace(self, index: str, namespace: str) -> dict[str, Any]:
        return _to_builtins(get_index(index).describe_namespace(name=namespace))

    def create_namespace(self, index: str, namespace: str) -> dict[str, Any]:
        return _to_builtins(get_index(index).create_namespace(name=namespace))

    def delete_namespace(
        self, index: str, namespace: str, *, confirm: bool = False
    ) -> dict[str, Any]:
        if not confirm:
            raise ConfirmationRequired(
                f"Deleting namespace {namespace!r} removes all of its records. "
                "Call again with confirm=true."
            )
        get_index(index).delete_namespace(name=namespace)
        return {"deleted_namespace": namespace, "index": index}

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def _prepare_documents(
        self,
        caps: IndexCapabilities,
        documents: Sequence[dict[str, Any]],
        *,
        embed_source_field: str | None,
        dense_field: str | None,
        sparse_field: str | None,
        embed_provider: str | None,
        embed_model: str | None,
        embed_dimension: int | None,
        sparse_provider: str | None,
        sparse_model: str | None,
        ttl_seconds: int | None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        docs = [dict(d) for d in documents]
        report: dict[str, Any] = {"embedded_dense": 0, "embedded_sparse": 0}

        if caps.api == "integrated":
            return ttl_mod.stamp(docs, ttl_seconds), {
                "embedded_dense": 0,
                "embedded_sparse": 0,
                "note": "Integrated index - Pinecone embeds the mapped text field server-side.",
            }

        if embed_source_field:
            if caps.dense_fields:
                target = resolve_field(caps, "dense", dense_field)
                pending = [d for d in docs if target not in d]
                if pending:
                    texts = [str(d.get(embed_source_field, "")) for d in pending]
                    if any(not t for t in texts):
                        raise CapabilityError(
                            f"Some documents have no {embed_source_field!r} value to embed."
                        )
                    embedder = get_dense_embedder(embed_provider, embed_model, embed_dimension)
                    expected = caps.dimensions.get(target)
                    vectors = embedder.embed_documents(texts)
                    if expected and vectors and len(vectors[0]) != expected:
                        raise CapabilityError(
                            f"Field {target!r} on index {caps.index!r} expects dimension "
                            f"{expected}, but {embedder.provider}/{embedder.model} produced "
                            f"{len(vectors[0])}. Pick a matching model or set embed_dimension."
                        )
                    for doc, vector in zip(pending, vectors):
                        doc[target] = vector
                    report["embedded_dense"] = len(pending)
                    report["dense_field"] = target
                    report["dense_model"] = f"{embedder.provider}/{embedder.model}"

            if caps.sparse_fields:
                target = resolve_field(caps, "sparse", sparse_field)
                pending = [d for d in docs if target not in d]
                if pending:
                    embedder = get_sparse_embedder(sparse_provider, sparse_model)
                    vectors = embedder.embed_documents(
                        [str(d.get(embed_source_field, "")) for d in pending]
                    )
                    for doc, vector in zip(pending, vectors):
                        doc[target] = vector
                    report["embedded_sparse"] = len(pending)
                    report["sparse_field"] = target
                    report["sparse_model"] = f"{embedder.provider}/{embedder.model}"

        missing_vectors = [
            d.get("_id", "<no _id>")
            for d in docs
            if caps.dense_fields and not any(f in d for f in caps.dense_fields)
        ]
        if missing_vectors and not embed_source_field:
            report["warning"] = (
                f"{len(missing_vectors)} document(s) carry no value for dense field(s) "
                f"{caps.dense_fields}. Pass embed_source_field to have them embedded, or "
                "include the vectors yourself."
            )

        return ttl_mod.stamp(docs, ttl_seconds), report

    def upsert_documents(
        self,
        index: str,
        namespace: str,
        documents: Sequence[dict[str, Any]],
        *,
        embed_source_field: str | None = None,
        dense_field: str | None = None,
        sparse_field: str | None = None,
        embed_provider: str | None = None,
        embed_model: str | None = None,
        embed_dimension: int | None = None,
        sparse_provider: str | None = None,
        sparse_model: str | None = None,
        ttl_seconds: int | None = None,
        batch_size: int = 50,
    ) -> dict[str, Any]:
        caps = self.capabilities(index)
        prepared, report = self._prepare_documents(
            caps,
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
        )

        if caps.api == "vectors":
            raise CapabilityError(
                f"Index {index!r} is a Vectors-API index: its records are "
                "{id, values, sparse_values, metadata}, not schema documents. Use "
                "pinecone_upsert_vectors instead."
            )

        validate_documents(prepared, caps)
        batch_size = max(1, min(batch_size, MAX_DOCUMENTS_PER_REQUEST))

        idx = get_index(index)
        if caps.api == "integrated":
            idx.upsert_records(records=prepared, namespace=namespace)
        elif len(prepared) > batch_size:
            idx.documents.batch_upsert(
                namespace=namespace,
                documents=prepared,
                batch_size=batch_size,
                show_progress=False,
            )
        else:
            idx.documents.upsert(namespace=namespace, documents=prepared)

        report.update(
            {
                "index": index,
                "namespace": namespace,
                "upserted": len(prepared),
                "ttl_seconds": ttl_seconds,
                "expires_at": ttl_mod.expiry_from_ttl(ttl_seconds),
                "ttl_field": ttl_mod.TTL_FIELD if ttl_seconds else None,
            }
        )
        return report

    def upsert_vectors(
        self,
        index: str,
        namespace: str,
        vectors: Sequence[dict[str, Any]],
        *,
        ttl_seconds: int | None = None,
        batch_size: int = 100,
    ) -> dict[str, Any]:
        """Legacy Vectors API upsert: {id, values, sparse_values, metadata}."""
        prepared = []
        expires_at = ttl_mod.expiry_from_ttl(ttl_seconds)
        for vector in vectors:
            item = dict(vector)
            metadata = dict(item.get("metadata") or {})
            for key in metadata:
                validate_field_name(key)
            if expires_at is not None:
                metadata.setdefault(ttl_mod.TTL_FIELD, expires_at)
            item["metadata"] = metadata
            prepared.append(item)

        idx = get_index(index)
        total = 0
        for start in range(0, len(prepared), batch_size):
            chunk = prepared[start : start + batch_size]
            idx.upsert(vectors=chunk, namespace=namespace)
            total += len(chunk)
        return {
            "index": index,
            "namespace": namespace,
            "upserted": total,
            "ttl_seconds": ttl_seconds,
        }

    def update_documents(
        self,
        index: str,
        namespace: str,
        *,
        documents: Sequence[dict[str, Any]] | None = None,  # noqa: D417
        filter: dict[str, Any] | None = None,
        set_fields: dict[str, Any] | None = None,
        remove_fields: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        if self.capabilities(index).api == "vectors":
            raise CapabilityError(
                f"Index {index!r} is a Vectors-API index. Use pinecone_update_vector "
                "(one record at a time, by id)."
            )
        kwargs: dict[str, Any] = {"namespace": namespace}
        if documents:
            kwargs["documents"] = [dict(d) for d in documents]
        if filter:
            kwargs["filter"] = filter
        if set_fields:
            kwargs["set_fields"] = set_fields
        if remove_fields:
            kwargs["remove_fields"] = list(remove_fields)
        return _to_builtins(get_index(index).documents.update(**kwargs))

    def update_vector(
        self,
        index: str,
        namespace: str,
        *,
        id: str,
        values: Sequence[float] | None = None,
        sparse_values: dict[str, Any] | None = None,
        set_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"id": id, "namespace": namespace}
        if values is not None:
            kwargs["values"] = list(values)
        if sparse_values is not None:
            kwargs["sparse_values"] = sparse_values
        if set_metadata is not None:
            kwargs["set_metadata"] = set_metadata
        return _to_builtins(get_index(index).update(**kwargs))

    def delete_documents(
        self,
        index: str,
        namespace: str,
        *,
        ids: Sequence[str] | None = None,
        filter: dict[str, Any] | None = None,
        delete_all: bool = False,
        confirm: bool = False,
    ) -> dict[str, Any]:
        if delete_all and not confirm:
            raise ConfirmationRequired(
                f"delete_all would empty namespace {namespace!r} of index {index!r}. "
                "Call again with confirm=true."
            )
        if not ids and not filter and not delete_all:
            raise CapabilityError("Pass ids, a filter, or delete_all=true.")

        caps = self.capabilities(index)
        idx = get_index(index)
        if caps.api == "documents":
            kwargs: dict[str, Any] = {"namespace": namespace, "delete_all": delete_all}
            if ids:
                kwargs["ids"] = list(ids)
            if filter:
                kwargs["filter"] = filter
            return _to_builtins(idx.documents.delete(**kwargs))

        idx.delete(
            ids=list(ids) if ids else None,
            filter=filter,
            delete_all=delete_all,
            namespace=namespace,
        )
        return {"deleted": True, "index": index, "namespace": namespace}

    def purge_expired(
        self, index: str, namespace: str, *, confirm: bool = False
    ) -> dict[str, Any]:
        if not confirm:
            raise ConfirmationRequired(
                "purge_expired permanently deletes every record whose TTL has lapsed. "
                "Call again with confirm=true."
            )
        result = self.delete_documents(
            index, namespace, filter=ttl_mod.expired_filter(), confirm=True
        )
        return {"purged_before": ttl_mod.now(), "result": result}

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def fetch_documents(
        self,
        index: str,
        namespace: str,
        *,
        ids: Sequence[str] | None = None,
        filter: dict[str, Any] | None = None,
        include_fields: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        caps = self.capabilities(index)
        idx = get_index(index)
        if caps.api == "documents":
            kwargs: dict[str, Any] = {"namespace": namespace}
            if ids:
                kwargs["ids"] = list(ids)
            if filter:
                kwargs["filter"] = filter
            if include_fields:
                kwargs["include_fields"] = list(include_fields)
            response = _to_builtins(idx.documents.fetch(**kwargs))
            docs = response.get("documents") or response.get("matches") or []
            return docs[:limit] if limit else docs

        # Vectors API. It fetches strictly by id, so "show me some records"
        # has to become "find some ids, then fetch them". A filter goes through
        # fetch_by_metadata, which is a lookup rather than a ranked search.
        def flatten(payload: Any) -> list[dict[str, Any]]:
            """Metadata up to the top level; vector values dropped by default.

            A fetch returns the raw vector alongside the metadata, and at 3072
            dimensions a handful of records is megabytes of floats that nobody
            asked for. Name "values" or "sparse_values" in include_fields to
            get them back.
            """
            wanted = set(include_fields or [])
            keep_dense = "values" in wanted
            keep_sparse = "sparse_values" in wanted or "sparseValues" in wanted

            records = []
            vectors = payload.get("vectors") or {}
            items = vectors.items() if isinstance(vectors, dict) else [
                (v.get("id"), v) for v in vectors
            ]
            for key, value in items:
                value = dict(value or {})
                record = {"_id": key or value.get("id")}
                record.update(value.pop("metadata", None) or {})
                if keep_dense and value.get("values") is not None:
                    record["values"] = value["values"]
                if keep_sparse and value.get("sparseValues") is not None:
                    record["sparse_values"] = value["sparseValues"]
                records.append(record)
            return records

        if not ids and filter:
            response = _to_builtins(
                idx.fetch_by_metadata(filter=filter, namespace=namespace, limit=limit or 20)
            )
            return flatten(response)

        if not ids:
            ids = self.list_ids(index, namespace, limit=limit or 20)
            if not ids:
                return []

        return flatten(_to_builtins(idx.fetch(ids=list(ids), namespace=namespace)))

    def list_ids(
        self, index: str, namespace: str, *, prefix: str | None = None, limit: int = 100
    ) -> list[str]:
        idx = get_index(index)
        caps = self.capabilities(index)
        ids: list[str] = []

        if caps.api == "documents":
            for record in idx.documents.list(namespace=namespace, prefix=prefix, limit=limit):
                data = _to_builtins(record)
                found = data.get("id") or data.get("_id")
                if found:
                    ids.append(found)
                if len(ids) >= limit:
                    break
            return ids

        # Vectors API: one ListResponse per page, each carrying `vectors` of
        # ids. Page size is capped at 100 server-side, so ask for a legal page
        # and keep paging until we have what the caller wanted.
        for page in idx.list(prefix=prefix, limit=min(limit, 100), namespace=namespace):
            data = _to_builtins(page)
            entries = data.get("vectors") or data.get("ids") or []
            if isinstance(entries, dict):
                entries = list(entries)
            for entry in entries:
                found = entry if isinstance(entry, str) else (entry or {}).get("id")
                if found:
                    ids.append(found)
                if len(ids) >= limit:
                    return ids
        return ids

    def sample_metadata(
        self, index: str, namespace: str, *, sample_size: int = 20
    ) -> dict[str, Any]:
        """Fetch a handful of records and describe the shape of their fields.

        Useful before writing a filter: it reports which fields exist, their
        observed types, and a few example values, without dumping the whole
        namespace into the conversation.
        """
        docs = self.fetch_documents(
            index, namespace, include_fields=["*"], limit=sample_size
        )

        profile: dict[str, dict[str, Any]] = {}
        for doc in docs[:sample_size]:
            payload = doc.get("metadata") if "metadata" in doc else doc
            for key, value in (payload or {}).items():
                if key.startswith("_") and key != ttl_mod.TTL_FIELD:
                    continue
                entry = profile.setdefault(
                    key, {"types": set(), "examples": [], "present_in": 0}
                )
                entry["types"].add(type(value).__name__)
                entry["present_in"] += 1
                if len(entry["examples"]) < 3 and not isinstance(value, (list, dict)):
                    entry["examples"].append(value)
                elif len(entry["examples"]) < 3 and isinstance(value, list) and len(value) < 12:
                    entry["examples"].append(value)

        caps = self.capabilities(index)
        return {
            "index": index,
            "namespace": namespace,
            "sampled": len(docs[:sample_size]),
            "fields": {
                key: {
                    "types": sorted(value["types"]),
                    "present_in": value["present_in"],
                    "examples": value["examples"],
                    "declared_in_schema": key
                    in (caps.dense_fields + caps.sparse_fields + caps.fts_fields),
                }
                for key, value in sorted(profile.items())
            },
            "hint": "Fields not declared in the schema are stored as metadata and filterable "
            "with $eq/$ne/$gt/$gte/$lt/$lte/$in/$nin/$exists/$and/$or/$not. Declared "
            "full-text fields additionally accept $match_phrase/$match_all/$match_any.",
        }

    def stats(self, index: str, *, namespace: str | None = None) -> dict[str, Any]:
        kwargs = {"namespace": namespace} if namespace else {}
        return _to_builtins(get_index(index).describe_index_stats(**kwargs))

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def _run_clauses(
        self,
        index: str,
        namespace: str,
        clauses: list[dict[str, Any]],
        *,
        top_k: int,
        include_fields: Sequence[str] | None,
        filter: dict[str, Any] | None,
    ) -> list[Hit]:
        """Run one search request.

        Several ``text`` / ``query_string`` clauses may travel together and are
        combined server-side with equal weight. A ``dense_vector`` or
        ``sparse_vector`` clause must be the only clause in the request - that
        constraint is why hybrid search is fused client-side.
        """
        validate_top_k(top_k)
        vector_clauses = [c for c in clauses if c["type"].endswith("_vector")]
        if vector_clauses and len(clauses) > 1:
            raise CapabilityError(
                "Pinecone requires a dense_vector or sparse_vector clause to be the only clause "
                "in a request. Use mode='hybrid' to combine a vector signal with text."
            )
        response = get_index(index).documents.search(
            namespace=namespace,
            score_by=clauses,
            top_k=top_k,
            include_fields=list(include_fields) if include_fields else ["*"],
            filter=filter,
        )
        data = _to_builtins(response)
        hits = []
        for match in data.get("matches", []):
            match = dict(match)
            doc_id = match.pop("_id", None) or match.pop("id", None)
            score = match.pop("_score", None)
            if score is None:
                score = match.pop("score", 0.0)
            hits.append(Hit(id=str(doc_id), score=float(score), fields=match))
        return hits

    def _dense_clause(
        self,
        caps: IndexCapabilities,
        query: str | None,
        vector: Sequence[float] | None,
        field: str | None,
        embed_provider: str | None,
        embed_model: str | None,
        embed_dimension: int | None,
    ) -> dict[str, Any]:
        target = resolve_field(caps, "dense", field)
        if vector is None:
            if not query:
                raise CapabilityError("A dense search needs either query text or a vector.")
            embedder = get_dense_embedder(embed_provider, embed_model, embed_dimension)
            vector = embedder.embed_query(query)
        expected = caps.dimensions.get(target)
        if expected and len(vector) != expected:
            raise CapabilityError(
                f"Query vector has dimension {len(vector)} but field {target!r} expects "
                f"{expected}. Use the same embedding model the index was written with."
            )
        return {"type": "dense_vector", "field": target, "values": list(vector)}

    def _sparse_clause(
        self,
        caps: IndexCapabilities,
        query: str | None,
        sparse_vector: dict[str, Any] | None,
        field: str | None,
        sparse_provider: str | None,
        sparse_model: str | None,
    ) -> dict[str, Any]:
        target = resolve_field(caps, "sparse", field)
        if sparse_vector is None:
            if not query:
                raise CapabilityError("A sparse search needs either query text or a sparse vector.")
            sparse_vector = get_sparse_embedder(sparse_provider, sparse_model).embed_query(query)
        return {"type": "sparse_vector", "field": target, "sparse_values": sparse_vector}

    def search(
        self,
        index: str,
        namespace: str,
        *,
        mode: str = "auto",
        query: str | None = None,
        field_queries: dict[str, str] | None = None,
        vector: Sequence[float] | None = None,
        sparse_vector: dict[str, Any] | None = None,
        fields: Sequence[str] | None = None,
        dense_field: str | None = None,
        sparse_field: str | None = None,
        top_k: int = 10,
        filter: dict[str, Any] | None = None,
        include_fields: Sequence[str] | None = None,
        embed_provider: str | None = None,
        embed_model: str | None = None,
        embed_dimension: int | None = None,
        sparse_provider: str | None = None,
        sparse_model: str | None = None,
        fusion: str = "rrf",
        weights: dict[str, float] | None = None,
        candidate_multiplier: int = 3,
        exclude_expired: bool = True,
    ) -> dict[str, Any]:
        caps = self.capabilities(index)

        if mode == "auto":
            if caps.api == "integrated":
                mode = "records"
            elif caps.api == "vectors":
                mode = "hybrid" if caps.sparse_fields else "dense"
            elif "hybrid" in caps.supported_search_modes:
                mode = "hybrid"
            elif caps.dense_fields:
                mode = "dense"
            elif caps.fts_fields:
                mode = "text"
            elif caps.sparse_fields:
                mode = "sparse"
            else:
                raise CapabilityError(f"Index {index!r} declares no searchable field.")

        if caps.api == "integrated" and mode != "records":
            raise CapabilityError(
                f"Index {index!r} uses integrated inference. Use pinecone_search_records "
                "(mode='records'); it embeds the query for you."
            )

        if caps.api == "vectors":
            if mode in {"text", "query_string"}:
                raise CapabilityError(
                    f"Index {index!r} is a Vectors-API index and declares no full-text field, "
                    "so BM25 and Lucene search are not available on it. Full-text search has to "
                    "be declared when an index is created, which means a new index."
                )
            return self._search_vectors_api(
                index,
                namespace,
                mode=mode,
                query=query,
                vector=vector,
                sparse_vector=sparse_vector,
                caps=caps,
                top_k=top_k,
                filter=filter,
                embed_provider=embed_provider,
                embed_model=embed_model,
                embed_dimension=embed_dimension,
                sparse_provider=sparse_provider,
                sparse_model=sparse_model,
                exclude_expired=exclude_expired,
            )

        effective_filter = ttl_mod.live_filter(filter) if exclude_expired else filter

        if mode in {"text", "query_string"}:
            require_mode(caps, mode)
            targets = list(field_queries) if field_queries else (list(fields) if fields else caps.fts_fields)
            unknown = [f for f in targets if f not in caps.fts_fields]
            if unknown:
                raise CapabilityError(
                    f"Fields {unknown} are not full-text searchable on {index!r}. "
                    f"FTS fields: {caps.fts_fields or 'none'}."
                )
            if not query and not field_queries:
                raise CapabilityError(f"A {mode} search needs a query string.")

            if mode == "text":
                # A `text` clause names exactly one field. Scoring across
                # several fields means several clauses in one request, which
                # the server combines with equal weight per field.
                clauses = [
                    {
                        "type": "text",
                        "field": name,
                        "query": (field_queries or {}).get(name, query),
                    }
                    for name in targets
                ]
                if len(clauses) > MAX_SCORE_BY_CLAUSES:
                    raise CapabilityError(
                        f"{len(clauses)} text clauses exceeds the {MAX_SCORE_BY_CLAUSES}-clause "
                        "limit for one request."
                    )
            else:
                # Lucene syntax targets fields inside the query string itself;
                # `fields` merely restricts which are searchable.
                clauses = [{"type": "query_string", "query": query}]
                if fields:
                    clauses[0]["fields"] = targets

            hits = self._run_clauses(
                index,
                namespace,
                clauses,
                top_k=top_k,
                include_fields=include_fields,
                filter=effective_filter,
            )
            return {
                "mode": mode,
                "fields": targets,
                "clauses": len(clauses),
                "hits": [h.model_dump() for h in hits],
            }

        if mode == "dense":
            require_mode(caps, "dense")
            clause = self._dense_clause(
                caps, query, vector, dense_field, embed_provider, embed_model, embed_dimension
            )
            hits = self._run_clauses(
                index,
                namespace,
                [clause],
                top_k=top_k,
                include_fields=include_fields,
                filter=effective_filter,
            )
            return {"mode": "dense", "field": clause["field"], "hits": [h.model_dump() for h in hits]}

        if mode == "sparse":
            require_mode(caps, "sparse")
            clause = self._sparse_clause(
                caps, query, sparse_vector, sparse_field, sparse_provider, sparse_model
            )
            hits = self._run_clauses(
                index,
                namespace,
                [clause],
                top_k=top_k,
                include_fields=include_fields,
                filter=effective_filter,
            )
            return {
                "mode": "sparse",
                "field": clause["field"],
                "hits": [h.model_dump() for h in hits],
            }

        if mode == "hybrid":
            require_mode(caps, "hybrid")
            candidates = max(top_k * max(candidate_multiplier, 1), top_k)
            result_sets: dict[str, list[Hit]] = {}

            if caps.dense_fields and (query or vector is not None):
                clause = self._dense_clause(
                    caps, query, vector, dense_field, embed_provider, embed_model, embed_dimension
                )
                result_sets["dense"] = self._run_clauses(
                    index,
                    namespace,
                    [clause],
                    top_k=candidates,
                    include_fields=include_fields,
                    filter=effective_filter,
                )
            if caps.sparse_fields and (query or sparse_vector is not None):
                clause = self._sparse_clause(
                    caps, query, sparse_vector, sparse_field, sparse_provider, sparse_model
                )
                result_sets["sparse"] = self._run_clauses(
                    index,
                    namespace,
                    [clause],
                    top_k=candidates,
                    include_fields=include_fields,
                    filter=effective_filter,
                )
            if caps.fts_fields and query:
                targets = [f for f in (fields or caps.fts_fields) if f in caps.fts_fields]
                result_sets["text"] = self._run_clauses(
                    index,
                    namespace,
                    [{"type": "text", "field": name, "query": query} for name in targets],
                    top_k=candidates,
                    include_fields=include_fields,
                    filter=effective_filter,
                )

            fused = fuse(result_sets, strategy=fusion, weights=weights, top_k=top_k)
            return {
                "mode": "hybrid",
                "fusion": fusion,
                "signals": {k: len(v) for k, v in result_sets.items()},
                "weights": weights or {k: 1.0 for k in result_sets},
                "hits": [h.model_dump() for h in fused],
            }

        raise CapabilityError(
            f"Unknown search mode {mode!r}. Use one of: text, query_string, dense, sparse, "
            "hybrid, auto."
        )

    def _search_vectors_api(
        self,
        index: str,
        namespace: str,
        *,
        mode: str,
        query: str | None,
        vector: Sequence[float] | None,
        sparse_vector: dict[str, Any] | None,
        caps: IndexCapabilities,
        top_k: int,
        filter: dict[str, Any] | None,
        embed_provider: str | None,
        embed_model: str | None,
        embed_dimension: int | None,
        sparse_provider: str | None,
        sparse_model: str | None,
        exclude_expired: bool,
    ) -> dict[str, Any]:
        """Serve a Vectors-API index through the same search() surface.

        No client-side fusion here: the Vectors API scores a dense and a sparse
        vector together in one server-side request, which is both cheaper and
        better than fusing two result sets.
        """
        want_dense = mode in {"dense", "hybrid"} and bool(caps.dense_fields)
        want_sparse = mode in {"sparse", "hybrid"} and bool(caps.sparse_fields)
        if mode == "sparse" and not caps.sparse_fields:
            raise CapabilityError(f"Index {index!r} declares no sparse vector field.")

        if want_dense and vector is None:
            if not query:
                raise CapabilityError("A dense search needs either query text or a vector.")
            embedder = get_dense_embedder(embed_provider, embed_model, embed_dimension)
            vector = embedder.embed_query(query)
            expected = next(iter(caps.dimensions.values()), None)
            if expected and len(vector) != expected:
                raise CapabilityError(
                    f"Query vector has dimension {len(vector)} but index {index!r} expects "
                    f"{expected}. Use the embedding model this index was written with."
                )
        if want_sparse and sparse_vector is None and query:
            sparse_vector = get_sparse_embedder(sparse_provider, sparse_model).embed_query(query)

        response = self.query_vectors(
            index,
            namespace,
            vector=list(vector) if want_dense and vector is not None else None,
            sparse_vector=sparse_vector if want_sparse else None,
            top_k=top_k,
            filter=filter,
            include_metadata=True,
            exclude_expired=exclude_expired,
        )
        hits = [
            {
                "id": match.get("id"),
                "score": match.get("score", 0.0),
                "fields": match.get("metadata") or {},
                "signals": {},
            }
            for match in response.get("matches", [])
        ]
        return {
            "mode": mode,
            "api": "vectors",
            "scored_by": [s for s, on in (("dense", want_dense), ("sparse", want_sparse)) if on],
            "fusion": "server-side (single request)",
            "hits": hits,
        }

    def search_records(
        self,
        index: str,
        namespace: str,
        *,
        query: str,
        top_k: int = 10,
        filter: dict[str, Any] | None = None,
        fields: Sequence[str] | None = None,
        rerank: dict[str, Any] | None = None,
        match_terms: dict[str, Any] | None = None,
        exclude_expired: bool = True,
    ) -> dict[str, Any]:
        """Integrated-inference search: Pinecone embeds the query server-side."""
        effective_filter = ttl_mod.live_filter(filter) if exclude_expired else filter
        response = get_index(index).search(
            namespace=namespace,
            inputs={"text": query},
            top_k=top_k,
            filter=effective_filter,
            fields=list(fields) if fields else None,
            rerank=rerank,
            match_terms=match_terms,
        )
        return _to_builtins(response)

    def query_vectors(
        self,
        index: str,
        namespace: str,
        *,
        vector: Sequence[float] | None = None,
        sparse_vector: dict[str, Any] | None = None,
        id: str | None = None,
        top_k: int = 10,
        filter: dict[str, Any] | None = None,
        include_metadata: bool = True,
        include_values: bool = False,
        exclude_expired: bool = True,
    ) -> dict[str, Any]:
        """Vectors API query - the one request that carries dense + sparse together."""
        kwargs: dict[str, Any] = {
            "namespace": namespace,
            "top_k": top_k,
            "include_metadata": include_metadata,
            "include_values": include_values,
        }
        effective_filter = ttl_mod.live_filter(filter) if exclude_expired else filter
        if effective_filter:
            kwargs["filter"] = effective_filter
        if id:
            kwargs["id"] = id
        if vector is not None:
            kwargs["vector"] = list(vector)
        if sparse_vector is not None:
            kwargs["sparse_vector"] = sparse_vector
        return _to_builtins(get_index(index).query(**kwargs))

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------
    def rerank(
        self,
        *,
        query: str,
        documents: Sequence[dict[str, Any] | str],
        model: str = "bge-reranker-v2-m3",
        top_n: int | None = None,
        rank_fields: Sequence[str] | None = None,
        return_documents: bool = True,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model,
            "query": query,
            "documents": list(documents),
            "return_documents": return_documents,
        }
        if top_n:
            kwargs["top_n"] = top_n
        if rank_fields:
            kwargs["rank_fields"] = list(rank_fields)
        return _to_builtins(get_client().inference.rerank(**kwargs))

    def list_models(self, *, model_type: str | None = None) -> list[dict[str, Any]]:
        kwargs = {"type": model_type} if model_type else {}
        return _to_builtins(get_client().inference.list_models(**kwargs))

    # ------------------------------------------------------------------
    # ABC conformance (generic names delegate to the richer methods)
    # ------------------------------------------------------------------
    def upsert(self, index: str, namespace: str, records: list[dict[str, Any]]) -> dict[str, Any]:
        return self.upsert_documents(index, namespace, records)

    def fetch(
        self, index: str, namespace: str, ids: list[str] | None = None
    ) -> list[dict[str, Any]]:
        return self.fetch_documents(index, namespace, ids=ids)

    def delete(
        self,
        index: str,
        namespace: str,
        ids: list[str] | None = None,
        filter: dict[str, Any] | None = None,
        delete_all: bool = False,
    ) -> dict[str, Any]:
        return self.delete_documents(
            index, namespace, ids=ids, filter=filter, delete_all=delete_all, confirm=delete_all
        )
