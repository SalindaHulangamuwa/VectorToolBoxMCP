"""Weaviate backend.

Weaviate's model sits between Pinecone's and Chroma's: a *collection* has a
typed property schema (like a table), one or more **named vectors**, each with
its own vectorizer module and vector index, an inverted index for BM25 and
filtering, optional cross-references between collections, and optional
multi-tenancy (one isolated shard per tenant).

On top of the v4 client this module adds:

* named clients (``client.py``): local, custom, cloud and embedded Weaviate;
* one filter language shared with the other backends (``filters.py``),
  checked against the collection's property types before sending;
* collection definitions validated before creation (``schema.py``), with a
  check that the vectorizer modules exist on *this* server;
* the toolbox's embedding providers for collections without a vectorizer
  (vectorizer ``none``): documents and queries are embedded here and sent as
  vectors, so near-text and hybrid search work on self-provided vectors too;
* surfaced silent behaviour: auto-schema adding misspelt properties, objects
  written without any vector, per-object batch errors, missing tenants.
"""

from __future__ import annotations

import uuid as uuid_mod
from datetime import date, datetime
from typing import Any

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
from .filters import compile_filter, normalize
from .schema import build_collection, check_collection_name, property_types, summarize

NIL_UUID = "00000000-0000-0000-0000-000000000000"
METADATA_FIELDS = ("distance", "certainty", "score", "explain_score", "creation_time", "last_update_time", "is_consistent")
MODULE_KINDS = ("text2vec", "multi2vec", "img2vec", "ref2vec", "text2multivec", "multi2multivec",
                "generative", "reranker", "qna", "ner", "sum", "backup", "offload", "text-spellcheck")
DELETE_PAGE = 10_000


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, uuid_mod.UUID):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if hasattr(value, "latitude") and hasattr(value, "longitude"):
        return {"latitude": value.latitude, "longitude": value.longitude}
    if hasattr(value, "value") and hasattr(value, "name") and type(value).__module__.startswith("enum"):
        return value.value
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    if hasattr(value, "__dict__"):
        return {k: jsonable(v) for k, v in vars(value).items() if not k.startswith("_")}
    return str(value)


def _uuid(value: Any, at: str) -> str:
    try:
        return str(uuid_mod.UUID(str(value)))
    except (ValueError, TypeError) as exc:
        raise ToolboxError(f"{at}: {value!r} is not a UUID. Weaviate object ids are UUIDs; use "
                           "id_from=<property> to derive a stable one from your own key.") from exc


def _shape_object(obj: Any, *, with_vector: bool) -> dict[str, Any]:
    out: dict[str, Any] = {"uuid": str(obj.uuid), "properties": jsonable(obj.properties or {})}
    meta = getattr(obj, "metadata", None)
    if meta is not None:
        m = {f: jsonable(getattr(meta, f, None)) for f in METADATA_FIELDS if getattr(meta, f, None) is not None}
        if m:
            out["metadata"] = m
    if with_vector and getattr(obj, "vector", None):
        out["vector"] = jsonable(obj.vector)
    refs = getattr(obj, "references", None)
    if refs:
        out["references"] = {
            name: [{"uuid": str(o.uuid), "properties": jsonable(o.properties or {})} for o in ref.objects]
            for name, ref in refs.items()
        }
    return out


def _shape_result(res: Any, *, with_vector: bool) -> dict[str, Any]:
    groups = getattr(res, "groups", None)
    if groups is not None:
        return {
            "groups": [
                {"group": jsonable(name), "count": len(g.objects),
                 "min_distance": getattr(g, "min_distance", None), "max_distance": getattr(g, "max_distance", None),
                 "objects": [_shape_object(o, with_vector=with_vector) for o in g.objects]}
                for name, g in groups.items()
            ]
        }
    objects = [_shape_object(o, with_vector=with_vector) for o in res.objects]
    return {"count": len(objects), "objects": objects}


class WeaviateBackend(VectorStoreBackend):
    name = "weaviate"

    # -- plumbing ----------------------------------------------------------
    @staticmethod
    def _entry(client: str | None):
        return client_mod.get_entry(client)

    def _raw_collection(self, client: str | None, name: str):
        entry = self._entry(client)
        if not entry.client.collections.exists(name):
            names = sorted(entry.client.collections.list_all(simple=True))
            hint = ""
            match = [n for n in names if n.lower() == name.lower()]
            if match:
                hint = f" Names are case-sensitive - did you mean {match[0]!r}?"
            raise NotFound(
                f"Collection {name!r} does not exist on client {entry.name!r} ({entry.kind}). "
                f"Existing: {', '.join(names[:30]) or 'none'}.{hint}"
            )
        return entry.client.collections.get(name)

    def _config(self, col: Any) -> dict[str, Any]:
        return col.config.get().to_dict()

    def _collection(self, client: str | None, name: str, tenant: str | None = None,
                    *, need_tenant: bool = True) -> tuple[Any, dict[str, Any]]:
        """Collection handle (tenant-scoped when needed) and its REST config."""
        col = self._raw_collection(client, name)
        config = self._config(col)
        mt = (config.get("multiTenancyConfig") or {}).get("enabled")
        if tenant and not mt:
            raise CapabilityError(f"Collection {name!r} is not multi-tenant; drop tenant={tenant!r}.")
        if mt and need_tenant and not tenant:
            tenants = sorted(col.tenants.get())[:10]
            raise CapabilityError(
                f"Collection {name!r} is multi-tenant: every data operation needs tenant=<name>. "
                f"Tenants (first 10): {', '.join(tenants) or 'none yet - create one with weaviate_create_tenants'}."
            )
        if tenant:
            col = col.with_tenant(tenant)
        return col, config

    @staticmethod
    def _enabled_modules(entry) -> set[str]:
        try:
            return set((entry.client.get_meta().get("modules") or {}).keys())
        except Exception:  # pragma: no cover
            return set()

    @staticmethod
    def _vectors(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return summarize(config)["vectors"]

    def _query_vector(
        self, config: dict[str, Any], target: str | list[str] | None, text: str,
        embed_provider: str | None, embed_model: str | None, embed_dimension: int | None,
    ) -> tuple[list[float] | None, str]:
        """Embed ``text`` here when the target vector has no vectorizer."""
        vectors = self._vectors(config)
        names = [target] if isinstance(target, str) else (target or list(vectors))
        if not names:
            raise CapabilityError("This collection has no vector to search.")
        unknown = [n for n in names if n not in vectors]
        if unknown:
            raise CapabilityError(f"Unknown target vector(s) {unknown}. Vectors: {', '.join(vectors)}.")
        self_provided = [n for n in names if vectors[n]["vectorizer"] == "none"]
        if not self_provided and not embed_provider:
            return None, "weaviate:" + "/".join(vectors[n]["vectorizer"] for n in names)
        if len(names) > 1:
            raise CapabilityError(
                "Embedding a text query here works for one target vector at a time; pass target_vector "
                "or a vector yourself."
            )
        from ...embeddings import get_dense_embedder

        embedder = get_dense_embedder(embed_provider, embed_model, embed_dimension)
        return embedder.embed_query(text), f"toolbox:{embedder.provider}/{embedder.model}"

    @staticmethod
    def _metadata(fields: list[str] | None, default: list[str]):
        from weaviate.classes.query import MetadataQuery

        chosen = default if fields is None else fields
        bad = [f for f in chosen if f not in METADATA_FIELDS]
        if bad:
            raise ToolboxError(f"return_metadata has unknown field(s) {bad}. Allowed: {', '.join(METADATA_FIELDS)}.")
        return MetadataQuery(**{f: True for f in chosen}) if chosen else None

    @staticmethod
    def _return_props(config: dict[str, Any], requested: list[str] | None) -> list[str] | None:
        if requested is None:
            return None
        known = set(property_types(config))
        unknown = [p for p in requested if p not in known]
        if unknown:
            raise ToolboxError(f"return_properties {unknown} are not properties. Properties: {', '.join(sorted(known))}.")
        return requested

    @staticmethod
    def _refs(config: dict[str, Any], requested: list[str] | None):
        if not requested:
            return None
        from weaviate.classes.query import QueryReference

        refs = {r["name"] for r in summarize(config)["references"]}
        unknown = [r for r in requested if r not in refs]
        if unknown:
            raise ToolboxError(f"return_references {unknown} are not reference properties. References: {sorted(refs)}.")
        return [QueryReference(link_on=r) for r in requested]

    @staticmethod
    def _group_by(spec: dict[str, Any] | None):
        if not spec:
            return None
        from weaviate.classes.query import GroupBy

        if "property" not in spec:
            raise ToolboxError("group_by needs {'property': ..., 'objects_per_group': n, 'number_of_groups': n}.")
        return GroupBy(prop=spec["property"], objects_per_group=int(spec.get("objects_per_group", 3)),
                       number_of_groups=int(spec.get("number_of_groups", 10)))

    @staticmethod
    def _rerank(spec: dict[str, Any] | None):
        if not spec:
            return None
        from weaviate.classes.query import Rerank

        if "property" not in spec:
            raise ToolboxError("rerank needs {'property': <text property>, 'query': <optional text>}.")
        return Rerank(prop=spec["property"], query=spec.get("query"))

    @staticmethod
    def _target_vector(target: str | list[str] | None, combination: str | None, weights: dict[str, float] | None):
        if not target or isinstance(target, str):
            return target
        from weaviate.classes.query import TargetVectors

        combination = combination or "average"
        if combination == "manual_weights" or weights:
            if not weights:
                raise ToolboxError("target_vector_combination='manual_weights' needs target_vector_weights.")
            return TargetVectors.manual_weights(weights)
        if combination not in {"average", "minimum", "sum", "relative_score"}:
            raise ToolboxError("target_vector_combination must be average, minimum, sum, relative_score or manual_weights.")
        fn = getattr(TargetVectors, combination)
        return fn(target) if combination != "relative_score" else fn({t: 1.0 for t in target})

    # ======================================================================
    # Server
    # ======================================================================
    def heartbeat(self, client: str | None) -> dict[str, Any]:
        entry = self._entry(client)
        c = entry.client
        meta = c.get_meta()
        out = entry.describe()
        out.update(ready=c.is_ready(), live=c.is_live(), server_version=meta.get("version"),
                   hostname=meta.get("hostname"), module_count=len(meta.get("modules") or {}))
        try:
            nodes = c.cluster.nodes()
            out["nodes"] = [{"name": n.name, "status": n.status, "version": n.version} for n in nodes]
        except Exception:
            pass
        return out

    def list_modules(self, client: str | None) -> dict[str, Any]:
        entry = self._entry(client)
        modules = entry.client.get_meta().get("modules") or {}
        grouped: dict[str, list[str]] = {}
        for name in sorted(modules):
            kind = next((k for k in MODULE_KINDS if name.startswith(k + "-") or name == k), "other")
            grouped.setdefault(kind, []).append(name)
        return {
            "client": entry.name,
            "modules": grouped,
            "self_provided_vectors": "always available: vectorizer 'none' (use embed_provider to embed here)",
            "forwarded_provider_headers": entry.params.get("provider_headers", []),
        }

    # ======================================================================
    # Collections
    # ======================================================================
    def list_collections(self, client: str | None, include_counts: bool = False) -> dict[str, Any]:
        entry = self._entry(client)
        configs = entry.client.collections.list_all(simple=False)
        out = []
        for name, cfg in sorted(configs.items()):
            s = summarize(cfg.to_dict())
            item = {"name": name, "description": s["description"],
                    "vectors": {k: v["vectorizer"] for k, v in s["vectors"].items()},
                    "properties": len(s["properties"]), "multi_tenancy": s["multi_tenancy"]["enabled"]}
            if include_counts:
                if s["multi_tenancy"]["enabled"]:
                    item["count"] = "per tenant - use weaviate_aggregate with tenant"
                else:
                    item["count"] = entry.client.collections.get(name).aggregate.over_all(total_count=True).total_count
            out.append(item)
        return {"client": entry.name, "count": len(out), "collections": out}

    def get_config(self, client: str | None, name: str | None = None, raw: bool = False) -> dict[str, Any]:
        entry = self._entry(client)
        if name is None:
            configs = entry.client.collections.list_all(simple=False)
            return {"client": entry.name, "collections": [
                (cfg.to_dict() if raw else summarize(cfg.to_dict())) for _, cfg in sorted(configs.items())
            ]}
        config = self._config(self._raw_collection(client, name))
        return config if raw else summarize(config)

    def create_collection(self, client: str | None, name: str, *, raw_config: dict[str, Any] | None = None,
                          if_not_exists: bool = False, **spec: Any) -> dict[str, Any]:
        entry = self._entry(client)
        if raw_config:
            if any(v for v in spec.values()):
                raise ConfigurationError("Pass raw_config alone, or the high-level arguments - not both.")
            cls = dict(raw_config)
            cls.setdefault("class", name)
            if cls["class"] != name:
                raise ConfigurationError(f"raw_config['class'] is {cls['class']!r} but name is {name!r}.")
            check_collection_name(name)
        else:
            cls = build_collection(name, enabled_modules=self._enabled_modules(entry), **spec)
        if entry.client.collections.exists(name):
            if if_not_exists:
                return {"created": False, "note": "Collection already exists; nothing changed.",
                        **summarize(self._config(entry.client.collections.get(name)))}
            raise ConfigurationError(f"Collection {name!r} already exists. Pass if_not_exists=true to reuse it.")
        col = entry.client.collections.create_from_dict(cls)
        return {"created": True, **summarize(self._config(col))}

    def update_collection(
        self, client: str | None, name: str, *,
        description: str | None = None,
        property_descriptions: dict[str, str] | None = None,
        bm25: dict[str, float] | None = None,
        stopwords_preset: str | None = None,
        stopwords_additions: list[str] | None = None,
        stopwords_removals: list[str] | None = None,
        auto_tenant_creation: bool | None = None,
        auto_tenant_activation: bool | None = None,
        replication_factor: int | None = None,
        async_replication: bool | None = None,
        vector_index: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        from weaviate.classes.config import Reconfigure

        col = self._raw_collection(client, name)
        before = summarize(self._config(col))
        kwargs: dict[str, Any] = {}
        if description is not None:
            kwargs["description"] = description
        if property_descriptions:
            known = {p["name"] for p in before["properties"]}
            unknown = [p for p in property_descriptions if p not in known]
            if unknown:
                raise ToolboxError(f"property_descriptions names unknown properties {unknown}.")
            kwargs["property_descriptions"] = property_descriptions
        if bm25 or stopwords_preset or stopwords_additions or stopwords_removals:
            kwargs["inverted_index_config"] = Reconfigure.inverted_index(
                bm25_b=(bm25 or {}).get("b"), bm25_k1=(bm25 or {}).get("k1"),
                stopwords_preset=stopwords_preset, stopwords_additions=stopwords_additions,
                stopwords_removals=stopwords_removals,
            )
        if auto_tenant_creation is not None or auto_tenant_activation is not None:
            if not before["multi_tenancy"]["enabled"]:
                raise CapabilityError("Multi-tenancy can only be switched on at creation; this collection has it off.")
            kwargs["multi_tenancy_config"] = Reconfigure.multi_tenancy(
                auto_tenant_creation=auto_tenant_creation, auto_tenant_activation=auto_tenant_activation)
        if replication_factor is not None or async_replication is not None:
            kwargs["replication_config"] = Reconfigure.replication(
                factor=replication_factor, async_enabled=async_replication)
        if vector_index:
            updates = []
            for vname, settings in vector_index.items():
                if vname not in before["vectors"]:
                    raise ToolboxError(f"Unknown vector {vname!r}. Vectors: {', '.join(before['vectors'])}.")
                index_type = before["vectors"][vname]["index_type"]
                updates.append((vname, self._index_update(index_type, settings, vname)))
            if any(v.get("legacy") for v in before["vectors"].values()):
                kwargs["vector_index_config"] = updates[0][1]
            else:
                kwargs["vector_config"] = [Reconfigure.Vectors.update(name=n, vector_index_config=c) for n, c in updates]
        if not kwargs:
            raise ToolboxError("Nothing to change - pass at least one setting.")
        col.config.update(**kwargs)
        after = summarize(self._config(col))
        changed = {k: {"before": before[k], "after": after[k]} for k in after if before.get(k) != after.get(k)}
        return {"collection": name, "changed": changed or "no visible change (value already set)"}

    @staticmethod
    def _index_update(index_type: str, settings: dict[str, Any], vname: str):
        from weaviate.classes.config import Reconfigure

        settings = dict(settings)
        quantizer = settings.pop("quantizer", None)
        q = None
        if quantizer:
            qname, qopts = (quantizer, {}) if isinstance(quantizer, str) else (quantizer.get("type"), {
                k: v for k, v in quantizer.items() if k != "type"})
            if qname not in {"pq", "bq", "sq", "rq"}:
                raise ToolboxError(f"vector_index[{vname}].quantizer must be pq, bq, sq or rq.")
            q = getattr(Reconfigure.VectorIndex.Quantizer, qname)(**qopts)
        allowed = {
            "hnsw": {"ef", "dynamic_ef_min", "dynamic_ef_max", "dynamic_ef_factor", "flat_search_cutoff",
                     "filter_strategy", "vector_cache_max_objects"},
            "flat": {"vector_cache_max_objects"},
            "dynamic": {"threshold"},
        }.get(index_type)
        if allowed is None:
            raise CapabilityError(f"Vector {vname!r} uses a {index_type} index, which cannot be tuned here.")
        unknown = set(settings) - allowed
        if unknown:
            raise CapabilityError(
                f"vector_index[{vname}] ({index_type}): {sorted(unknown)} cannot change after creation. "
                f"Mutable: {', '.join(sorted(allowed))}, quantizer."
            )
        if "filter_strategy" in settings:
            from weaviate.classes.config import VectorFilterStrategy

            settings["filter_strategy"] = VectorFilterStrategy(settings["filter_strategy"])
        fn = getattr(Reconfigure.VectorIndex, index_type)
        return fn(quantizer=q, **settings) if q is not None else fn(**settings)

    def add_to_collection(self, client: str | None, name: str, *, property: dict[str, Any] | None = None,
                          reference: dict[str, Any] | None = None, vector: dict[str, Any] | None = None) -> dict[str, Any]:
        from weaviate.classes.config import (
            Configure,
            DataType,
            Property,
            ReferenceProperty,
            Tokenization,
        )

        if sum(x is not None for x in (property, reference, vector)) != 1:
            raise ToolboxError("Pass exactly one of property, reference or vector.")
        col = self._raw_collection(client, name)
        config = self._config(col)
        existing = set(property_types(config))
        if property is not None:
            spec = build_collection("Tmp", properties=[property], vectors=[{"vectorizer": "none"}])["properties"][0]
            if spec["name"] in existing:
                raise ToolboxError(f"Property {spec['name']!r} already exists.")

            def to_prop(p: dict[str, Any]) -> Property:
                return Property(
                    name=p["name"], data_type=DataType(p["dataType"][0]), description=p.get("description"),
                    index_filterable=p.get("indexFilterable"), index_searchable=p.get("indexSearchable"),
                    index_range_filters=p.get("indexRangeFilters"),
                    tokenization=Tokenization(p["tokenization"]) if p.get("tokenization") else None,
                    nested_properties=[to_prop(n) for n in p["nestedProperties"]] if p.get("nestedProperties") else None,
                )

            col.config.add_property(to_prop(spec))
            return {"collection": name, "added_property": spec,
                    "note": "Existing objects have no value for it; vectors are not recomputed."}
        if reference is not None:
            rname, target = reference.get("name"), reference.get("target")
            if not rname or not target:
                raise ToolboxError("reference needs {'name': ..., 'target': 'OtherCollection'}.")
            if rname in existing:
                raise ToolboxError(f"{rname!r} already exists on {name}.")
            self._raw_collection(client, target)  # target must exist
            col.config.add_reference(ReferenceProperty(name=rname, target_collection=target,
                                                       description=reference.get("description")))
            return {"collection": name, "added_reference": {"name": rname, "target": target}}
        vname = vector.get("name")
        if not vname:
            raise ToolboxError("vector needs a name.")
        if vname in self._vectors(config):
            raise ToolboxError(f"Vector {vname!r} already exists.")
        if (vector.get("vectorizer") or "none") != "none":
            raise CapabilityError(
                "Adding a vectorized named vector after creation needs that module's own settings; this tool "
                "adds self-provided vectors (vectorizer 'none'). Existing objects are not back-filled either way."
            )
        col.config.add_vector(vector_config=Configure.Vectors.self_provided(name=vname))
        return {"collection": name, "added_vector": vname,
                "note": "Existing objects have no value for this vector until you update them."}

    def delete_collection(self, client: str | None, name: str, confirm: bool = False) -> dict[str, Any]:
        entry = self._entry(client)
        col = self._raw_collection(client, name)
        s = summarize(self._config(col))
        if s["multi_tenancy"]["enabled"]:
            size = f"{len(col.tenants.get())} tenant(s) and all their objects"
        else:
            size = f"{col.aggregate.over_all(total_count=True).total_count} object(s)"
        if not confirm:
            raise ConfirmationRequired(
                f"Deleting collection {name!r} removes {size} on client {entry.name!r} and cannot be undone. "
                "Call again with confirm=true."
            )
        entry.client.collections.delete(name)
        return {"deleted": name, "removed": size}

    def capabilities(self, index: str, client: str | None = None) -> IndexCapabilities:
        col = self._raw_collection(client, index)
        config = self._config(col)
        s = summarize(config)
        searchable = [p["name"] for p in s["properties"] if p.get("searchable")]
        filterable = [p["name"] for p in s["properties"] if p.get("filterable")]
        modes = []
        if s["vectors"]:
            modes += ["near_vector", "near_object"]
            modes.append("near_text")  # natively, or via toolbox embedding for vectorizer 'none'
        if searchable:
            modes.append("bm25")
        if s["vectors"] and searchable:
            modes.append("hybrid")
        notes = []
        for vname, v in s["vectors"].items():
            how = "Weaviate embeds text with " + v["vectorizer"] if v["vectorizer"] != "none" else \
                "self-provided: pass vectors, or embed_provider so this server embeds text"
            notes.append(f"vector {vname!r}: {how}; {v['index_type']} index, {v['distance']} distance"
                         + (f", {v['quantizer']} quantization" if v.get("quantizer") else ""))
        if s["multi_tenancy"]["enabled"]:
            notes.append("multi-tenant: every data operation needs tenant=<name>")
        range_props = [p["name"] for p in s["properties"] if p.get("range_filters")]
        if range_props:
            notes.append(f"range-filter index on: {', '.join(range_props)}")
        return IndexCapabilities(
            index=index, api="collection",
            dense_fields=list(s["vectors"]), fts_fields=searchable, filterable_fields=filterable,
            metrics={k: v["distance"] for k, v in s["vectors"].items() if v.get("distance")},
            supported_search_modes=modes, notes=notes,
        )

    # ======================================================================
    # Tenants
    # ======================================================================
    def list_tenants(self, client: str | None, collection: str) -> dict[str, Any]:
        col = self._raw_collection(client, collection)
        if not summarize(self._config(col))["multi_tenancy"]["enabled"]:
            raise CapabilityError(f"Collection {collection!r} is not multi-tenant.")
        tenants = col.tenants.get()
        rows = sorted(({"name": n, "activity_status": jsonable(t.activity_status)} for n, t in tenants.items()),
                      key=lambda r: r["name"])
        counts: dict[str, int] = {}
        for r in rows:
            counts[r["activity_status"]] = counts.get(r["activity_status"], 0) + 1
        return {"collection": collection, "count": len(rows), "by_status": counts, "tenants": rows}

    def create_tenants(self, client: str | None, collection: str, tenants: list[Any]) -> dict[str, Any]:
        from weaviate.classes.tenants import Tenant, TenantCreateActivityStatus

        col = self._raw_collection(client, collection)
        if not summarize(self._config(col))["multi_tenancy"]["enabled"]:
            raise CapabilityError(f"Collection {collection!r} is not multi-tenant; tenants cannot be added.")
        existing = set(col.tenants.get())
        objs, skipped = [], []
        for i, t in enumerate(tenants):
            name, status = (t, None) if isinstance(t, str) else (t.get("name"), t.get("activity_status"))
            if not name:
                raise ToolboxError(f"tenants[{i}] needs a name.")
            if name in existing:
                skipped.append(name)
                continue
            if status and status not in {"ACTIVE", "INACTIVE"}:
                raise ToolboxError(f"tenants[{i}]: a new tenant starts ACTIVE or INACTIVE.")
            objs.append(Tenant(name=name, activity_status=TenantCreateActivityStatus(status)) if status else Tenant(name=name))
        if objs:
            col.tenants.create(objs)
        return {"collection": collection, "created": [t.name for t in objs], "skipped_existing": skipped}

    def update_tenants(self, client: str | None, collection: str, tenants: list[dict[str, Any]]) -> dict[str, Any]:
        from weaviate.classes.tenants import Tenant, TenantUpdateActivityStatus

        col = self._raw_collection(client, collection)
        existing = col.tenants.get()
        objs = []
        for i, t in enumerate(tenants):
            name, status = t.get("name"), t.get("activity_status")
            if name not in existing:
                raise NotFound(f"tenants[{i}]: no tenant {name!r} in {collection}.")
            if status not in {"ACTIVE", "INACTIVE", "OFFLOADED"}:
                raise ToolboxError(
                    f"tenants[{i}].activity_status must be ACTIVE, INACTIVE or OFFLOADED "
                    "(OFFLOADED needs an offload module such as offload-s3)."
                )
            objs.append(Tenant(name=name, activity_status=TenantUpdateActivityStatus(status)))
        col.tenants.update(objs)
        after = col.tenants.get()
        return {"collection": collection,
                "tenants": [{"name": t.name, "activity_status": jsonable(after[t.name].activity_status)} for t in objs]}

    def delete_tenants(self, client: str | None, collection: str, tenants: list[str], confirm: bool = False) -> dict[str, Any]:
        col = self._raw_collection(client, collection)
        existing = col.tenants.get()
        missing = [t for t in tenants if t not in existing]
        if missing:
            raise NotFound(f"No such tenant(s): {missing}.")
        if not confirm:
            raise ConfirmationRequired(
                f"Deleting tenant(s) {tenants} removes all of their objects and cannot be undone. "
                "Call again with confirm=true (or set them INACTIVE / OFFLOADED to keep the data)."
            )
        col.tenants.remove(tenants)
        return {"collection": collection, "deleted_tenants": tenants}

    # ======================================================================
    # Objects
    # ======================================================================
    def _prepare_objects(
        self, config: dict[str, Any], objects: list[dict[str, Any]], *,
        id_from: str | None, embed_source: str | None, embed_vectors: list[str] | None,
        embed_provider: str | None, embed_model: str | None, embed_dimension: int | None,
        allow_new_properties: bool,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        from weaviate.util import generate_uuid5

        types = property_types(config)
        refs = {r["name"] for r in summarize(config)["references"]}
        vectors = self._vectors(config)
        prepared, new_props = [], set()
        for i, obj in enumerate(objects):
            if not isinstance(obj, dict):
                raise ToolboxError(f"objects[{i}] must be an object like {{'properties': {{...}}}}.")
            unknown_keys = set(obj) - {"uuid", "id", "properties", "vector", "vectors", "references"}
            if unknown_keys:
                raise ToolboxError(
                    f"objects[{i}] has unknown keys {sorted(unknown_keys)}. Put fields under 'properties'; "
                    "allowed top-level keys: uuid, properties, vector(s), references."
                )
            props = dict(obj.get("properties") or {})
            for key in props:
                if key not in types and key not in refs:
                    new_props.add(key)
            uid = obj.get("uuid") or obj.get("id")
            if uid is None and id_from:
                if id_from not in props:
                    raise ToolboxError(f"objects[{i}] has no {id_from!r} property to derive its id from.")
                uid = generate_uuid5(f"{config['class']}:{props[id_from]}")
            item: dict[str, Any] = {"properties": props, "uuid": _uuid(uid, f"objects[{i}].uuid") if uid else None}
            vec = obj.get("vectors", obj.get("vector"))
            if vec is not None:
                if isinstance(vec, dict):
                    bad = [k for k in vec if k not in vectors]
                    if bad:
                        raise ToolboxError(f"objects[{i}]: unknown vector name(s) {bad}. Vectors: {', '.join(vectors)}.")
                elif len(vectors) > 1:
                    raise ToolboxError(
                        f"objects[{i}]: this collection has named vectors {list(vectors)}; give "
                        "{'vector_name': [...]} instead of a bare list."
                    )
                elif vectors:
                    vec = {next(iter(vectors)): vec}
                item["vector"] = vec
            ref_values = obj.get("references")
            if ref_values:
                bad = [k for k in ref_values if k not in refs]
                if bad:
                    raise ToolboxError(f"objects[{i}]: {bad} are not reference properties. References: {sorted(refs)}.")
                item["references"] = {k: ([_uuid(u, f"objects[{i}].references.{k}") for u in v]
                                          if isinstance(v, list) else _uuid(v, f"objects[{i}].references.{k}"))
                                      for k, v in ref_values.items()}
            prepared.append(item)

        if new_props and not allow_new_properties:
            raise ToolboxError(
                f"Properties {sorted(new_props)} are not in {config['class']}'s schema. Weaviate's auto-schema "
                "would silently add them (often a typo like 'titel'). Fix the names, add them with "
                f"weaviate_add_to_collection, or pass allow_new_properties=true. Known: {', '.join(sorted(types))}."
            )

        report: dict[str, Any] = {}
        self_provided = [n for n, v in vectors.items() if v["vectorizer"] == "none"]
        targets = embed_vectors or (self_provided if embed_source else [])
        if targets:
            bad = [t for t in targets if t not in vectors]
            if bad:
                raise ToolboxError(f"embed_vectors {bad} are not vectors of this collection.")
            if not embed_source:
                raise ToolboxError("embed_vectors needs embed_source=<text property to embed>.")
            todo = [(k, p) for k, p in enumerate(prepared)
                    if any(t not in (p.get("vector") or {}) for t in targets)]
            missing_text = [k for k, p in todo if not isinstance(p["properties"].get(embed_source), str)]
            if missing_text:
                raise ToolboxError(f"objects {missing_text[:10]} have no text in {embed_source!r} to embed.")
            if todo:
                from ...embeddings import get_dense_embedder

                embedder = get_dense_embedder(embed_provider, embed_model, embed_dimension)
                vecs = embedder.embed_documents([p["properties"][embed_source] for _, p in todo])
                for (_k, p), v in zip(todo, vecs):
                    p.setdefault("vector", {})
                    for t in targets:
                        p["vector"].setdefault(t, v)
                report["embedded_by"] = f"toolbox:{embedder.provider}/{embedder.model}"
                report["embedded"] = len(todo)
        unvectored = [k for k, p in enumerate(prepared)
                      if any(n not in (p.get("vector") or {}) for n in self_provided)]
        if unvectored:
            report["warning"] = (
                f"{len(unvectored)} object(s) have no value for self-provided vector(s) {self_provided} and "
                "will not be found by vector search. Pass vectors, or embed_source=<text property>."
            )
        return prepared, report

    def insert_objects(self, client: str | None, collection: str, objects: list[dict[str, Any]], *,
                       tenant: str | None = None, upsert: bool = False, merge: bool = False,
                       id_from: str | None = None, embed_source: str | None = None,
                       embed_vectors: list[str] | None = None, embed_provider: str | None = None,
                       embed_model: str | None = None, embed_dimension: int | None = None,
                       allow_new_properties: bool = False) -> dict[str, Any]:
        from weaviate.classes.data import DataObject
        from weaviate.classes.query import Filter

        if not objects:
            raise ToolboxError("No objects given.")
        col, config = self._collection(client, collection, tenant)
        prepared, report = self._prepare_objects(
            config, objects, id_from=id_from, embed_source=embed_source, embed_vectors=embed_vectors,
            embed_provider=embed_provider, embed_model=embed_model, embed_dimension=embed_dimension,
            allow_new_properties=allow_new_properties,
        )
        ids = [p["uuid"] for p in prepared if p["uuid"]]
        dupes = sorted({u for u in ids if ids.count(u) > 1})
        if dupes:
            raise ToolboxError(f"Duplicate uuids in one request: {dupes[:5]}.")

        existing: set[str] = set()
        if ids:
            for start in range(0, len(ids), 1000):
                chunk = ids[start:start + 1000]
                res = col.query.fetch_objects(filters=Filter.by_id().contains_any(chunk), limit=len(chunk),
                                              return_properties=[])
                existing.update(str(o.uuid) for o in res.objects)

        result: dict[str, Any] = {"collection": collection, **({"tenant": tenant} if tenant else {})}
        to_insert = [p for p in prepared if not (p["uuid"] and p["uuid"] in existing)]
        to_write = [p for p in prepared if p["uuid"] and p["uuid"] in existing]
        if to_write and not upsert:
            result["skipped_existing"] = [p["uuid"] for p in to_write][:50]
            to_write = []

        errors: list[dict[str, Any]] = []
        inserted: list[str] = []
        for start in range(0, len(to_insert), 1000):
            batch = to_insert[start:start + 1000]
            ret = col.data.insert_many([
                DataObject(properties=p["properties"], uuid=p["uuid"], vector=p.get("vector"),
                           references=p.get("references")) for p in batch
            ])
            for uid in ret.uuids.values():
                inserted.append(str(uid))
            for k, err in ret.errors.items():
                errors.append({"index": start + k, "uuid": batch[k]["uuid"], "message": err.message})
        updated = 0
        for p in to_write:
            try:
                if merge:
                    col.data.update(uuid=p["uuid"], properties=p["properties"], vector=p.get("vector"),
                                    references=p.get("references"))
                else:
                    col.data.replace(uuid=p["uuid"], properties=p["properties"], vector=p.get("vector"),
                                     references=p.get("references"))
                updated += 1
            except Exception as exc:
                errors.append({"uuid": p["uuid"], "message": str(exc)[:300]})
        result.update(inserted=len(inserted), uuids=inserted[:100], **report)
        if upsert:
            result["updated" if merge else "replaced"] = updated
        if errors:
            result["errors"] = errors[:50]
            result["error_count"] = len(errors)
        return result

    def update_object(self, client: str | None, collection: str, uuid: str, *, properties: dict[str, Any] | None = None,
                      vector: Any = None, references: dict[str, Any] | None = None, replace: bool = False,
                      tenant: str | None = None, allow_new_properties: bool = False,
                      embed_source: str | None = None, embed_provider: str | None = None,
                      embed_model: str | None = None) -> dict[str, Any]:
        col, config = self._collection(client, collection, tenant)
        uid = _uuid(uuid, "uuid")
        if not col.data.exists(uid):
            raise NotFound(f"No object {uid} in {collection}{f' (tenant {tenant})' if tenant else ''}.")
        obj = {"properties": properties or {}, **({"vector": vector} if vector is not None else {}),
               **({"references": references} if references else {})}
        [p], report = self._prepare_objects(
            config, [obj], id_from=None, embed_source=embed_source, embed_vectors=None,
            embed_provider=embed_provider, embed_model=embed_model, embed_dimension=None,
            allow_new_properties=allow_new_properties)
        report.pop("warning", None)  # partial updates keep the existing vectors
        fn = col.data.replace if replace else col.data.update
        fn(uuid=uid, properties=p["properties"], vector=p.get("vector"), references=p.get("references"))
        return {"collection": collection, "uuid": uid, "mode": "replaced" if replace else "merged", **report}

    def delete_objects(self, client: str | None, collection: str, *, ids: list[str] | None = None,
                       filters: dict[str, Any] | None = None, delete_all: bool = False, dry_run: bool = False,
                       confirm: bool = False, tenant: str | None = None) -> dict[str, Any]:
        from weaviate.classes.query import Filter

        col, config = self._collection(client, collection, tenant)
        parts = []
        if ids:
            parts.append(Filter.by_id().contains_any([_uuid(u, "ids") for u in ids]))
        if filters:
            parts.append(compile_filter(filters, property_types(config)))
        if delete_all:
            if parts:
                raise ToolboxError("delete_all deletes everything; do not combine it with ids or filters.")
            parts.append(Filter.by_id().not_equal(NIL_UUID))
        if not parts:
            raise ToolboxError("Say what to delete: ids, filters, or delete_all=true with confirm=true.")
        where = parts[0] if len(parts) == 1 else Filter.all_of(parts)

        preview = col.data.delete_many(where=where, dry_run=True)
        if dry_run:
            return {"collection": collection, "dry_run": True, "would_delete": preview.matches}
        if delete_all and not confirm:
            raise ConfirmationRequired(
                f"delete_all removes all {preview.matches} object(s) from {collection}"
                f"{f' tenant {tenant}' if tenant else ''}. Call again with confirm=true (the collection is kept)."
            )
        deleted, failed = 0, 0
        while True:
            res = col.data.delete_many(where=where)
            deleted += res.successful
            failed += res.failed
            if res.matches < DELETE_PAGE or res.successful == 0:
                break
        out = {"collection": collection, "deleted": deleted}
        if failed:
            out["failed"] = failed
        return out

    def add_references(self, client: str | None, collection: str, references: list[dict[str, Any]],
                       tenant: str | None = None) -> dict[str, Any]:
        from weaviate.classes.data import DataReference

        col, config = self._collection(client, collection, tenant)
        refs = {r["name"] for r in summarize(config)["references"]}
        items = []
        for i, r in enumerate(references):
            prop = r.get("from_property")
            if prop not in refs:
                raise ToolboxError(f"references[{i}].from_property {prop!r} is not a reference. References: {sorted(refs)}.")
            to = r.get("to")
            to = [_uuid(u, f"references[{i}].to") for u in to] if isinstance(to, list) else _uuid(to, f"references[{i}].to")
            items.append(DataReference(from_property=prop, from_uuid=_uuid(r.get("from_uuid"), f"references[{i}].from_uuid"),
                                       to_uuid=to))
        ret = col.data.reference_add_many(items)
        errors = [{"index": k, "message": e.message} for k, e in (ret.errors or {}).items()]
        return {"collection": collection, "added": len(items) - len(errors), **({"errors": errors} if errors else {})}

    # ======================================================================
    # Reading and search
    # ======================================================================
    def fetch_objects(self, client: str | None, collection: str, *, ids: list[str] | None = None,
                      filters: dict[str, Any] | None = None, limit: int = 25, offset: int | None = None,
                      after: str | None = None, sort: list[dict[str, Any]] | None = None,
                      return_properties: list[str] | None = None, return_references: list[str] | None = None,
                      return_metadata: list[str] | None = None, include_vector: bool = False,
                      tenant: str | None = None) -> dict[str, Any]:
        from weaviate.classes.query import Filter, Sort

        col, config = self._collection(client, collection, tenant)
        parts = []
        if ids:
            parts.append(Filter.by_id().contains_any([_uuid(u, "ids") for u in ids]))
        if filters:
            parts.append(compile_filter(filters, property_types(config)))
        where = None if not parts else parts[0] if len(parts) == 1 else Filter.all_of(parts)
        sorting = None
        if sort:
            if after:
                raise ToolboxError("after (cursor paging) cannot be combined with sort.")
            for i, spec in enumerate(sort):
                prop = spec.get("property")
                asc = bool(spec.get("ascending", True))
                if prop in ("_creationTimeUnix", "creation_time"):
                    method, args = "by_creation_time", {"ascending": asc}
                elif prop in ("_lastUpdateTimeUnix", "update_time"):
                    method, args = "by_update_time", {"ascending": asc}
                elif prop in property_types(config):
                    method, args = "by_property", {"name": prop, "ascending": asc}
                else:
                    raise ToolboxError(f"sort[{i}]: unknown property {prop!r}.")
                sorting = getattr(Sort if sorting is None else sorting, method)(**args)
        if after and (where is not None or offset):
            raise ToolboxError("after (cursor paging) works on the whole collection only - no filters or offset.")
        res = col.query.fetch_objects(
            limit=limit, offset=offset, after=_uuid(after, "after") if after else None, filters=where,
            sort=sorting, include_vector=include_vector,
            return_metadata=self._metadata(return_metadata, ["creation_time", "last_update_time"]),
            return_properties=self._return_props(config, return_properties),
            return_references=self._refs(config, return_references),
        )
        out = {"collection": collection, **_shape_result(res, with_vector=include_vector)}
        if res.objects and len(res.objects) == limit and not where and not sorting and not offset:
            out["next_after"] = str(res.objects[-1].uuid)
        return out

    def _common(self, config, filters, return_properties, return_references, return_metadata,
                default_meta, group_by, rerank, include_vector) -> dict[str, Any]:
        return {
            "filters": compile_filter(filters, property_types(config)) if filters else None,
            "return_properties": self._return_props(config, return_properties),
            "return_references": self._refs(config, return_references),
            "return_metadata": self._metadata(return_metadata, default_meta),
            "group_by": self._group_by(group_by),
            "rerank": self._rerank(rerank),
            "include_vector": include_vector,
        }

    def hybrid(self, client: str | None, collection: str, query: str, *, alpha: float = 0.75,
               vector: list[float] | dict[str, list[float]] | None = None, query_properties: list[str] | None = None,
               fusion_type: str = "relative_score", max_vector_distance: float | None = None,
               bm25_operator: str | None = None, minimum_match: int | None = None,
               target_vector: str | list[str] | None = None, target_vector_combination: str | None = None,
               target_vector_weights: dict[str, float] | None = None,
               filters: dict[str, Any] | None = None, limit: int = 10, offset: int | None = None,
               auto_limit: int | None = None, group_by: dict[str, Any] | None = None,
               rerank: dict[str, Any] | None = None, return_properties: list[str] | None = None,
               return_references: list[str] | None = None, return_metadata: list[str] | None = None,
               include_vector: bool = False, tenant: str | None = None, embed_provider: str | None = None,
               embed_model: str | None = None, embed_dimension: int | None = None) -> dict[str, Any]:
        from weaviate.classes.query import BM25Operator, HybridFusion

        if not 0.0 <= alpha <= 1.0:
            raise ToolboxError("alpha must be between 0.0 (pure keyword) and 1.0 (pure vector).")
        col, config = self._collection(client, collection, tenant)
        embedded_by = "provided" if vector is not None else None
        if vector is None and alpha > 0:
            vector, embedded_by = self._query_vector(config, target_vector, query, embed_provider, embed_model, embed_dimension)
        if fusion_type not in {"relative_score", "ranked"}:
            raise ToolboxError("fusion_type must be 'relative_score' (default) or 'ranked'.")
        op = None
        if bm25_operator:
            if bm25_operator == "and":
                op = BM25Operator.and_()
            elif bm25_operator == "or":
                op = BM25Operator.or_(minimum_match=minimum_match or 1)
            else:
                raise ToolboxError("bm25_operator must be 'and' or 'or'.")
        self._check_query_props(config, query_properties)
        res = col.query.hybrid(
            query=query, alpha=alpha, vector=vector, query_properties=query_properties,
            fusion_type=HybridFusion.RELATIVE_SCORE if fusion_type == "relative_score" else HybridFusion.RANKED,
            max_vector_distance=max_vector_distance, bm25_operator=op, limit=limit, offset=offset,
            auto_limit=auto_limit, target_vector=self._target_vector(target_vector, target_vector_combination, target_vector_weights),
            **self._common(config, filters, return_properties, return_references, return_metadata,
                           ["score", "explain_score"], group_by, rerank, include_vector),
        )
        return {"collection": collection, "mode": "hybrid", "alpha": alpha,
                "query_vector": embedded_by or "none (alpha=0)", **_shape_result(res, with_vector=include_vector)}

    def semantic(self, client: str | None, collection: str, *, query: str | None = None,
                 vector: list[float] | dict[str, list[float]] | None = None, near_object: str | None = None,
                 target_vector: str | list[str] | None = None, target_vector_combination: str | None = None,
                 target_vector_weights: dict[str, float] | None = None, distance: float | None = None,
                 certainty: float | None = None, filters: dict[str, Any] | None = None, limit: int = 10,
                 offset: int | None = None, auto_limit: int | None = None, group_by: dict[str, Any] | None = None,
                 rerank: dict[str, Any] | None = None, return_properties: list[str] | None = None,
                 return_references: list[str] | None = None, return_metadata: list[str] | None = None,
                 include_vector: bool = False, tenant: str | None = None, embed_provider: str | None = None,
                 embed_model: str | None = None, embed_dimension: int | None = None) -> dict[str, Any]:
        if sum(x is not None for x in (query, vector, near_object)) != 1:
            raise ToolboxError("Pass exactly one of query (text), vector, or near_object (a uuid).")
        if distance is not None and certainty is not None:
            raise ToolboxError("Use distance or certainty as the threshold, not both.")
        col, config = self._collection(client, collection, tenant)
        common = self._common(config, filters, return_properties, return_references, return_metadata,
                              ["distance", "certainty"], group_by, rerank, include_vector)
        tv = self._target_vector(target_vector, target_vector_combination, target_vector_weights)
        kw = dict(distance=distance, certainty=certainty, limit=limit, offset=offset, auto_limit=auto_limit,
                  target_vector=tv, **common)
        if near_object is not None:
            res, how = col.query.near_object(near_object=_uuid(near_object, "near_object"), **kw), "near_object"
        elif vector is not None:
            res, how = col.query.near_vector(near_vector=vector, **kw), "near_vector (provided)"
        else:
            qvec, embedded_by = self._query_vector(config, target_vector, query, embed_provider, embed_model, embed_dimension)
            if qvec is None:
                res, how = col.query.near_text(query=query, **kw), f"near_text ({embedded_by})"
            else:
                res, how = col.query.near_vector(near_vector=qvec, **kw), f"near_vector ({embedded_by})"
        return {"collection": collection, "mode": how, **_shape_result(res, with_vector=include_vector)}

    @staticmethod
    def _check_query_props(config: dict[str, Any], query_properties: list[str] | None) -> None:
        if not query_properties:
            return
        searchable = {p["name"] for p in summarize(config)["properties"] if p.get("searchable")}
        names = [q.split("^")[0] for q in query_properties]
        bad = [n for n in names if n not in searchable]
        if bad:
            raise CapabilityError(
                f"query_properties {bad} are not BM25-searchable text properties. Searchable: "
                f"{', '.join(sorted(searchable)) or 'none'}. Boost with 'title^2'."
            )

    def keyword(self, client: str | None, collection: str, query: str, *, query_properties: list[str] | None = None,
                operator: str | None = None, minimum_match: int | None = None, filters: dict[str, Any] | None = None,
                limit: int = 10, offset: int | None = None, auto_limit: int | None = None,
                group_by: dict[str, Any] | None = None, rerank: dict[str, Any] | None = None,
                return_properties: list[str] | None = None, return_references: list[str] | None = None,
                return_metadata: list[str] | None = None, include_vector: bool = False,
                tenant: str | None = None) -> dict[str, Any]:
        from weaviate.classes.query import BM25Operator

        col, config = self._collection(client, collection, tenant)
        searchable = [p["name"] for p in summarize(config)["properties"] if p.get("searchable")]
        if not searchable:
            raise CapabilityError(f"{collection} has no BM25-searchable text property; keyword search cannot match.")
        self._check_query_props(config, query_properties)
        op = None
        if operator == "and":
            op = BM25Operator.and_()
        elif operator == "or":
            op = BM25Operator.or_(minimum_match=minimum_match or 1)
        elif operator is not None:
            raise ToolboxError("operator must be 'and' or 'or'.")
        res = col.query.bm25(
            query=query, query_properties=query_properties, operator=op, limit=limit, offset=offset,
            auto_limit=auto_limit,
            **self._common(config, filters, return_properties, return_references, return_metadata,
                           ["score", "explain_score"], group_by, rerank, include_vector),
        )
        return {"collection": collection, "mode": "bm25", **_shape_result(res, with_vector=include_vector)}

    def aggregate(self, client: str | None, collection: str, *, metrics: list[dict[str, Any]] | None = None,
                  filters: dict[str, Any] | None = None, group_by: dict[str, Any] | None = None,
                  tenant: str | None = None) -> dict[str, Any]:
        from weaviate.classes.aggregate import GroupByAggregate, Metrics

        col, config = self._collection(client, collection, tenant)
        types = property_types(config)
        built = []
        for i, m in enumerate(metrics or []):
            prop = m.get("property")
            if prop not in types:
                raise ToolboxError(f"metrics[{i}]: unknown property {prop!r}.")
            base = types[prop].rstrip("[]")
            kind = {"int": "integer", "number": "number", "text": "text", "boolean": "boolean", "date": "date_"}.get(base)
            if kind is None:
                raise CapabilityError(f"metrics[{i}]: cannot aggregate a {types[prop]} property.")
            wanted = set(m.get("metrics") or [])
            builder = getattr(Metrics(prop), kind)
            if kind == "text":
                built.append(builder(count=True, top_occurrences_count=True, top_occurrences_value=True,
                                     limit=m.get("top_occurrences", 5)))
            else:
                allowed = {"integer": {"count", "maximum", "mean", "median", "minimum", "mode", "sum"},
                           "number": {"count", "maximum", "mean", "median", "minimum", "mode", "sum"},
                           "boolean": {"count", "percentage_true", "percentage_false", "total_true", "total_false"},
                           "date_": {"count", "maximum", "median", "minimum", "mode"}}[kind]
                wanted = wanted or allowed
                bad = wanted - allowed
                if bad:
                    raise ToolboxError(f"metrics[{i}] ({prop}): {sorted(bad)} not available; use {sorted(allowed)}.")
                built.append(builder(**{("sum_" if w == "sum" else w): True for w in wanted}))
        gb = None
        if group_by:
            if group_by.get("property") not in types:
                raise ToolboxError("group_by needs {'property': <existing property>, 'limit': n}.")
            gb = GroupByAggregate(prop=group_by["property"], limit=group_by.get("limit", 10))
        res = col.aggregate.over_all(
            filters=compile_filter(filters, types) if filters else None, group_by=gb,
            total_count=True, return_metrics=built or None,
        )
        if gb is not None:
            groups = [{"group": jsonable(g.grouped_by.value), "count": g.total_count,
                       "metrics": jsonable(g.properties)} for g in res.groups]
            return {"collection": collection, "groups": groups}
        return {"collection": collection, "total_count": res.total_count, "metrics": jsonable(res.properties)}

    def validate_filter(self, client: str | None, filters: dict[str, Any], collection: str | None = None) -> dict[str, Any]:
        canonical = normalize(filters)
        if collection:
            col = self._raw_collection(client, collection)
            compile_filter(filters, property_types(self._config(col)))
        else:
            compile_filter(filters)
        return {"valid": True, "normalized": canonical,
                "checked_against": collection or "syntax only (pass collection to check property names and types)"}

    # ======================================================================
    # VectorStoreBackend contract (index == collection, namespace == tenant)
    # ======================================================================
    def list_indexes(self) -> list[dict[str, Any]]:
        return self.list_collections(None)["collections"]

    def describe_index(self, index: str) -> dict[str, Any]:
        return self.get_config(None, index)

    def delete_index(self, index: str, confirm: bool = False) -> dict[str, Any]:
        return self.delete_collection(None, index, confirm=confirm)

    def list_namespaces(self, index: str) -> list[dict[str, Any]]:
        return self.list_tenants(None, index)["tenants"]

    def delete_namespace(self, index: str, namespace: str, confirm: bool = False) -> dict[str, Any]:
        return self.delete_tenants(None, index, [namespace], confirm=confirm)

    def upsert(self, index: str, namespace: str, records: list[dict[str, Any]]) -> dict[str, Any]:
        return self.insert_objects(None, index, records, tenant=namespace or None, upsert=True)

    def fetch(self, index: str, namespace: str, ids: list[str] | None = None) -> list[dict[str, Any]]:
        return self.fetch_objects(None, index, ids=ids, tenant=namespace or None)["objects"]

    def delete(self, index: str, namespace: str, ids: list[str] | None = None,
               filter: dict[str, Any] | None = None, delete_all: bool = False) -> dict[str, Any]:
        return self.delete_objects(None, index, ids=ids, filters=filter, delete_all=delete_all,
                                   confirm=delete_all, tenant=namespace or None)

    def search(self, index: str, namespace: str, *, mode: str, query: str | None = None,
               vector: list[float] | None = None, fields: list[str] | None = None, top_k: int = 10,
               filter: dict[str, Any] | None = None, include_fields: list[str] | None = None) -> list[Hit]:
        tenant = namespace or None
        if mode == "text":
            res = self.keyword(None, index, query or "", query_properties=fields, filters=filter, limit=top_k, tenant=tenant)
        elif mode in {"dense", "semantic"}:
            res = self.semantic(None, index, query=query if vector is None else None, vector=vector,
                                filters=filter, limit=top_k, tenant=tenant)
        elif mode in {"hybrid", "auto"}:
            res = self.hybrid(None, index, query or "", vector=vector, query_properties=fields,
                              filters=filter, limit=top_k, tenant=tenant)
        else:
            raise CapabilityError(f"Weaviate supports modes text, dense, hybrid - not {mode!r}.")
        hits = []
        for o in res.get("objects", []):
            m = o.get("metadata", {})
            score = m.get("score") if m.get("score") is not None else (1 - m["distance"] if "distance" in m else 0.0)
            hits.append(Hit(id=o["uuid"], score=float(score), fields=o.get("properties", {})))
        return hits

    def stats(self, index: str) -> dict[str, Any]:
        return self.get_config(None, index)
