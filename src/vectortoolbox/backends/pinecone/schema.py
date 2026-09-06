"""Schema construction and capability detection.

Two jobs:

1. Turn the toolbox's field specs into the ``schema={"fields": {...}}`` dict
   that ``pc.indexes.create`` wants. Pinecone only lets you *declare*
   searchable fields - ``dense_vector``, ``sparse_vector`` and ``string`` with
   full-text search on. Plain metadata is indexed automatically the first time
   it appears on a record, so declaring it is an error, not a no-op.

2. Read a live index back and work out what it can answer. Every search tool
   checks this first, which is what makes "search with different options
   depending on how the index was created" safe rather than trial and error.
"""

from __future__ import annotations

from typing import Any, Iterable

from ...core.types import (
    DenseFieldSpec,
    IndexCapabilities,
    SparseFieldSpec,
    TextFieldSpec,
)
from ...errors import CapabilityError

DECLARABLE_TYPES = {"dense_vector", "sparse_vector", "string"}

# Pinecone surfaces an older Vectors-style index through the same schema API,
# naming its vector fields with these system names. Their leading underscore is
# the giveaway: a user-declared field can never start with "_".
SYSTEM_VECTOR_FIELDS = {"_values", "_sparse_values"}

# Documented Pinecone limits, checked here so a bad schema fails before the
# network call rather than as an opaque 400.
MAX_DENSE_FIELDS = 1
MAX_SPARSE_FIELDS = 1
MAX_FTS_FIELDS = 100
MAX_FIELD_NAME_BYTES = 64
MAX_DOCUMENTS_PER_REQUEST = 1000
MAX_SCORE_BY_CLAUSES = 100
MAX_TOP_K = 10_000


def validate_field_name(name: str) -> None:
    """Field names are unique, non-empty, <=64 bytes, and never _ or $ prefixed.

    The prefixes are reserved: ``_`` for system fields (``_id``, ``_score``)
    and ``$`` for filter operators. This applies to metadata fields too, not
    only schema-declared ones, and one bad name fails an entire upsert.
    """
    if not name:
        raise CapabilityError("Field names must be non-empty strings.")
    if name.startswith("_"):
        raise CapabilityError(
            f"Field name {name!r} starts with '_', which Pinecone reserves for system fields "
            "such as _id and _score. One invalid field name fails the whole request."
        )
    if name.startswith("$"):
        raise CapabilityError(
            f"Field name {name!r} starts with '$', which Pinecone reserves for filter operators."
        )
    if len(name.encode("utf-8")) > MAX_FIELD_NAME_BYTES:
        raise CapabilityError(
            f"Field name {name!r} is longer than {MAX_FIELD_NAME_BYTES} bytes."
        )


def build_schema(
    dense_fields: Iterable[DenseFieldSpec] | None = None,
    sparse_fields: Iterable[SparseFieldSpec] | None = None,
    text_fields: Iterable[TextFieldSpec] | None = None,
) -> dict[str, Any]:
    fields: dict[str, Any] = {}

    for spec in dense_fields or []:
        validate_field_name(spec.name)
        entry: dict[str, Any] = {
            "type": "dense_vector",
            "dimension": spec.dimension,
            "metric": spec.metric,
        }
        if spec.description:
            entry["description"] = spec.description
        fields[spec.name] = entry

    for spec in sparse_fields or []:
        validate_field_name(spec.name)
        entry = {"type": "sparse_vector"}
        if spec.description:
            entry["description"] = spec.description
        fields[spec.name] = entry

    for spec in text_fields or []:
        validate_field_name(spec.name)
        if not spec.full_text_search:
            raise CapabilityError(
                f"Text field {spec.name!r} has full_text_search disabled. Pinecone schemas may "
                "only declare searchable fields; a filter-only string should simply be included "
                "on the records you upsert - it is indexed for filtering automatically."
            )
        fts: dict[str, Any] = {"language": spec.language}
        if spec.stemming is not None:
            fts["stemming"] = spec.stemming
        if spec.stop_words is not None:
            fts["stop_words"] = spec.stop_words
        if spec.ngram:
            fts["ngram"] = spec.ngram
        entry = {"type": "string", "full_text_search": fts}
        if spec.filterable:
            entry["filterable"] = True
        fields[spec.name] = entry

    if not fields:
        raise CapabilityError(
            "An index needs at least one searchable field: a dense vector field, a sparse "
            "vector field, or a string field with full-text search enabled."
        )

    dense_count = len(list(dense_fields or []))
    sparse_count = len(list(sparse_fields or []))
    fts_count = len(list(text_fields or []))
    if dense_count > MAX_DENSE_FIELDS:
        raise CapabilityError(
            f"A Pinecone index may declare at most {MAX_DENSE_FIELDS} dense_vector field; "
            f"{dense_count} were given. Combine the signals into one vector, or use separate "
            "indexes."
        )
    if sparse_count > MAX_SPARSE_FIELDS:
        raise CapabilityError(
            f"A Pinecone index may declare at most {MAX_SPARSE_FIELDS} sparse_vector field; "
            f"{sparse_count} were given."
        )
    if fts_count > MAX_FTS_FIELDS:
        raise CapabilityError(
            f"A Pinecone index may declare at most {MAX_FTS_FIELDS} full-text string fields; "
            f"{fts_count} were given."
        )
    return {"fields": fields}


def _to_builtins(obj: Any) -> Any:
    """Normalise msgspec structs / pydantic models / dicts to plain dicts."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _to_builtins(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtins(v) for v in obj]
    try:
        import msgspec

        return msgspec.to_builtins(obj)
    except Exception:
        pass
    for attr in ("to_dict", "model_dump", "dict"):
        method = getattr(obj, attr, None)
        if callable(method):
            try:
                return _to_builtins(method())
            except Exception:
                continue
    if hasattr(obj, "__dict__"):
        return {k: _to_builtins(v) for k, v in vars(obj).items() if not k.startswith("_")}
    return str(obj)


def capabilities_from_index_model(model: Any) -> IndexCapabilities:
    data = _to_builtins(model)
    name = data.get("name", "<unknown>")
    schema = data.get("schema") or {}
    fields: dict[str, Any] = schema.get("fields") or {}

    dense: list[str] = []
    sparse: list[str] = []
    fts: list[str] = []
    semantic: list[str] = []
    filterable: list[str] = []
    dimensions: dict[str, int] = {}
    metrics: dict[str, str] = {}

    for field_name, config in fields.items():
        config = config or {}
        ftype = config.get("type")
        if ftype == "dense_vector":
            dense.append(field_name)
            if config.get("dimension"):
                dimensions[field_name] = int(config["dimension"])
            if config.get("metric"):
                metrics[field_name] = str(config["metric"])
        elif ftype == "sparse_vector":
            sparse.append(field_name)
        elif ftype == "semantic_text":
            semantic.append(field_name)
            if config.get("dimension"):
                dimensions[field_name] = int(config["dimension"])
        elif ftype == "string" and config.get("full_text_search"):
            fts.append(field_name)
            if config.get("filterable"):
                filterable.append(field_name)
        else:
            filterable.append(field_name)

    modes: list[str] = []
    notes: list[str] = []

    if fts:
        modes += ["text", "query_string"]
    if dense or semantic:
        modes.append("dense")
    if sparse:
        modes.append("sparse")

    signal_kinds = sum(bool(x) for x in (dense or semantic, sparse, fts))
    hybrid_capable = signal_kinds >= 2

    # An index whose only vector fields carry Pinecone's system names, with no
    # text fields, is a Vectors-API index wearing a schema. Reading it through
    # the Documents API fails, so it has to be classified here rather than
    # discovered at query time.
    vector_named = [f for f in dense + sparse if f in SYSTEM_VECTOR_FIELDS]
    legacy = bool(vector_named) and len(vector_named) == len(dense + sparse) and not fts

    if semantic:
        api = "integrated"
        notes.append(
            "Integrated index: text is embedded server-side. Use pinecone_search_records - "
            "you do not supply vectors."
        )
    elif legacy:
        api = "vectors"
        notes.append(
            "Vectors-API index: its vector fields carry Pinecone's system names "
            f"({', '.join(sorted(vector_named))}), so it is read with pinecone_query_vectors "
            "and written with pinecone_upsert_vectors. The Documents API does not serve it."
        )
    else:
        api = "documents"

    vectors_api = len(dense) == 1 and not fts and not semantic and len(sparse) <= 1
    if vectors_api:
        modes.append("vectors_api")
        if not legacy:
            notes.append(
                "Single-vector index: pinecone_query_vectors can run a dense+sparse hybrid in "
                "one request via the Vectors API."
            )
        else:
            notes.append(
                "Dense and sparse are scored together in one server-side request - no "
                "client-side fusion needed."
            )

    if not fts:
        notes.append(
            "No full-text search field. FTS fields must be declared at creation time and a "
            "schema cannot be altered afterwards - the index would need recreating."
        )

    if hybrid_capable:
        modes.append("hybrid")
        if api == "documents":
            notes.append(
                "Pinecone requires a dense_vector or sparse_vector clause to stand alone, so "
                "hybrid search here runs one request per signal and fuses the results with RRF."
            )

    return IndexCapabilities(
        index=name,
        api=api,
        dense_fields=dense,
        sparse_fields=sparse,
        fts_fields=fts,
        semantic_text_fields=semantic,
        filterable_fields=filterable,
        dimensions=dimensions,
        metrics=metrics,
        supported_search_modes=sorted(set(modes)),
        host=data.get("host"),
        deletion_protection=data.get("deletion_protection"),
        notes=notes,
    )


def require_mode(caps: IndexCapabilities, mode: str) -> None:
    if mode in caps.supported_search_modes:
        return
    raise CapabilityError(
        f"Index {caps.index!r} cannot run a {mode!r} search. It supports: "
        f"{', '.join(caps.supported_search_modes) or 'nothing searchable'}. "
        f"Declared fields - dense: {caps.dense_fields or 'none'}, "
        f"sparse: {caps.sparse_fields or 'none'}, "
        f"full-text: {caps.fts_fields or 'none'}, "
        f"semantic_text: {caps.semantic_text_fields or 'none'}. "
        "Schemas are fixed at creation time, so adding a signal means creating a new index."
    )


def resolve_field(caps: IndexCapabilities, kind: str, requested: str | None) -> str:
    available = {
        "dense": caps.dense_fields,
        "sparse": caps.sparse_fields,
        "fts": caps.fts_fields,
    }[kind]
    if requested:
        if requested not in available:
            raise CapabilityError(
                f"Index {caps.index!r} has no {kind} field named {requested!r}. "
                f"Available: {', '.join(available) or 'none'}."
            )
        return requested
    if not available:
        raise CapabilityError(f"Index {caps.index!r} declares no {kind} field.")
    return available[0]


def validate_documents(documents, caps: IndexCapabilities) -> None:
    """Catch the upsert failures Pinecone rejects the whole request for.

    A single bad document fails the entire batch and nothing is written, so it
    is worth naming every offender in one message rather than discovering them
    one round trip at a time.
    """
    schema_fields = set(
        caps.dense_fields + caps.sparse_fields + caps.fts_fields + caps.semantic_text_fields
    )
    missing_id: list[int] = []
    metadata_only: list[str] = []
    bad_names: list[str] = []
    seen: set[str] = set()
    duplicates: list[str] = []

    for position, document in enumerate(documents):
        doc_id = document.get("_id") or document.get("id")
        if not doc_id:
            missing_id.append(position)
            continue
        if doc_id in seen:
            duplicates.append(str(doc_id))
        seen.add(doc_id)

        for name in document:
            if name in {"_id", "id"}:
                continue
            if name.startswith("_") or name.startswith("$"):
                bad_names.append(f"{doc_id}.{name}")

        if schema_fields and not (schema_fields & set(document)):
            metadata_only.append(str(doc_id))

    problems = []
    if missing_id:
        problems.append(f"documents at position(s) {missing_id} have no _id")
    if bad_names:
        problems.append(
            f"field names reserved by Pinecone (leading _ or $): {sorted(set(bad_names))[:10]}"
        )
    if metadata_only:
        problems.append(
            f"documents carrying only metadata and no schema field: {sorted(set(metadata_only))[:10]} "
            f"(the schema declares {sorted(schema_fields)})"
        )
    if duplicates:
        problems.append(f"duplicate _id values in one request: {sorted(set(duplicates))[:10]}")

    if problems:
        raise CapabilityError(
            "This upsert would be rejected in full - Pinecone fails the whole request if any "
            "document is invalid. Problems: " + "; ".join(problems) + "."
        )

    if len(documents) > MAX_DOCUMENTS_PER_REQUEST:
        raise CapabilityError(
            f"{len(documents)} documents exceeds the {MAX_DOCUMENTS_PER_REQUEST}-per-request "
            "limit. Lower batch_size so the toolbox splits the write."
        )


def validate_top_k(top_k: int) -> None:
    if not 1 <= top_k <= MAX_TOP_K:
        raise CapabilityError(f"top_k must be between 1 and {MAX_TOP_K}; got {top_k}.")
