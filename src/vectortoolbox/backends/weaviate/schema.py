"""Collection definitions: high-level spec -> Weaviate class JSON, and back.

``build_collection`` turns the tool's arguments into the REST class object
that ``collections.create_from_dict`` accepts. That format is the most
complete one Weaviate has (every module, index and quantizer option is
reachable), and ``raw_config`` lets a caller pass it directly. Everything is
validated before the request so mistakes come back as explanations.

``summarize`` turns a collection's full config into the compact view the
tools return: properties, references, named vectors (vectorizer, index,
distance, quantizer), multi-tenancy, replication, BM25 and modules.
"""

from __future__ import annotations

import re
from typing import Any

from ...errors import CapabilityError, ConfigurationError

DATA_TYPES = {
    "text", "text[]", "int", "int[]", "number", "number[]", "boolean", "boolean[]",
    "date", "date[]", "uuid", "uuid[]", "geoCoordinates", "phoneNumber", "blob", "object", "object[]",
}
TOKENIZATIONS = {"word", "lowercase", "whitespace", "field", "trigram", "gse", "kagome_ja", "kagome_kr", "gse_ch"}
INDEX_TYPES = {"hnsw", "flat", "dynamic", "hfresh"}
DISTANCES = {"cosine", "dot", "l2-squared", "hamming", "manhattan"}
QUANTIZERS = {"pq", "bq", "sq", "rq"}
STOPWORD_PRESETS = {"en", "none"}
COLLECTION_RE = re.compile(r"^[A-Z][_0-9A-Za-z]*$")
PROPERTY_RE = re.compile(r"^[_A-Za-z][_0-9A-Za-z]*$")
RESERVED_PROPERTIES = {"id", "_id", "_additional", "vector", "vectors"}

# Snake-case keys the tool accepts for property flags -> REST names.
PROPERTY_FLAGS = {
    "filterable": "indexFilterable",
    "searchable": "indexSearchable",
    "range_filters": "indexRangeFilters",
    "index_filterable": "indexFilterable",
    "index_searchable": "indexSearchable",
    "index_range_filters": "indexRangeFilters",
}


def check_collection_name(name: str) -> str:
    if not isinstance(name, str) or not name:
        raise ConfigurationError("A collection needs a name.")
    if not COLLECTION_RE.match(name):
        fixed = name[:1].upper() + name[1:]
        hint = f" Did you mean {fixed!r}?" if COLLECTION_RE.match(fixed) else ""
        raise ConfigurationError(
            f"Invalid collection name {name!r}: it must start with a capital letter and contain only "
            f"letters, digits and underscores (Weaviate collections are GraphQL types).{hint}"
        )
    return name


def _property(spec: dict[str, Any], at: str, module_names: set[str]) -> dict[str, Any]:
    if not isinstance(spec, dict) or "name" not in spec:
        raise ConfigurationError(f"{at} needs at least {{'name': ..., 'data_type': ...}}.")
    name = spec["name"]
    if not isinstance(name, str) or not PROPERTY_RE.match(name):
        raise ConfigurationError(
            f"{at}: invalid property name {name!r} - letters, digits and underscores, not starting with a digit."
        )
    if name in RESERVED_PROPERTIES:
        raise ConfigurationError(f"{at}: {name!r} is reserved by Weaviate; choose another name.")
    dtype = spec.get("data_type") or spec.get("dataType") or spec.get("type")
    if isinstance(dtype, list):
        dtype = dtype[0] if dtype else None
    if dtype not in DATA_TYPES:
        raise ConfigurationError(
            f"{at} ({name}): data_type must be one of {', '.join(sorted(DATA_TYPES))}; got {dtype!r}. "
            "For a link to another collection use references=[...]."
        )
    out: dict[str, Any] = {"name": name, "dataType": [dtype]}
    if spec.get("description"):
        out["description"] = spec["description"]
    tok = spec.get("tokenization")
    if tok is not None:
        if tok not in TOKENIZATIONS:
            raise ConfigurationError(f"{at} ({name}): tokenization must be one of {', '.join(sorted(TOKENIZATIONS))}.")
        if not dtype.startswith("text"):
            raise ConfigurationError(f"{at} ({name}): tokenization only applies to text properties.")
        out["tokenization"] = tok
    for key, rest in PROPERTY_FLAGS.items():
        if key in spec:
            out[rest] = bool(spec[key])
    if out.get("indexRangeFilters") and dtype not in {"int", "number", "date"}:
        raise ConfigurationError(f"{at} ({name}): range_filters needs an int, number or date property.")
    if out.get("indexSearchable") and not dtype.startswith("text"):
        raise ConfigurationError(f"{at} ({name}): searchable (BM25) needs a text property.")
    skip = spec.get("skip_vectorization")
    vpn = spec.get("vectorize_property_name")
    if skip is not None or vpn is not None:
        out["moduleConfig"] = {
            m: {k: v for k, v in (("skip", skip), ("vectorizePropertyName", vpn)) if v is not None}
            for m in module_names
        }
    if dtype.startswith("object"):
        nested = spec.get("nested_properties") or spec.get("nestedProperties")
        if not nested:
            raise ConfigurationError(f"{at} ({name}): object properties need nested_properties=[...].")
        out["nestedProperties"] = [
            _property(n, f"{at}.nested_properties[{i}]", set()) for i, n in enumerate(nested)
        ]
    return out


def _vector(spec: dict[str, Any], at: str, prop_names: set[str]) -> tuple[str, dict[str, Any]]:
    name = spec.get("name", "default")
    if not isinstance(name, str) or not PROPERTY_RE.match(name):
        raise ConfigurationError(f"{at}: invalid vector name {name!r}.")
    vectorizer = spec.get("vectorizer") or "none"
    vconf = dict(spec.get("vectorizer_config") or {})
    if spec.get("model"):
        vconf.setdefault("model", spec["model"])
    if spec.get("dimensions"):
        vconf.setdefault("dimensions", spec["dimensions"])
    sources = spec.get("source_properties")
    if sources:
        if vectorizer == "none":
            raise ConfigurationError(f"{at} ({name}): source_properties needs a vectorizer, not 'none'.")
        unknown = [p for p in sources if p not in prop_names]
        if unknown:
            raise ConfigurationError(f"{at} ({name}): source_properties {unknown} are not properties of this collection.")
        vconf["properties"] = list(sources)
    index_type = spec.get("index_type", "hnsw")
    if index_type not in INDEX_TYPES:
        raise ConfigurationError(f"{at} ({name}): index_type must be one of {', '.join(sorted(INDEX_TYPES))}.")
    index_conf = dict(spec.get("index_config") or {})
    distance = spec.get("distance")
    if distance:
        if distance not in DISTANCES:
            raise ConfigurationError(f"{at} ({name}): distance must be one of {', '.join(sorted(DISTANCES))}.")
        index_conf["distance"] = distance
    quantizer = spec.get("quantizer")
    if quantizer and quantizer != "none":
        if quantizer not in QUANTIZERS:
            raise ConfigurationError(f"{at} ({name}): quantizer must be one of {', '.join(sorted(QUANTIZERS))} or 'none'.")
        if index_type == "flat" and quantizer not in {"bq", "rq"}:
            raise ConfigurationError(f"{at} ({name}): a flat index supports only bq or rq quantization.")
        index_conf[quantizer] = {"enabled": True, **(spec.get("quantizer_config") or {})}
    out = {"vectorizer": {vectorizer: vconf}, "vectorIndexType": index_type}
    if index_conf:
        out["vectorIndexConfig"] = index_conf
    return name, out


def build_collection(
    name: str,
    *,
    description: str | None = None,
    properties: list[dict[str, Any]] | None = None,
    references: list[dict[str, Any]] | None = None,
    vectors: list[dict[str, Any]] | None = None,
    multi_tenancy: dict[str, Any] | bool | None = None,
    replication_factor: int | None = None,
    bm25: dict[str, float] | None = None,
    stopwords_preset: str | None = None,
    index_timestamps: bool | None = None,
    index_null_state: bool | None = None,
    index_property_length: bool | None = None,
    generative: dict[str, Any] | None = None,
    reranker: dict[str, Any] | None = None,
    enabled_modules: set[str] | None = None,
) -> dict[str, Any]:
    check_collection_name(name)
    vectors = vectors or [{"name": "default", "vectorizer": "none"}]
    vector_modules = {v.get("vectorizer") for v in vectors if v.get("vectorizer") not in (None, "none")}

    props = [_property(p, f"properties[{i}]", vector_modules) for i, p in enumerate(properties or [])]
    names = [p["name"] for p in props]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ConfigurationError(f"Duplicate property names: {dupes}.")

    for i, ref in enumerate(references or []):
        rname, target = ref.get("name"), ref.get("target") or ref.get("target_collection")
        if not rname or not target:
            raise ConfigurationError(f"references[{i}] needs {{'name': ..., 'target': 'OtherCollection'}}.")
        if not PROPERTY_RE.match(rname) or rname in names:
            raise ConfigurationError(f"references[{i}]: invalid or duplicate name {rname!r}.")
        targets = target if isinstance(target, list) else [target]
        for t in targets:
            check_collection_name(t)
        entry = {"name": rname, "dataType": targets}
        if ref.get("description"):
            entry["description"] = ref["description"]
        props.append(entry)
        names.append(rname)

    vector_config: dict[str, Any] = {}
    for i, v in enumerate(vectors):
        vname, vconf = _vector(v, f"vectors[{i}]", set(names))
        if vname in vector_config:
            raise ConfigurationError(f"Duplicate vector name {vname!r}.")
        vector_config[vname] = vconf

    if enabled_modules is not None:
        wanted = set(vector_modules)
        for block in (generative, reranker):
            if block and block.get("module"):
                wanted.add(block["module"])
        missing = sorted(wanted - enabled_modules)
        if missing:
            available = sorted(m for m in enabled_modules if m.split("-")[0] in {"text2vec", "multi2vec", "generative", "reranker", "img2vec", "text2multivec"})
            raise CapabilityError(
                f"Module(s) {missing} are not enabled on this Weaviate server. Enabled model modules: "
                f"{', '.join(available) or 'none'}. Use vectorizer 'none' and send your own vectors "
                "(embed_provider on the write and search tools), or enable the module on the server."
            )

    cls: dict[str, Any] = {"class": name, "properties": props, "vectorConfig": vector_config}
    if description:
        cls["description"] = description

    inverted: dict[str, Any] = {}
    if bm25:
        unknown = set(bm25) - {"b", "k1"}
        if unknown:
            raise ConfigurationError(f"bm25 takes 'b' and 'k1', got {sorted(unknown)}.")
        inverted["bm25"] = bm25
    if stopwords_preset:
        if stopwords_preset not in STOPWORD_PRESETS:
            raise ConfigurationError("stopwords_preset must be 'en' or 'none'.")
        inverted["stopwords"] = {"preset": stopwords_preset}
    for key, rest in (("index_timestamps", "indexTimestamps"), ("index_null_state", "indexNullState"),
                      ("index_property_length", "indexPropertyLength")):
        value = {"index_timestamps": index_timestamps, "index_null_state": index_null_state,
                 "index_property_length": index_property_length}[key]
        if value is not None:
            inverted[rest] = bool(value)
    if inverted:
        cls["invertedIndexConfig"] = inverted

    if multi_tenancy:
        mt = {"enabled": True} if multi_tenancy is True else {"enabled": True, **{
            {"auto_tenant_creation": "autoTenantCreation", "auto_tenant_activation": "autoTenantActivation"}.get(k, k): v
            for k, v in multi_tenancy.items() if k != "enabled"}}
        cls["multiTenancyConfig"] = mt
    if replication_factor:
        cls["replicationConfig"] = {"factor": int(replication_factor)}

    module_config: dict[str, Any] = {}
    for block in (generative, reranker):
        if block:
            if not block.get("module"):
                raise ConfigurationError("generative / reranker need {'module': 'generative-openai', ...settings}.")
            module_config[block["module"]] = {k: v for k, v in block.items() if k != "module"}
    if module_config:
        cls["moduleConfig"] = module_config
    return cls


def property_types(config: dict[str, Any]) -> dict[str, str]:
    """name -> dataType string, for filter validation."""
    out = {}
    for p in config.get("properties", []):
        dtype = (p.get("dataType") or ["?"])[0]
        out[p["name"]] = dtype
    return out


def summarize(config: dict[str, Any]) -> dict[str, Any]:
    """Compact view of a collection's REST config."""
    props, refs = [], []
    for p in config.get("properties", []):
        dtype = p.get("dataType") or []
        if dtype and dtype[0][:1].isupper():  # a class name = cross-reference
            refs.append({"name": p["name"], "targets": dtype})
            continue
        entry = {"name": p["name"], "data_type": dtype[0] if dtype else None}
        for flag, rest in (("filterable", "indexFilterable"), ("searchable", "indexSearchable"),
                           ("range_filters", "indexRangeFilters")):
            if p.get(rest):
                entry[flag] = True
        if p.get("tokenization"):
            entry["tokenization"] = p["tokenization"]
        if p.get("description"):
            entry["description"] = p["description"]
        if p.get("nestedProperties"):
            entry["nested_properties"] = [n["name"] for n in p["nestedProperties"]]
        props.append(entry)

    vectors = {}
    for vname, v in (config.get("vectorConfig") or {}).items():
        vec = v.get("vectorizer") or {}
        module = next(iter(vec), "none")
        settings = vec.get(module) or {}
        idx = v.get("vectorIndexConfig") or {}
        quantizer = next((q for q in QUANTIZERS if (idx.get(q) or {}).get("enabled")), None)
        vectors[vname] = {
            "vectorizer": module,
            **({"model": settings["model"]} if settings.get("model") else {}),
            **({"source_properties": settings["properties"]} if settings.get("properties") else {}),
            "index_type": v.get("vectorIndexType"),
            "distance": idx.get("distanceMetric") or idx.get("distance"),
            "quantizer": quantizer,
            **({"ef": idx["ef"]} if "ef" in idx else {}),
        }
    if not vectors and config.get("vectorizer"):  # legacy single-vector collection
        idx = config.get("vectorIndexConfig") or {}
        vectors["default"] = {"vectorizer": config["vectorizer"], "index_type": config.get("vectorIndexType"),
                              "distance": idx.get("distanceMetric") or idx.get("distance"),
                              "legacy": True}

    mt = config.get("multiTenancyConfig") or {}
    inv = config.get("invertedIndexConfig") or {}
    modules = config.get("moduleConfig") or {}
    return {
        "name": config.get("class"),
        "description": config.get("description"),
        "properties": props,
        "references": refs,
        "vectors": vectors,
        "multi_tenancy": {"enabled": bool(mt.get("enabled")),
                          "auto_tenant_creation": bool(mt.get("autoTenantCreation")),
                          "auto_tenant_activation": bool(mt.get("autoTenantActivation"))},
        "replication_factor": (config.get("replicationConfig") or {}).get("factor"),
        "bm25": inv.get("bm25"),
        "stopwords": (inv.get("stopwords") or {}).get("preset"),
        "index_timestamps": inv.get("indexTimestamps"),
        "index_null_state": inv.get("indexNullState"),
        "index_property_length": inv.get("indexPropertyLength"),
        "generative": next((m for m in modules if m.startswith("generative-")), None),
        "reranker": next((m for m in modules if m.startswith("reranker-")), None),
    }
