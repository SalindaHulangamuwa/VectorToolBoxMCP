"""Weaviate pieces that need no server: filters, schema building, client headers, tool registration."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("weaviate")

from vectortoolbox.backends.weaviate.filters import (  # noqa: E402
    FilterError,
    compile_filter,
    normalize,
)
from vectortoolbox.backends.weaviate.schema import build_collection, summarize  # noqa: E402
from vectortoolbox.errors import CapabilityError, ConfigurationError  # noqa: E402

TYPES = {"title": "text", "year": "int", "price": "number", "tags": "text[]", "ok": "boolean",
         "loc": "geoCoordinates", "published": "date"}


# --------------------------------------------------------------------------
# filters
# --------------------------------------------------------------------------
def test_native_and_mongo_normalise_to_the_same_shape():
    native = {"operator": "And", "operands": [
        {"path": ["year"], "operator": "GreaterThanEqual", "valueInt": 2020},
        {"path": "tags", "operator": "ContainsAny", "valueTextArray": ["ai"]}]}
    mongo = {"year": {"$gte": 2020}, "tags": {"$in": ["ai"]}}
    assert normalize(native) == normalize(mongo) == {"operator": "And", "operands": [
        {"path": ["year"], "operator": "GreaterThanEqual", "value": 2020},
        {"path": ["tags"], "operator": "ContainsAny", "value": ["ai"]}]}


def test_shorthands():
    assert normalize({}) is None
    assert normalize({"title": "x"}) == {"path": ["title"], "operator": "Equal", "value": "x"}
    assert normalize({"title": None}) == {"path": ["title"], "operator": "IsNull", "value": True}
    assert normalize({"tags": {"$contains": "ai"}})["value"] == ["ai"]
    single = {"operator": "Or", "operands": [{"path": ["year"], "operator": "Equal", "value": 1}]}
    assert normalize(single) == {"path": ["year"], "operator": "Equal", "value": 1}
    assert normalize({"$not": {"ok": True}})["operator"] == "Not"


@pytest.mark.parametrize("bad, message", [
    ({"operator": "Equals", "path": ["x"], "value": 1}, "unknown operator"),
    ({"operator": "Equal", "path": ["x"]}, "missing a value"),
    ({"operator": "Equal", "path": ["x"], "valueInt": 1, "valueText": "a"}, "one value key"),
    ({"operator": "Not", "operands": [{"x": 1}, {"y": 2}]}, "exactly one operand"),
    ({"operator": "And", "operands": []}, "non-empty"),
    ({"tags": ["a", "b"]}, r"\$in"),
    ({"year": {"$between": [1, 2]}}, "unknown operator"),
    ({"$where": "x"}, "not a top-level operator"),
    ({"path": ["a", "B"], "operator": "Equal", "value": 1}, "reference path"),
    ({"operator": "ContainsAny", "path": ["tags"], "value": "ai"}, "non-empty list"),
    ({"operator": "WithinGeoRange", "path": ["loc"], "value": {"lat": 1}}, "WithinGeoRange needs"),
])
def test_bad_filters_explain_themselves(bad, message):
    with pytest.raises(FilterError, match=message):
        normalize(bad)


@pytest.mark.parametrize("where, message", [
    ({"yeer": 1}, "no property 'yeer'"),
    ({"year": {"$gt": "soon"}}, "not a number"),
    ({"ok": "yes"}, "true/false"),
    ({"title": {"$gt": 3}}, "compares numbers"),
    ({"path": "year", "operator": "Like", "value": "2*"}, "Like works on text"),
    ({"loc": "here"}, "WithinGeoRange"),
    ({"published": {"$gt": "last week"}}, "RFC 3339"),
])
def test_type_checks_against_schema(where, message):
    with pytest.raises(FilterError, match=message):
        compile_filter(where, TYPES)


def test_compiles_every_operator_family():
    for where in [
        {"year": {"$gte": 2020, "$lt": 2030}},
        {"title": {"$like": "neur*"}},
        {"tags": {"$all": ["a", "b"]}},
        {"tags": {"$nin": ["x"]}},
        {"published": {"$gt": "2026-01-01T00:00:00Z"}},
        {"id": {"$in": ["00000000-0000-0000-0000-000000000001"]}},
        {"_creationTimeUnix": {"$gt": "2026-01-01T00:00:00+00:00"}},
        {"len(title)": {"$gt": 3}},
        {"count(author)": {"$gte": 1}},
        {"path": ["author", "Person", "name"], "operator": "Equal", "value": "Ada"},
        {"operator": "WithinGeoRange", "path": ["loc"],
         "valueGeoRange": {"geoCoordinates": {"latitude": 7.0, "longitude": 80.6}, "distance": {"max": 5000}}},
        {"$or": [{"ok": True}, {"$not": {"price": {"$gt": 10}}}]},
    ]:
        assert compile_filter(where, TYPES) is not None, where


# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------
def test_build_collection_maps_to_rest_shape():
    cls = build_collection(
        "Article",
        properties=[{"name": "title", "data_type": "text", "searchable": True, "tokenization": "word"},
                    {"name": "year", "data_type": "int", "range_filters": True}],
        references=[{"name": "author", "target": "Person"}],
        vectors=[{"name": "content", "vectorizer": "text2vec-openai", "model": "text-embedding-3-small",
                  "source_properties": ["title"], "distance": "cosine", "quantizer": "rq"},
                 {"name": "own", "index_type": "flat"}],
        multi_tenancy={"auto_tenant_creation": True}, bm25={"b": 0.7}, replication_factor=2,
        enabled_modules={"text2vec-openai"},
    )
    assert cls["class"] == "Article"
    assert cls["properties"][0] == {"name": "title", "dataType": ["text"], "tokenization": "word", "indexSearchable": True}
    assert cls["properties"][-1] == {"name": "author", "dataType": ["Person"]}
    content = cls["vectorConfig"]["content"]
    assert content["vectorizer"] == {"text2vec-openai": {"model": "text-embedding-3-small", "properties": ["title"]}}
    assert content["vectorIndexConfig"] == {"distance": "cosine", "rq": {"enabled": True}}
    assert cls["vectorConfig"]["own"]["vectorizer"] == {"none": {}}
    assert cls["multiTenancyConfig"] == {"enabled": True, "autoTenantCreation": True}
    assert cls["invertedIndexConfig"] == {"bm25": {"b": 0.7}}
    s = summarize(cls)
    assert s["references"] == [{"name": "author", "targets": ["Person"]}]
    assert s["vectors"]["content"]["quantizer"] == "rq"


@pytest.mark.parametrize("kwargs, message", [
    ({"name": "article"}, "capital letter"),
    ({"name": "A", "properties": [{"name": "id", "data_type": "text"}]}, "reserved"),
    ({"name": "A", "properties": [{"name": "x", "data_type": "string"}]}, "data_type must be"),
    ({"name": "A", "properties": [{"name": "x", "data_type": "int", "searchable": True}]}, "searchable"),
    ({"name": "A", "properties": [{"name": "x", "data_type": "text", "range_filters": True}]}, "range_filters"),
    ({"name": "A", "properties": [{"name": "x", "data_type": "int", "tokenization": "word"}]}, "tokenization"),
    ({"name": "A", "properties": [{"name": "x", "data_type": "text"}, {"name": "x", "data_type": "int"}]}, "Duplicate"),
    ({"name": "A", "vectors": [{"source_properties": ["x"]}]}, "needs a vectorizer"),
    ({"name": "A", "vectors": [{"index_type": "ivf"}]}, "index_type"),
    ({"name": "A", "vectors": [{"index_type": "flat", "quantizer": "pq"}]}, "flat index supports only"),
    ({"name": "A", "bm25": {"b": 0.7, "c": 1}}, "bm25 takes"),
    ({"name": "A", "references": [{"name": "r", "target": "lowercase"}]}, "capital letter"),
])
def test_schema_mistakes_are_explained(kwargs, message):
    with pytest.raises(ConfigurationError, match=message):
        build_collection(**kwargs)


def test_unknown_module_is_a_capability_error():
    with pytest.raises(CapabilityError, match="not enabled"):
        build_collection("A", vectors=[{"vectorizer": "text2vec-openai"}], enabled_modules={"generative-openai"})


# --------------------------------------------------------------------------
# client + tools
# --------------------------------------------------------------------------
def test_provider_keys_forwarded_by_name_only(monkeypatch):
    from vectortoolbox.backends.weaviate.client import provider_headers

    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    monkeypatch.setenv("MY_KEY", "abc")
    headers, names = provider_headers({"X-Custom-Api-Key": "MY_KEY"})
    assert headers["X-OpenAI-Api-Key"] == "sk-secret" and headers["X-Custom-Api-Key"] == "abc"
    assert "X-OpenAI-Api-Key" in names and "sk-secret" not in str(names)
    with pytest.raises(ConfigurationError, match="not set"):
        provider_headers({"X-Other": "UNSET_VAR_XYZ"})


def test_cloud_client_needs_url_and_key(monkeypatch):
    from vectortoolbox.backends.weaviate.client import build_client

    monkeypatch.delenv("WEAVIATE_URL", raising=False)
    monkeypatch.delenv("WEAVIATE_API_KEY", raising=False)
    with pytest.raises(ConfigurationError, match="cluster URL"):
        build_client("cloud")
    with pytest.raises(ConfigurationError, match="API key"):
        build_client("cloud", url="https://x.weaviate.cloud")


def test_weaviate_tools_registered_with_builtin_equivalents():
    from vectortoolbox.server import build_server

    tools = {t.name: t for t in asyncio.run(build_server().list_tools())}
    names = {n for n in tools if n.startswith("weaviate_")}
    assert len(names) == 27
    for builtin_equivalent in ("weaviate_get_collection_config", "weaviate_list_tenants",
                               "weaviate_hybrid_search", "weaviate_upsert_objects"):
        assert builtin_equivalent in names
    for name in names:
        assert tools[name].description, name
    assert "ContainsAny" in tools["weaviate_hybrid_search"].description
