"""Weaviate tools against a real server.

Skipped unless a Weaviate is reachable. Point at one with
``VTB_TEST_WEAVIATE=host:http_port:grpc_port`` (default localhost:8080:50051):

    docker run -d -p 8080:8080 -p 50051:50051 \
      -e DEFAULT_VECTORIZER_MODULE=none cr.weaviate.io/semitechnologies/weaviate:1.37.0

Only collections prefixed ``VtbTest`` are created, and they are deleted after.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import socket

import pytest

weaviate = pytest.importorskip("weaviate")

HOST, HTTP_PORT, GRPC_PORT = (os.environ.get("VTB_TEST_WEAVIATE", "localhost:8080:50051").split(":") + ["", "", ""])[:3]


def _reachable() -> bool:
    try:
        with socket.create_connection((HOST, int(HTTP_PORT)), timeout=0.5):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _reachable(), reason=f"no Weaviate at {HOST}:{HTTP_PORT}")


class HashEmbedder:
    provider, model = "fake", "hash-8"

    def __init__(self, *a, **k):
        self._dimension = 8

    def _vec(self, text):
        v = [0.0] * 8
        for w in text.lower().split():
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 8] += 1.0
        n = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)

    @property
    def dimension(self):
        return 8


@pytest.fixture
def wv(monkeypatch):
    from vectortoolbox import embeddings
    from vectortoolbox.backends.weaviate import client as client_mod
    from vectortoolbox.backends.weaviate.backend import WeaviateBackend

    monkeypatch.setattr(embeddings, "get_dense_embedder", lambda *a, **k: HashEmbedder())
    client_mod.reset_clients()
    entry = client_mod.register_client("default", "local", host=HOST, port=int(HTTP_PORT), grpc_port=int(GRPC_PORT))

    def cleanup():
        for name in list(entry.client.collections.list_all(simple=True)):
            if name.startswith("VtbTest"):
                entry.client.collections.delete(name)

    cleanup()
    yield WeaviateBackend()
    cleanup()
    client_mod.reset_clients()


BOOKS = [
    {"properties": {"sku": "b1", "title": "Deep learning for vision", "body": "neural networks see images",
                    "year": 2019, "price": 40.0, "tags": ["ai", "vision"], "in_print": True}},
    {"properties": {"sku": "b2", "title": "Gardening basics", "body": "soil water and sunlight",
                    "year": 2021, "price": 15.5, "tags": ["garden"], "in_print": True}},
    {"properties": {"sku": "b3", "title": "Neural search engines", "body": "vector search with neural networks",
                    "year": 2023, "price": 55.0, "tags": ["ai", "search"], "in_print": False}},
]
PROPS = [
    {"name": "sku", "data_type": "text", "tokenization": "field"},
    {"name": "title", "data_type": "text", "searchable": True},
    {"name": "body", "data_type": "text"},
    {"name": "year", "data_type": "int", "range_filters": True},
    {"name": "price", "data_type": "number"},
    {"name": "tags", "data_type": "text[]"},
    {"name": "in_print", "data_type": "boolean"},
]


def _books(b, name="VtbTestBooks", **kw):
    b.create_collection(None, name, properties=PROPS, vectors=[{"name": "default", "distance": "cosine"}],
                        index_property_length=True, **kw)
    out = b.insert_objects(None, name, BOOKS, id_from="sku", embed_source="body")
    assert out["inserted"] == 3 and "errors" not in out, out
    return out


# --------------------------------------------------------------------------
def test_create_and_describe(wv):
    out = wv.create_collection(None, "VtbTestBooks", properties=PROPS,
                               vectors=[{"name": "default", "distance": "cosine", "quantizer": "rq"},
                                        {"name": "titles", "index_type": "flat"}],
                               bm25={"b": 0.7, "k1": 1.1})
    assert out["created"] and set(out["vectors"]) == {"default", "titles"}
    assert out["vectors"]["default"]["quantizer"] == "rq" and out["vectors"]["titles"]["index_type"] == "flat"
    assert out["bm25"] == {"b": 0.7, "k1": 1.1}
    cfg = wv.get_config(None, "VtbTestBooks")
    assert {p["name"] for p in cfg["properties"]} == {p["name"] for p in PROPS}
    assert any(c["name"] == "VtbTestBooks" for c in wv.list_collections(None)["collections"])
    again = wv.create_collection(None, "VtbTestBooks", properties=PROPS, if_not_exists=True)
    assert again["created"] is False


def test_missing_module_is_explained(wv):
    from vectortoolbox.errors import CapabilityError

    with pytest.raises(CapabilityError, match="not enabled on this Weaviate server"):
        wv.create_collection(None, "VtbTestX", vectors=[{"vectorizer": "text2vec-nonexistent"}])


def test_insert_is_idempotent_with_id_from_and_reports_new_properties(wv):
    from vectortoolbox.errors import ToolboxError

    _books(wv)
    again = wv.insert_objects(None, "VtbTestBooks", BOOKS, id_from="sku", embed_source="body")
    assert again["inserted"] == 0 and len(again["skipped_existing"]) == 3
    with pytest.raises(ToolboxError, match="auto-schema"):
        wv.insert_objects(None, "VtbTestBooks", [{"properties": {"titel": "typo"}}])
    warn = wv.insert_objects(None, "VtbTestBooks", [{"properties": {"sku": "b9", "title": "no vector"}}])
    assert "warning" in warn


def test_upsert_replace_vs_merge_and_update(wv):
    out = _books(wv)
    first = out["uuids"][0]
    wv.insert_objects(None, "VtbTestBooks", [{"uuid": first, "properties": {"price": 1.0}}], upsert=True, merge=True)
    obj = wv.fetch_objects(None, "VtbTestBooks", ids=[first])["objects"][0]
    assert obj["properties"]["price"] == 1.0 and obj["properties"]["title"] == "Deep learning for vision"
    wv.update_object(None, "VtbTestBooks", first, properties={"year": 2020})
    assert wv.fetch_objects(None, "VtbTestBooks", ids=[first])["objects"][0]["properties"]["year"] == 2020


def test_filters_native_and_mongo_style_agree(wv):
    _books(wv)
    native = {"operator": "And", "operands": [
        {"path": ["year"], "operator": "GreaterThanEqual", "valueInt": 2020},
        {"path": ["tags"], "operator": "ContainsAny", "valueTextArray": ["ai"]}]}
    mongo = {"year": {"$gte": 2020}, "tags": {"$in": ["ai"]}}
    a = wv.fetch_objects(None, "VtbTestBooks", filters=native)["objects"]
    b = wv.fetch_objects(None, "VtbTestBooks", filters=mongo)["objects"]
    assert [o["properties"]["sku"] for o in a] == [o["properties"]["sku"] for o in b] == ["b3"]
    like = wv.fetch_objects(None, "VtbTestBooks", filters={"path": "title", "operator": "Like", "value": "*neural*"})
    assert [o["properties"]["sku"] for o in like["objects"]] == ["b3"]
    notf = wv.fetch_objects(None, "VtbTestBooks", filters={"$not": {"in_print": True}})
    assert [o["properties"]["sku"] for o in notf["objects"]] == ["b3"]
    length = wv.fetch_objects(None, "VtbTestBooks", filters={"len(tags)": {"$gte": 2}})
    assert sorted(o["properties"]["sku"] for o in length["objects"]) == ["b1", "b3"]


def test_filter_type_errors_are_caught_before_sending(wv):
    from vectortoolbox.backends.weaviate.filters import FilterError

    _books(wv)
    with pytest.raises(FilterError, match="no property 'yeer'"):
        wv.fetch_objects(None, "VtbTestBooks", filters={"yeer": 2020})
    with pytest.raises(FilterError, match="not a number"):
        wv.fetch_objects(None, "VtbTestBooks", filters={"year": {"$gt": "recent"}})


def test_hybrid_keyword_semantic(wv):
    _books(wv)
    h = wv.hybrid(None, "VtbTestBooks", "neural networks", alpha=0.5, limit=2)
    assert h["query_vector"].startswith("toolbox:") and h["objects"][0]["properties"]["sku"] in {"b1", "b3"}
    assert "score" in h["objects"][0]["metadata"]
    k = wv.keyword(None, "VtbTestBooks", "gardening", query_properties=["title^2"])
    assert [o["properties"]["sku"] for o in k["objects"]] == ["b2"]
    s = wv.semantic(None, "VtbTestBooks", query="vector search neural networks", limit=1)
    assert s["objects"][0]["properties"]["sku"] == "b3" and "distance" in s["objects"][0]["metadata"]
    first = s["objects"][0]["uuid"]
    more = wv.semantic(None, "VtbTestBooks", near_object=first, limit=2)
    assert more["objects"][0]["uuid"] == first
    g = wv.hybrid(None, "VtbTestBooks", "neural", group_by={"property": "in_print", "objects_per_group": 2})
    assert {grp["group"] for grp in g["groups"]} <= {True, False, "true", "false"}


def test_keyword_needs_searchable_property(wv):
    from vectortoolbox.errors import CapabilityError

    _books(wv)
    with pytest.raises(CapabilityError, match="not BM25-searchable"):
        wv.keyword(None, "VtbTestBooks", "x", query_properties=["year"])


def test_aggregate_and_delete(wv):
    from vectortoolbox.errors import ConfirmationRequired

    _books(wv)
    agg = wv.aggregate(None, "VtbTestBooks", metrics=[{"property": "price", "metrics": ["mean", "maximum"]}])
    assert agg["total_count"] == 3 and agg["metrics"]["price"]["maximum"] == 55.0
    grp = wv.aggregate(None, "VtbTestBooks", group_by={"property": "in_print"})
    assert sorted(g["count"] for g in grp["groups"]) == [1, 2]
    dry = wv.delete_objects(None, "VtbTestBooks", filters={"year": {"$lt": 2022}}, dry_run=True)
    assert dry["would_delete"] == 2
    assert wv.delete_objects(None, "VtbTestBooks", filters={"year": {"$lt": 2022}})["deleted"] == 2
    with pytest.raises(ConfirmationRequired):
        wv.delete_objects(None, "VtbTestBooks", delete_all=True)
    assert wv.delete_objects(None, "VtbTestBooks", delete_all=True, confirm=True)["deleted"] == 1
    with pytest.raises(ConfirmationRequired):
        wv.delete_collection(None, "VtbTestBooks")
    assert wv.delete_collection(None, "VtbTestBooks", confirm=True)["deleted"] == "VtbTestBooks"


def test_update_collection_and_add_property(wv):
    _books(wv)
    out = wv.update_collection(None, "VtbTestBooks", description="books", bm25={"b": 0.6},
                               vector_index={"default": {"ef": 64}})
    assert "description" in out["changed"] and out["changed"]["bm25"]["after"]["b"] == 0.6
    wv.add_to_collection(None, "VtbTestBooks", property={"name": "isbn", "data_type": "text"})
    assert any(p["name"] == "isbn" for p in wv.get_config(None, "VtbTestBooks")["properties"])


def test_multi_tenancy_flow(wv):
    from vectortoolbox.errors import CapabilityError, ConfirmationRequired

    wv.create_collection(None, "VtbTestNotes", properties=[{"name": "text", "data_type": "text"}],
                         multi_tenancy=True)
    with pytest.raises(CapabilityError, match="multi-tenant"):
        wv.insert_objects(None, "VtbTestNotes", [{"properties": {"text": "hi"}}])
    wv.create_tenants(None, "VtbTestNotes", ["acme", {"name": "globex", "activity_status": "INACTIVE"}])
    listed = wv.list_tenants(None, "VtbTestNotes")
    assert listed["by_status"] == {"ACTIVE": 1, "INACTIVE": 1}
    wv.insert_objects(None, "VtbTestNotes", [{"properties": {"text": "hello acme"}, "vector": [1.0, 0.0]}],
                      tenant="acme")
    assert wv.aggregate(None, "VtbTestNotes", tenant="acme")["total_count"] == 1
    wv.update_tenants(None, "VtbTestNotes", [{"name": "globex", "activity_status": "ACTIVE"}])
    assert wv.list_tenants(None, "VtbTestNotes")["by_status"] == {"ACTIVE": 2}
    with pytest.raises(ConfirmationRequired):
        wv.delete_tenants(None, "VtbTestNotes", ["globex"])
    wv.delete_tenants(None, "VtbTestNotes", ["globex"], confirm=True)


def test_references(wv):
    wv.create_collection(None, "VtbTestAuthor", properties=[{"name": "name", "data_type": "text"}])
    wv.create_collection(None, "VtbTestPost", properties=[{"name": "title", "data_type": "text"}],
                         references=[{"name": "author", "target": "VtbTestAuthor"}])
    a = wv.insert_objects(None, "VtbTestAuthor", [{"properties": {"name": "Ada"}, "vector": [1.0, 0.0]}])["uuids"][0]
    p = wv.insert_objects(None, "VtbTestPost", [{"properties": {"title": "On engines"}, "vector": [0.0, 1.0]}])["uuids"][0]
    assert wv.add_references(None, "VtbTestPost", [{"from_uuid": p, "from_property": "author", "to": a}])["added"] == 1
    got = wv.fetch_objects(None, "VtbTestPost", ids=[p], return_references=["author"])["objects"][0]
    assert got["references"]["author"][0]["properties"]["name"] == "Ada"
    through = wv.fetch_objects(None, "VtbTestPost", filters={"path": ["author", "VtbTestAuthor", "name"],
                                                             "operator": "Equal", "value": "Ada"})
    assert [o["uuid"] for o in through["objects"]] == [p]


def test_tools_through_mcp(wv, monkeypatch):
    from vectortoolbox.backends.weaviate import tools as tools_mod
    from vectortoolbox.server import build_server

    monkeypatch.setattr(tools_mod, "_backend", lambda: wv)
    server = build_server()

    def call(tool, **kw):
        r = asyncio.run(server.call_tool(tool, kw))
        sc = getattr(r, "structured_content", None) or getattr(r, "structuredContent", None)
        return sc.get("result", sc) if sc is not None else json.loads(r.content[0].text)

    _books(wv)
    cfg = call("weaviate_get_collection_config", collection_name="VtbTestBooks")
    assert cfg["name"] == "VtbTestBooks"
    hy = call("weaviate_hybrid_search", collection="VtbTestBooks", query="neural", alpha=0.0,
              filters={"operator": "Equal", "path": ["in_print"], "valueBoolean": False})
    assert [o["properties"]["sku"] for o in hy["objects"]] == ["b3"]
    up = call("weaviate_upsert_objects", collection="VtbTestBooks", id_from="sku", embed_source="body",
              objects=[{"properties": {"sku": "b2", "title": "Gardening 2nd ed", "body": "soil"}}])
    assert up["replaced"] == 1
    bad = call("weaviate_get_collection_config", collection_name="vtbtestbooks")
    assert bad["error"] == "NotFound" and "case-sensitive" in bad["message"]
    caps = call("weaviate_collection_capabilities", collection="VtbTestBooks")
    assert "hybrid" in caps["supported_search_modes"]
    status = call("vectortoolbox_status")
    assert status["weaviate"]["installed"] is True
