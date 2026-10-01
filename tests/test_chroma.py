"""Chroma backend - exercised against a real in-memory Chroma, not fakes.

Chroma runs in-process, so these tests use the actual SDK: an ephemeral
client for most cases and a temp-dir persistent client for persistence. No
embedding model is downloaded - records carry explicit vectors, or a tiny
deterministic toolbox embedder is patched in.
"""

from __future__ import annotations

import asyncio
import hashlib
import json

import pytest

chromadb = pytest.importorskip("chromadb")


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
class HashEmbedder:
    """Deterministic 8-dim bag-of-words embedder - no model, no network."""

    provider = "fake"
    model = "hash-8"

    def __init__(self, dimension: int | None = 8):
        self._dimension = dimension or 8
        self.calls: list[list[str]] = []

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self._dimension
        for word in text.lower().split():
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self._dimension] += 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        self.calls.append([text])
        return self._vec(text)

    @property
    def dimension(self):
        return self._dimension


@pytest.fixture
def chroma(monkeypatch):
    from vectortoolbox import embeddings
    from vectortoolbox.backends.chroma import client as client_mod
    from vectortoolbox.backends.chroma.backend import ChromaBackend

    embedder = HashEmbedder()
    monkeypatch.setattr(embeddings, "get_dense_embedder", lambda *a, **k: embedder)
    monkeypatch.setitem(embeddings.DENSE_PROVIDERS, "fake", HashEmbedder)

    client_mod.reset_clients()
    entry = client_mod.register_client("default", "ephemeral")
    backend = ChromaBackend()
    yield {"backend": backend, "client": entry.client, "embedder": embedder}
    for col in entry.client.list_collections():
        entry.client.delete_collection(col.name)
    client_mod.reset_clients()


def _movies(b, name="movies", **create):
    b.create_collection("default", name, hnsw={"space": "cosine"}, **create)
    b.write(
        "default", name, "add",
        records=[
            {"id": "m1", "document": "An action film about fast cars",
             "embedding": [1, 0, 0, 0, 0, 0, 0, 0], "metadata": {"genres": ["action", "comedy"], "year": 2020}},
            {"id": "m2", "document": "A quiet drama about family",
             "embedding": [0, 1, 0, 0, 0, 0, 0, 0], "metadata": {"genres": ["drama"], "year": 2021}},
            {"id": "m3", "document": "Thriller with action and a heist",
             "embedding": [0.9, 0.1, 0, 0, 0, 0, 0, 0], "metadata": {"genres": ["action", "thriller"], "year": 2022}},
        ],
    )


# --------------------------------------------------------------------------
# Clients
# --------------------------------------------------------------------------
def test_persistent_client_survives_a_new_client(tmp_path):
    from vectortoolbox.backends.chroma import client as client_mod
    from vectortoolbox.backends.chroma.backend import ChromaBackend

    client_mod.reset_clients()
    path = str(tmp_path / "store")
    client_mod.register_client("disk", "persistent", path=path)
    b = ChromaBackend()
    b.create_collection("disk", "kept", hnsw={"space": "cosine"})
    b.write("disk", "kept", "add", ids=["a"], embeddings=[[1.0, 0.0]], documents=["hello"])

    client_mod.register_client("disk2", "persistent", path=path)
    assert b.count("disk2", "kept")["count"] == 1
    assert b.heartbeat("disk")["path"] == path
    client_mod.reset_clients()


def test_cloud_client_without_key_explains_itself(monkeypatch):
    from vectortoolbox.backends.chroma import client as client_mod
    from vectortoolbox.errors import ConfigurationError

    monkeypatch.delenv("CHROMA_API_KEY", raising=False)
    with pytest.raises(ConfigurationError, match="CHROMA_API_KEY"):
        client_mod.register_client("c", "cloud")


def test_unknown_client_and_duplicate_names(chroma):
    from vectortoolbox.backends.chroma import client as client_mod
    from vectortoolbox.errors import ConfigurationError, NotFound

    with pytest.raises(NotFound, match="chroma_create_client"):
        client_mod.get_entry("nope")
    with pytest.raises(ConfigurationError, match="replace=true"):
        client_mod.register_client("default", "ephemeral")


# --------------------------------------------------------------------------
# Collections + configuration
# --------------------------------------------------------------------------
def test_create_describe_modify_delete_collection(chroma):
    from vectortoolbox.errors import ConfirmationRequired

    b = chroma["backend"]
    out = b.create_collection("default", "docs", metadata={"owner": "me"},
                              hnsw={"space": "cosine", "ef_construction": 200})
    assert out["space"] == "cosine"
    assert out["configuration"]["hnsw"]["ef_construction"] == 200

    again = b.create_collection("default", "docs", get_or_create=True, metadata={"x": 1})
    assert again["created"] is False and "NOT applied" in again["note"]

    b.modify_collection("default", "docs", metadata={"team": "search"})
    meta = b.describe_collection("default", "docs")["metadata"]
    assert meta == {"owner": "me", "team": "search"}
    b.modify_collection("default", "docs", metadata={"owner": None})
    assert b.describe_collection("default", "docs")["metadata"] == {"team": "search"}

    b.modify_collection("default", "docs", new_name="docs-v2")
    assert [c["name"] for c in b.list_collections("default")["collections"]] == ["docs-v2"]

    with pytest.raises(ConfirmationRequired):
        b.delete_collection("default", "docs-v2")
    assert b.delete_collection("default", "docs-v2", confirm=True)["deleted"] == "docs-v2"


def test_index_configuration_rules(chroma):
    from vectortoolbox.errors import CapabilityError

    b = chroma["backend"]
    b.create_collection("default", "cfg", hnsw={"space": "cosine"})
    out = b.configure_collection("default", "cfg", hnsw={"ef_search": 42})
    assert out["before"]["ef_search"] == 100 and out["after"]["ef_search"] == 42

    with pytest.raises(CapabilityError, match="fixed at creation"):
        b.configure_collection("default", "cfg", hnsw={"space": "l2"})
    with pytest.raises(CapabilityError, match="not both"):
        b.create_collection("default", "x1x", hnsw={"space": "l2"}, spann={"search_nprobe": 64})
    with pytest.raises(CapabilityError, match="silently ignore"):
        b.create_collection("default", "x2x", spann={"search_nprobe": 64})
    with pytest.raises(CapabilityError, match="1-128"):
        b.create_collection("default", "x3x", spann={"search_nprobe": 500})
    with pytest.raises(CapabilityError, match="hnsw="):
        b.create_collection("default", "x4x", metadata={"hnsw:space": "cosine"})


def test_embedding_function_spec_refuses_raw_keys(chroma):
    from vectortoolbox.errors import ConfigurationError

    b = chroma["backend"]
    with pytest.raises(ConfigurationError, match="api_key_env_var"):
        b.create_collection("default", "efx", embedding_function={"name": "openai", "kwargs": {"api_key": "sk-x"}})
    with pytest.raises(ConfigurationError, match="Unknown Chroma embedding function"):
        b.create_collection("default", "efy", embedding_function={"name": "nope"})


def test_toolbox_embedder_is_remembered_by_the_collection(chroma):
    b, embedder = chroma["backend"], chroma["embedder"]
    out = b.create_collection("default", "tbx", toolbox_embedder={"provider": "fake", "model": "hash-8"},
                              hnsw={"space": "cosine"})
    assert out["embedding"] == {"route": "toolbox", "provider": "fake", "model": "hash-8"}

    report = b.write("default", "tbx", "add", ids=["a", "b"], documents=["red apples", "blue cars"])
    assert report["embedded_by"] == "toolbox:fake/hash-8"
    assert embedder.calls[-1] == ["red apples", "blue cars"]

    res = b.query("default", "tbx", query_texts=["apples"], n_results=1)
    assert res["embedded_by"].startswith("toolbox:")
    assert res["queries"][0]["results"][0]["id"] == "a"


# --------------------------------------------------------------------------
# Writes
# --------------------------------------------------------------------------
def test_add_reports_existing_ids_instead_of_silently_skipping(chroma):
    b = chroma["backend"]
    _movies(b)
    report = b.write("default", "movies", "add", ids=["m1", "m9"],
                     embeddings=[[1] + [0] * 7, [0, 0, 1, 0, 0, 0, 0, 0]], documents=["dup", "new"])
    assert report["skipped_existing"] == ["m1"] and report["written"] == 1
    assert b.get("default", "movies", ids=["m1"])["records"][0]["document"].startswith("An action")

    from vectortoolbox.errors import ToolboxError
    with pytest.raises(ToolboxError, match="add never overwrites"):
        b.write("default", "movies", "add", ids=["m1"], embeddings=[[1] + [0] * 7], on_conflict="error")


def test_update_merges_metadata_and_reports_missing(chroma):
    b = chroma["backend"]
    _movies(b)
    report = b.write("default", "movies", "update", ids=["m2", "ghost"],
                     metadatas=[{"rating": 4.5, "year": None}, {"rating": 1}])
    assert report["skipped_missing"] == ["ghost"] and report["written"] == 1
    meta = b.get("default", "movies", ids=["m2"])["records"][0]["metadata"]
    assert meta == {"genres": ["drama"], "rating": 4.5}


def test_upsert_counts_created_and_updated(chroma):
    b = chroma["backend"]
    _movies(b)
    report = b.write("default", "movies", "upsert", ids=["m1", "m4"],
                     embeddings=[[1] + [0] * 7, [0] * 7 + [1]], documents=["changed", "brand new"])
    assert (report["created"], report["updated"]) == (1, 1)
    assert b.count("default", "movies")["count"] == 4


def test_write_validation_happens_before_sending(chroma):
    from vectortoolbox.errors import ToolboxError

    b = chroma["backend"]
    _movies(b)
    cases = [
        (dict(ids=["a", "a"], embeddings=[[1] * 8, [1] * 8]), "Duplicate ids"),
        (dict(ids=["a", "b"], embeddings=[[1] * 8]), "must match"),
        (dict(ids=["a"], embeddings=[[1] * 8], metadatas=[{"tags": []}]), "empty array"),
        (dict(ids=["a"], embeddings=[[1] * 8], metadatas=[{"tags": [1, "x"]}]), "one type"),
        (dict(ids=["a"], embeddings=[[1] * 8], metadatas=[{"nested": {"a": 1}}]), "flat"),
        (dict(ids=["a", "b"], embeddings=[[1] * 8, None], documents=["x", "y"]), "all or none"),
    ]
    for kwargs, message in cases:
        with pytest.raises(ToolboxError, match=message):
            b.write("default", "movies", "add", **kwargs)


def test_delete_by_filter_dry_run_and_delete_all(chroma):
    from vectortoolbox.errors import ConfirmationRequired

    b = chroma["backend"]
    _movies(b)
    dry = b.delete_records("default", "movies", where={"genres": {"$contains": "action"}}, dry_run=True)
    assert dry["would_delete"] == 2 and b.count("default", "movies")["count"] == 3

    out = b.delete_records("default", "movies", where_document={"$contains": "drama"})
    assert out["deleted"] == 1

    with pytest.raises(ConfirmationRequired):
        b.delete_records("default", "movies", delete_all=True)
    assert b.delete_records("default", "movies", delete_all=True, confirm=True)["deleted"] == 2
    assert b.count("default", "movies")["count"] == 0


# --------------------------------------------------------------------------
# Conditional transactions
# --------------------------------------------------------------------------
def test_conditional_transaction_refuses_without_sdk_support(chroma):
    from vectortoolbox.errors import CapabilityError

    b = chroma["backend"]
    _movies(b)
    col = chroma["client"].get_collection("movies")
    if hasattr(col, "conditional"):
        pytest.skip("installed chromadb supports conditional transactions natively")
    with pytest.raises(CapabilityError, match="allow_non_atomic"):
        b.conditional("default", "movies", checks=[{"id": "m1"}],
                      writes=[{"op": "update", "ids": ["m1"], "metadatas": [{"x": 1}]}])


def test_conditional_checks_gate_the_writes(chroma):
    b = chroma["backend"]
    _movies(b)
    common = dict(allow_non_atomic=True)

    failed = b.conditional(
        "default", "movies",
        checks=[{"id": "m1", "metadata": {"year": 1999}}],
        writes=[{"op": "update", "ids": ["m1"], "metadatas": [{"status": "published"}]}], **common,
    )
    assert failed["committed"] is False and "metadata differs" in failed["checks"][0]["reason"]
    assert "status" not in b.get("default", "movies", ids=["m1"])["records"][0]["metadata"]

    ok = b.conditional(
        "default", "movies",
        checks=[{"id": "m1", "metadata": {"year": 2020}}, {"id": "m7", "exists": False},
                {"where": {"year": {"$gte": 2021}}, "max_count": 2}],
        writes=[{"op": "update", "ids": ["m1"], "metadatas": [{"status": "published"}]},
                {"op": "add", "ids": ["m7"], "embeddings": [[0] * 7 + [1]], "documents": ["seventh"]},
                {"op": "delete", "ids": ["m2"]}], **common,
    )
    assert ok["committed"] is True and all(c["passed"] for c in ok["checks"])
    assert b.count("default", "movies")["count"] == 3


def test_conditional_rejects_what_chroma_forbids(chroma):
    from vectortoolbox.errors import ToolboxError

    b = chroma["backend"]
    _movies(b)
    with pytest.raises(ToolboxError, match="at most one write per id"):
        b.conditional("default", "movies", checks=[], allow_non_atomic=True, writes=[
            {"op": "update", "ids": ["m1"], "metadatas": [{"a": 1}]},
            {"op": "delete", "ids": ["m1"]},
        ])
    with pytest.raises(ToolboxError, match="explicit ids only"):
        b.conditional("default", "movies", allow_non_atomic=True,
                      writes=[{"op": "delete", "ids": ["m1"], "where": {"year": 2020}}])


def test_conditional_uses_native_transactions_when_available(chroma, monkeypatch):
    """Simulate an SDK with Collection.conditional() and check the call shape."""
    b = chroma["backend"]
    _movies(b)
    real = chroma["client"].get_collection("movies")
    log = []

    class Txn:
        def get(self, **kw):
            log.append(("get", kw))
            return real.get(**kw)

        def update(self, **kw):
            log.append(("update", kw))

        def run(self, fn, max_retries):
            log.append(("run", max_retries))
            fn(self)
            return {"committed": True}

    class Col:
        def __getattr__(self, item):
            return getattr(real, item)

        def conditional(self):
            return Txn()

    monkeypatch.setattr(type(b), "_collection", lambda self, client, name, embedding_function=None: Col())
    out = b.conditional("default", "movies", checks=[{"id": "m1"}], max_retries=5,
                        writes=[{"op": "update", "ids": ["m1"], "metadatas": [{"x": 1}]}])
    assert out["atomic"] is True and out["committed"] is True
    assert log[0] == ("run", 5) and log[1][0] == "get" and log[2][0] == "update"


# --------------------------------------------------------------------------
# Query / get / results shape
# --------------------------------------------------------------------------
def test_query_rows_columns_and_similarity(chroma):
    b = chroma["backend"]
    _movies(b)
    res = b.query("default", "movies", query_embeddings=[[1] + [0] * 7, [0, 1] + [0] * 6],
                  n_results=2, output="both")
    assert [q["results"][0]["id"] for q in res["queries"]] == ["m1", "m2"]
    top = res["queries"][0]["results"][0]
    assert top["similarity"] == pytest.approx(1.0) and top["distance"] == pytest.approx(0.0, abs=1e-6)
    assert len(res["columns"]["ids"]) == 2 and len(res["columns"]["ids"][0]) == 2
    json.dumps(res)  # JSON-safe


def test_include_controls_returned_fields(chroma):
    from vectortoolbox.errors import ToolboxError

    b = chroma["backend"]
    _movies(b)
    res = b.get("default", "movies", ids=["m1"], include=["embeddings"])
    assert set(res["records"][0]) == {"id", "embedding"} and len(res["records"][0]["embedding"]) == 8
    json.dumps(res)
    with pytest.raises(ToolboxError, match="distances only exist"):
        b.get("default", "movies", include=["distances"])
    paged = b.get("default", "movies", limit=2)
    assert paged["next_offset"] == 2


# --------------------------------------------------------------------------
# Metadata filtering + FTS
# --------------------------------------------------------------------------
def test_filters_logical_inclusion_and_arrays(chroma):
    b = chroma["backend"]
    _movies(b)
    ids = lambda where: sorted(r["id"] for r in b.get("default", "movies", where=where)["records"])  # noqa: E731
    assert ids({"genres": {"$contains": "action"}}) == ["m1", "m3"]
    assert ids({"genres": {"$not_contains": "action"}}) == ["m2"]
    assert ids({"year": {"$in": [2020, 2022]}}) == ["m1", "m3"]
    assert ids({"year": {"$gte": 2021, "$lt": 2022}}) == ["m2"]           # rewritten to $and
    assert ids({"genres": {"$contains": "action"}, "year": 2022}) == ["m3"]  # implicit $and
    assert ids({"$or": [{"year": 2020}, {"genres": {"$contains": "drama"}}]}) == ["m1", "m2"]
    assert ids({"$and": [{"year": 2021}]}) == ["m2"]                      # single-child unwrap


def test_filter_errors_are_explanations():
    from vectortoolbox.backends.chroma.filters import (
        FilterError,
        normalize_where,
        normalize_where_document,
    )

    for bad, message in [
        ({"genres": ["action"]}, r"\$contains"),
        ({"x": None}, "null"),
        ({"x": {"$exists": True}}, "no \\$exists"),
        ({"$not": {"x": 1}}, "no \\$not"),
        ({"year": {"$gt": "2020"}}, "needs a number"),
        ({"x": {"$in": [1, "a"]}}, "one type"),
        ({"$and": [], }, "non-empty list"),
        ({"x": {"$regex": "a"}}, "documents only"),
    ]:
        with pytest.raises(FilterError, match=message):
            normalize_where(bad)
    with pytest.raises(FilterError, match="look-around"):
        normalize_where_document({"$regex": "foo(?=bar)"})
    with pytest.raises(FilterError, match="Metadata conditions belong in `where`"):
        normalize_where_document({"year": 2020})
    assert normalize_where({}) is None


def test_full_text_search_filter_and_combined_with_vector(chroma):
    b = chroma["backend"]
    _movies(b)
    res = b.full_text_search("default", "movies", contains=["action"])
    assert sorted(r["id"] for r in res["records"]) == ["m1", "m3"] and "none" in res["ranking"]

    res = b.full_text_search("default", "movies", contains=["heist", "family"], match="any")
    assert sorted(r["id"] for r in res["records"]) == ["m2", "m3"]

    res = b.full_text_search("default", "movies", regex=["(?i)^an action"])
    assert [r["id"] for r in res["records"]] == ["m1"]

    res = b.full_text_search("default", "movies", contains=["action"], not_contains=["cars"],
                             where={"year": {"$gte": 2020}})
    assert [r["id"] for r in res["records"]] == ["m3"]

    res = b.full_text_search("default", "movies", contains=["action"], query_embedding=[0.9, 0.1] + [0] * 6)
    assert [r["id"] for r in res["queries"][0]["results"]] == ["m3", "m1"]


def test_sample_metadata_marks_arrays(chroma):
    b = chroma["backend"]
    _movies(b)
    fields = b.sample_metadata("default", "movies")["fields"]
    assert fields["genres"]["types"] == ["str[]"] and "$contains" in fields["genres"]["operators"]
    assert "$gte" in fields["year"]["operators"]


# --------------------------------------------------------------------------
# Tool layer
# --------------------------------------------------------------------------
@pytest.fixture
def server(chroma, monkeypatch):
    from vectortoolbox.backends.chroma import tools as tools_mod
    from vectortoolbox.server import build_server

    monkeypatch.setattr(tools_mod, "_backend", lambda: chroma["backend"])
    return build_server()


def call(server, tool, **kwargs):
    result = asyncio.run(server.call_tool(tool, kwargs))
    if isinstance(result, tuple):
        return result[1].get("result", result[1])
    structured = getattr(result, "structured_content", None) or getattr(result, "structuredContent", None)
    if structured is not None:
        return structured.get("result", structured)
    return json.loads(result.content[0].text)


def test_chroma_tools_registered_with_descriptions(server):
    tools = {t.name: t for t in asyncio.run(server.list_tools())}
    chroma_tools = {n for n in tools if n.startswith("chroma_")}
    assert len(chroma_tools) == 24
    for name in chroma_tools:
        assert tools[name].description, name
    assert "Metadata is flat" in tools["chroma_add"].description


def test_tool_round_trip_and_readable_errors(server):
    created = call(server, "chroma_create_collection", name="tooltest", hnsw={"space": "cosine"})
    assert created["space"] == "cosine"
    added = call(server, "chroma_add", collection="tooltest",
                 records=[{"id": "a", "document": "hello", "embedding": [1.0, 0.0]}])
    assert added["written"] == 1
    bad = call(server, "chroma_get", collection="tooltest", where={"tags": ["x"]})
    assert bad["error"] == "FilterError"
    missing = call(server, "chroma_count", collection="nope")
    assert missing["error"] == "NotFound" and "tooltest" in missing["message"]
    status = call(server, "vectortoolbox_status")
    assert status["chroma"]["installed"] is True


def test_collection_name_is_checked_up_front(chroma):
    from vectortoolbox.errors import ToolboxError

    with pytest.raises(ToolboxError, match="3-512"):
        chroma["backend"].create_collection("default", "ab")
