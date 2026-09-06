"""Backend behaviour with the Pinecone client mocked out."""

from __future__ import annotations

import pytest

from conftest import (
    DENSE_FIELD,
    FTS_FIELD,
    LEGACY_VECTOR_FIELDS,
    SPARSE_FIELD,
    index_model,
)
from vectortoolbox.errors import CapabilityError, ConfirmationRequired


class StubEmbedder:
    provider = "stub"
    model = "stub-8"

    def __init__(self, dim=8):
        self.dim = dim

    def embed_documents(self, texts):
        return [[0.1] * self.dim for _ in texts]

    def embed_query(self, text):
        return [0.1] * self.dim

    def describe(self):
        return {"provider": self.provider, "model": self.model, "dimension": self.dim}


class StubSparse:
    provider = "stub"
    model = "stub-sparse"

    def embed_documents(self, texts):
        return [{"indices": [1, 2], "values": [0.5, 0.5]} for _ in texts]

    def embed_query(self, text):
        return {"indices": [1, 2], "values": [0.5, 0.5]}


@pytest.fixture
def stub_embedders(monkeypatch):
    from vectortoolbox.backends.pinecone import backend as backend_mod

    monkeypatch.setattr(backend_mod, "get_dense_embedder", lambda *a, **k: StubEmbedder())
    monkeypatch.setattr(backend_mod, "get_sparse_embedder", lambda *a, **k: StubSparse())


def _register(wired, name, fields):
    wired["models"][name] = index_model(name, fields)
    return wired["backend"]


def test_create_index_builds_managed_deployment(wired):
    backend = wired["backend"]
    from vectortoolbox.core.types import DenseFieldSpec, TextFieldSpec

    backend.create_index(
        name="articles",
        dense_fields=[DenseFieldSpec(name="embedding", dimension=8)],
        text_fields=[TextFieldSpec(name="body")],
        region="us-west-2",
    )
    call = wired["client"].indexes.created[0]
    assert call["deployment"] == {
        "deployment_type": "managed",
        "cloud": "aws",
        "region": "us-west-2",
    }
    assert set(call["schema"]["fields"]) == {"embedding", "body"}


def test_delete_index_requires_confirmation(wired):
    backend = _register(wired, "doomed", {"body": FTS_FIELD})
    with pytest.raises(ConfirmationRequired, match="confirm=true"):
        backend.delete_index("doomed")


def test_search_rejects_fts_on_a_dense_only_index(wired):
    backend = _register(wired, "dense-only", {"emb": DENSE_FIELD})
    with pytest.raises(CapabilityError, match="cannot run a 'text' search"):
        backend.search("dense-only", "ns", mode="text", query="hello")


def test_search_rejects_unknown_fts_field(wired):
    backend = _register(wired, "fts", {"body": FTS_FIELD})
    with pytest.raises(CapabilityError, match="not full-text searchable"):
        backend.search("fts", "ns", mode="text", query="hi", fields=["title"])


def test_multi_field_fts_sends_one_clause_per_field(wired):
    """A `text` clause names exactly one field; several fields = several clauses."""
    backend = _register(wired, "multi-fts", {"body": FTS_FIELD, "summary": FTS_FIELD})
    result = backend.search("multi-fts", "ns", mode="text", query="machine learning")
    assert result["fields"] == ["body", "summary"]
    assert result["clauses"] == 2
    assert wired["index"].calls[-1][1]["score_by"] == [
        {"type": "text", "field": "body", "query": "machine learning"},
        {"type": "text", "field": "summary", "query": "machine learning"},
    ]


def test_field_queries_allow_a_different_query_per_field(wired):
    backend = _register(wired, "multi-fts", {"body": FTS_FIELD, "summary": FTS_FIELD})
    backend.search(
        "multi-fts", "ns", mode="text",
        field_queries={"body": "disappointing", "summary": "Disappointing"},
    )
    assert wired["index"].calls[-1][1]["score_by"] == [
        {"type": "text", "field": "body", "query": "disappointing"},
        {"type": "text", "field": "summary", "query": "Disappointing"},
    ]


def test_a_vector_clause_is_never_combined_with_another(wired, stub_embedders):
    """Pinecone rejects a vector clause that shares a request with any other."""
    backend = _register(wired, "dense", {"emb": DENSE_FIELD})
    with pytest.raises(CapabilityError, match="only clause"):
        backend._run_clauses(
            "dense", "ns",
            [{"type": "dense_vector", "field": "emb", "values": [0.1] * 8},
             {"type": "text", "field": "body", "query": "x"}],
            top_k=5, include_fields=None, filter=None,
        )


def test_query_string_mode_passes_lucene_through(wired):
    backend = _register(wired, "fts", {"body": FTS_FIELD})
    backend.search("fts", "ns", mode="query_string", query="body:(machine AND learning)")
    clause = wired["index"].calls[-1][1]["score_by"][0]
    assert clause["type"] == "query_string"
    assert clause["query"] == "body:(machine AND learning)"


def test_dense_search_embeds_the_query(wired, stub_embedders):
    backend = _register(wired, "dense", {"emb": DENSE_FIELD})
    backend.search("dense", "ns", mode="dense", query="hello")
    clause = wired["index"].calls[-1][1]["score_by"][0]
    assert clause["type"] == "dense_vector"
    assert len(clause["values"]) == 8


def test_dense_search_rejects_a_mismatched_vector(wired):
    backend = _register(wired, "dense", {"emb": DENSE_FIELD})
    with pytest.raises(CapabilityError, match="dimension 3 but field 'emb' expects 8"):
        backend.search("dense", "ns", mode="dense", vector=[0.1, 0.2, 0.3])


def test_hybrid_runs_one_request_per_signal_and_fuses(wired, stub_embedders):
    backend = _register(
        wired, "multi", {"emb": DENSE_FIELD, "sp": SPARSE_FIELD, "body": FTS_FIELD}
    )
    result = backend.search("multi", "ns", mode="hybrid", query="refund policy", top_k=3)
    assert result["mode"] == "hybrid"
    assert set(result["signals"]) == {"dense", "sparse", "text"}
    searches = [c for c in wired["index"].calls if c[0] == "documents.search"]
    assert len(searches) == 3
    # Every clause is sent alone - Pinecone rejects a vector clause combined with others.
    assert all(len(c[1]["score_by"]) == 1 for c in searches)
    # 'shared' appears in all three result sets, so fusion should rank it first.
    assert result["hits"][0]["id"] == "shared"


def test_auto_mode_picks_hybrid_when_available(wired, stub_embedders):
    backend = _register(wired, "multi", {"emb": DENSE_FIELD, "body": FTS_FIELD})
    assert backend.search("multi", "ns", query="x")["mode"] == "hybrid"


def test_auto_mode_on_fts_only_index(wired):
    backend = _register(wired, "fts", {"body": FTS_FIELD})
    assert backend.search("fts", "ns", query="x")["mode"] == "text"


def test_integrated_index_refuses_document_search(wired):
    backend = _register(wired, "integrated", {"chunk_text": {"type": "semantic_text"}})
    with pytest.raises(CapabilityError, match="pinecone_search_records"):
        backend.search("integrated", "ns", mode="dense", query="x")


def test_search_filters_out_expired_records_by_default(wired):
    backend = _register(wired, "fts", {"body": FTS_FIELD})
    backend.search("fts", "ns", mode="text", query="x", filter={"year": {"$eq": 2026}})
    sent = wired["index"].calls[-1][1]["filter"]
    assert "$and" in sent
    assert {"year": {"$eq": 2026}} in sent["$and"]


def test_exclude_expired_false_leaves_the_filter_alone(wired):
    backend = _register(wired, "fts", {"body": FTS_FIELD})
    backend.search("fts", "ns", mode="text", query="x", filter={"a": 1}, exclude_expired=False)
    assert wired["index"].calls[-1][1]["filter"] == {"a": 1}


def test_upsert_embeds_dense_and_sparse_from_one_text_field(wired, stub_embedders):
    backend = _register(wired, "multi", {"emb": DENSE_FIELD, "sp": SPARSE_FIELD})
    report = backend.upsert_documents(
        "multi",
        "ns",
        [{"_id": "d1", "body": "hello"}, {"_id": "d2", "body": "world"}],
        embed_source_field="body",
        ttl_seconds=3600,
    )
    assert report["embedded_dense"] == 2
    assert report["embedded_sparse"] == 2
    sent = wired["index"].calls[-1][1]["documents"]
    assert len(sent[0]["emb"]) == 8
    assert sent[0]["sp"]["indices"] == [1, 2]
    assert sent[0]["vtb_expires_at"] > 0


def test_upsert_leaves_precomputed_vectors_alone(wired, stub_embedders):
    backend = _register(wired, "dense", {"emb": DENSE_FIELD})
    report = backend.upsert_documents(
        "dense",
        "ns",
        [{"_id": "d1", "body": "x", "emb": [0.9] * 8}],
        embed_source_field="body",
    )
    assert report["embedded_dense"] == 0
    assert wired["index"].calls[-1][1]["documents"][0]["emb"] == [0.9] * 8


def test_upsert_rejects_a_dimension_mismatch_before_sending(wired, monkeypatch):
    from vectortoolbox.backends.pinecone import backend as backend_mod

    monkeypatch.setattr(backend_mod, "get_dense_embedder", lambda *a, **k: StubEmbedder(dim=1536))
    backend = _register(wired, "dense", {"emb": DENSE_FIELD})
    with pytest.raises(CapabilityError, match="expects dimension 8"):
        backend.upsert_documents(
            "dense", "ns", [{"_id": "d1", "body": "x"}], embed_source_field="body"
        )
    assert not [c for c in wired["index"].calls if "upsert" in c[0]]


def test_delete_all_requires_confirmation(wired):
    backend = _register(wired, "fts", {"body": FTS_FIELD})
    with pytest.raises(ConfirmationRequired):
        backend.delete_documents("fts", "ns", delete_all=True)


def test_delete_needs_some_target(wired):
    backend = _register(wired, "fts", {"body": FTS_FIELD})
    with pytest.raises(CapabilityError, match="ids, a filter, or delete_all"):
        backend.delete_documents("fts", "ns")


def test_purge_expired_deletes_by_ttl_filter(wired):
    backend = _register(wired, "fts", {"body": FTS_FIELD})
    backend.purge_expired("fts", "ns", confirm=True)
    sent = wired["index"].calls[-1][1]["filter"]
    assert {"vtb_expires_at": {"$exists": True}} in sent["$and"]


def test_sample_metadata_profiles_fields(wired):
    wired["index"].records = [
        {"_id": "1", "category": "tech", "year": 2026, "body": "a"},
        {"_id": "2", "category": "news", "year": 2025, "body": "b"},
    ]
    backend = _register(wired, "fts", {"body": FTS_FIELD})
    profile = backend.sample_metadata("fts", "ns")
    assert profile["fields"]["category"]["types"] == ["str"]
    assert profile["fields"]["year"]["types"] == ["int"]
    assert profile["fields"]["body"]["declared_in_schema"] is True
    assert profile["fields"]["category"]["declared_in_schema"] is False


def test_upsert_rejects_a_document_with_no_id(wired):
    backend = _register(wired, "fts", {"body": FTS_FIELD})
    with pytest.raises(CapabilityError, match="no _id"):
        backend.upsert_documents("fts", "ns", [{"body": "x"}])


def test_upsert_rejects_a_reserved_field_name(wired):
    """One bad field name fails the whole request server-side, so catch it here."""
    backend = _register(wired, "fts", {"body": FTS_FIELD})
    with pytest.raises(CapabilityError, match="leading _ or"):
        backend.upsert_documents("fts", "ns", [{"_id": "1", "body": "x", "_mine": 1}])


def test_upsert_rejects_a_metadata_only_document(wired):
    backend = _register(wired, "fts", {"body": FTS_FIELD})
    with pytest.raises(CapabilityError, match="only metadata and no schema field"):
        backend.upsert_documents("fts", "ns", [{"_id": "1", "category": "tech"}])


def test_upsert_reports_every_offender_at_once(wired):
    backend = _register(wired, "fts", {"body": FTS_FIELD})
    with pytest.raises(CapabilityError) as exc:
        backend.upsert_documents(
            "fts", "ns",
            [{"body": "x"}, {"_id": "2", "category": "t"}, {"_id": "3", "body": "y", "$bad": 1}],
        )
    message = str(exc.value)
    assert "no _id" in message and "only metadata" in message and "$bad" in message


def test_top_k_bounds_are_checked(wired):
    backend = _register(wired, "fts", {"body": FTS_FIELD})
    with pytest.raises(CapabilityError, match="between 1 and 10000"):
        backend.search("fts", "ns", mode="text", query="x", top_k=20000)


# ---------------------------------------------------------------------------
# Vectors-API indexes (the shape of every index in a real legacy project)
# ---------------------------------------------------------------------------
def test_auto_mode_on_a_vectors_index_uses_the_vectors_api(wired, stub_embedders):
    backend = _register(wired, "olivet-prod", LEGACY_VECTOR_FIELDS)
    result = backend.search("olivet-prod", "ns", query="refund policy", top_k=3)

    assert result["api"] == "vectors"
    assert result["scored_by"] == ["dense", "sparse"]
    assert result["fusion"] == "server-side (single request)"

    # It must go through query(), never documents.search().
    assert [c[0] for c in wired["index"].calls] == ["query"]
    sent = wired["index"].calls[0][1]
    assert len(sent["vector"]) == 8
    assert sent["sparse_vector"]["indices"] == [1, 2]


def test_vectors_index_returns_metadata_as_fields(wired, stub_embedders):
    backend = _register(wired, "olivet-prod", LEGACY_VECTOR_FIELDS)
    hits = backend.search("olivet-prod", "ns", query="x")["hits"]
    assert hits[0]["id"] == "v1"
    assert hits[0]["fields"] == {"category": "policy"}


def test_dense_only_mode_on_a_vectors_index_sends_no_sparse_vector(wired, stub_embedders):
    backend = _register(wired, "olivet-prod", LEGACY_VECTOR_FIELDS)
    backend.search("olivet-prod", "ns", mode="dense", query="x")
    assert wired["index"].calls[0][1].get("sparse_vector") is None


def test_full_text_search_on_a_vectors_index_is_refused_with_a_reason(wired):
    backend = _register(wired, "olivet-prod", LEGACY_VECTOR_FIELDS)
    with pytest.raises(CapabilityError) as exc:
        backend.search("olivet-prod", "ns", mode="text", query="refund")
    message = str(exc.value)
    assert "declares no full-text field" in message
    assert "new index" in message


def test_document_upsert_to_a_vectors_index_points_at_the_right_tool(wired, stub_embedders):
    backend = _register(wired, "olivet-prod", LEGACY_VECTOR_FIELDS)
    with pytest.raises(CapabilityError, match="pinecone_upsert_vectors"):
        backend.upsert_documents("olivet-prod", "ns", [{"_id": "1", "body": "x"}])
    assert not wired["index"].calls


def test_document_update_to_a_vectors_index_points_at_the_right_tool(wired):
    backend = _register(wired, "olivet-prod", LEGACY_VECTOR_FIELDS)
    with pytest.raises(CapabilityError, match="pinecone_update_vector"):
        backend.update_documents("olivet-prod", "ns", set_fields={"a": 1})


def test_vectors_index_still_filters_expired_records(wired, stub_embedders):
    backend = _register(wired, "olivet-prod", LEGACY_VECTOR_FIELDS)
    backend.search("olivet-prod", "ns", query="x", filter={"category": {"$eq": "policy"}})
    sent = wired["index"].calls[0][1]["filter"]
    assert {"category": {"$eq": "policy"}} in sent["$and"]


def test_sample_metadata_on_a_vectors_index_lists_ids_before_fetching(wired):
    """The Vectors API fetches strictly by id, so "show me some records" has to
    become "find ids, then fetch them" - passing an empty id list is an error."""
    wired["index"].records = [
        {"_id": "std-1", "standard": "Standard 3", "title": "Personal care", "version": 2},
        {"_id": "std-2", "standard": "Standard 5", "title": "Service environment", "version": 2},
    ]
    backend = _register(wired, "demo", LEGACY_VECTOR_FIELDS)
    profile = backend.sample_metadata("demo", "ns", sample_size=5)

    steps = [c[0] for c in wired["index"].calls]
    assert steps == ["list", "fetch"]
    assert wired["index"].calls[1][1]["ids"] == ["std-1", "std-2"]
    assert profile["sampled"] == 2
    assert profile["fields"]["standard"]["types"] == ["str"]
    assert profile["fields"]["version"]["types"] == ["int"]


def test_fetch_with_a_filter_on_a_vectors_index_uses_fetch_by_metadata(wired):
    wired["index"].records = [{"_id": "std-1", "standard": "Standard 3"}]
    backend = _register(wired, "demo", LEGACY_VECTOR_FIELDS)
    docs = backend.fetch_documents("demo", "ns", filter={"standard": {"$eq": "Standard 3"}})
    assert [c[0] for c in wired["index"].calls] == ["fetch_by_metadata"]
    assert docs[0]["_id"] == "std-1"


def test_fetch_on_an_empty_vectors_namespace_returns_nothing(wired):
    wired["index"].records = []
    backend = _register(wired, "demo", LEGACY_VECTOR_FIELDS)
    assert backend.fetch_documents("demo", "ns") == []


def test_vector_values_are_not_returned_unless_asked_for(wired):
    """A 3072-dim vector per record turns a 10-record fetch into megabytes of
    floats nobody asked for, so values are dropped by default."""
    wired["index"].records = [{"_id": "std-1", "standard": "Standard 3"}]
    backend = _register(wired, "demo", LEGACY_VECTOR_FIELDS)

    plain = backend.fetch_documents("demo", "ns", ids=["std-1"])
    assert plain == [{"_id": "std-1", "standard": "Standard 3"}]
    assert "values" not in plain[0]

    with_values = backend.fetch_documents(
        "demo", "ns", ids=["std-1"], include_fields=["values"]
    )
    assert with_values[0]["values"] == [0.5, 0.5]
