"""The schema layer is what keeps searches honest, so it gets the most tests."""

from __future__ import annotations

import pytest

from conftest import (
    DENSE_FIELD,
    FTS_FIELD,
    LEGACY_VECTOR_FIELDS,
    SPARSE_FIELD,
    index_model,
)
from vectortoolbox.backends.pinecone.schema import (
    build_schema,
    capabilities_from_index_model,
    require_mode,
    resolve_field,
)
from vectortoolbox.core.types import DenseFieldSpec, SparseFieldSpec, TextFieldSpec
from vectortoolbox.errors import CapabilityError


def test_recipe_1_single_fts_field():
    schema = build_schema(text_fields=[TextFieldSpec(name="body")])
    assert schema["fields"]["body"]["type"] == "string"
    assert schema["fields"]["body"]["full_text_search"]["language"] == "en"


def test_recipe_2_multi_field_fts():
    schema = build_schema(
        text_fields=[TextFieldSpec(name="body"), TextFieldSpec(name="summary", stemming=True)]
    )
    assert set(schema["fields"]) == {"body", "summary"}
    assert schema["fields"]["summary"]["full_text_search"]["stemming"] is True


def test_recipe_3_dense_plus_fts():
    schema = build_schema(
        dense_fields=[DenseFieldSpec(name="embedding", dimension=1024)],
        text_fields=[TextFieldSpec(name="body")],
    )
    assert schema["fields"]["embedding"]["dimension"] == 1024
    assert schema["fields"]["embedding"]["metric"] == "cosine"


def test_recipe_4_multi_signal():
    schema = build_schema(
        dense_fields=[DenseFieldSpec(name="frame_embedding", dimension=1024)],
        sparse_fields=[SparseFieldSpec(name="caption_sparse")],
        text_fields=[TextFieldSpec(name="transcript")],
    )
    assert len(schema["fields"]) == 3


def test_filter_only_string_field_is_rejected():
    """Pinecone only accepts searchable fields in a schema - fail early and say so."""
    with pytest.raises(CapabilityError, match="only declare searchable fields"):
        build_schema(text_fields=[TextFieldSpec(name="category", full_text_search=False)])


def test_empty_schema_is_rejected():
    with pytest.raises(CapabilityError, match="at least one searchable field"):
        build_schema()


def test_capabilities_multi_signal_index():
    caps = capabilities_from_index_model(
        index_model("multi", {"emb": DENSE_FIELD, "sp": SPARSE_FIELD, "body": FTS_FIELD})
    )
    assert caps.dense_fields == ["emb"]
    assert caps.sparse_fields == ["sp"]
    assert caps.fts_fields == ["body"]
    assert caps.dimensions == {"emb": 8}
    assert set(caps.supported_search_modes) >= {"dense", "sparse", "text", "query_string", "hybrid"}
    # Three signals in one schema is not a single-vector index.
    assert "vectors_api" not in caps.supported_search_modes


def test_capabilities_recipe_5_single_vector_index():
    caps = capabilities_from_index_model(
        index_model("hybrid-single", {"emb": DENSE_FIELD, "sp": SPARSE_FIELD})
    )
    assert "vectors_api" in caps.supported_search_modes
    assert "text" not in caps.supported_search_modes


def test_capabilities_integrated_index():
    caps = capabilities_from_index_model(
        index_model("integrated", {"chunk_text": {"type": "semantic_text", "dimension": 1024}})
    )
    assert caps.api == "integrated"
    assert caps.semantic_text_fields == ["chunk_text"]


def test_non_searchable_field_is_reported_as_filterable():
    caps = capabilities_from_index_model(
        index_model("mixed", {"body": FTS_FIELD, "year": {"type": "integer"}})
    )
    assert caps.filterable_fields == ["year"]


def test_require_mode_error_names_what_is_available():
    caps = capabilities_from_index_model(index_model("dense-only", {"emb": DENSE_FIELD}))
    with pytest.raises(CapabilityError) as exc:
        require_mode(caps, "text")
    message = str(exc.value)
    assert "cannot run a 'text' search" in message
    assert "dense" in message
    assert "creation time" in message


def test_resolve_field_rejects_unknown_name():
    caps = capabilities_from_index_model(index_model("i", {"emb": DENSE_FIELD}))
    assert resolve_field(caps, "dense", None) == "emb"
    with pytest.raises(CapabilityError, match="no dense field named 'nope'"):
        resolve_field(caps, "dense", "nope")


def test_at_most_one_dense_field_per_index():
    with pytest.raises(CapabilityError, match="at most 1 dense_vector field"):
        build_schema(
            dense_fields=[
                DenseFieldSpec(name="a", dimension=8),
                DenseFieldSpec(name="b", dimension=8),
            ]
        )


def test_at_most_one_sparse_field_per_index():
    with pytest.raises(CapabilityError, match="at most 1 sparse_vector field"):
        build_schema(
            sparse_fields=[SparseFieldSpec(name="a"), SparseFieldSpec(name="b")]
        )


def test_at_most_one_hundred_fts_fields():
    fields = [TextFieldSpec(name=f"f{i}") for i in range(101)]
    with pytest.raises(CapabilityError, match="at most 100 full-text string fields"):
        build_schema(text_fields=fields)


def test_reserved_field_name_prefixes_are_rejected():
    for bad in ("_body", "$body"):
        with pytest.raises(CapabilityError, match="reserve"):
            build_schema(text_fields=[TextFieldSpec(name=bad)])


def test_field_names_are_capped_at_64_bytes():
    with pytest.raises(CapabilityError, match="longer than 64 bytes"):
        build_schema(text_fields=[TextFieldSpec(name="f" * 65)])


def test_system_named_vector_fields_mean_a_vectors_api_index():
    """Pinecone shows an older Vectors index through the schema API using
    system field names. Reading it with the Documents API fails, so the
    classification has to happen here, not at query time."""
    caps = capabilities_from_index_model(index_model("olivet-prod", LEGACY_VECTOR_FIELDS))
    assert caps.api == "vectors"
    assert caps.dense_fields == ["_values"]
    assert caps.sparse_fields == ["_sparse_values"]
    assert "vectors_api" in caps.supported_search_modes
    assert "text" not in caps.supported_search_modes
    assert any("pinecone_query_vectors" in note for note in caps.notes)


def test_user_named_dense_and_sparse_stays_a_document_index():
    """The same two signals under declared names are a real schema index."""
    caps = capabilities_from_index_model(
        index_model("declared", {"embedding": DENSE_FIELD, "keywords": SPARSE_FIELD})
    )
    assert caps.api == "documents"


def test_a_text_field_disqualifies_the_legacy_reading():
    fields = dict(LEGACY_VECTOR_FIELDS)
    fields["body"] = FTS_FIELD
    caps = capabilities_from_index_model(index_model("mixed", fields))
    assert caps.api == "documents"


def test_legacy_index_does_not_advertise_client_side_fusion():
    caps = capabilities_from_index_model(index_model("legacy", LEGACY_VECTOR_FIELDS))
    assert not any("RRF" in note for note in caps.notes)
    assert any("one server-side request" in note for note in caps.notes)
