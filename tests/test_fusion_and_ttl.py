from __future__ import annotations

from vectortoolbox.backends.pinecone import ttl as ttl_mod
from vectortoolbox.core.fusion import fuse
from vectortoolbox.core.types import Hit


def _hits(*pairs):
    return [Hit(id=i, score=s) for i, s in pairs]


def test_rrf_rewards_documents_ranked_by_several_signals():
    result = fuse(
        {
            "dense": _hits(("a", 0.9), ("shared", 0.5)),
            "text": _hits(("b", 12.0), ("shared", 4.0)),
        },
        strategy="rrf",
        top_k=3,
    )
    # 'shared' is second in both lists; RRF should lift it above either winner.
    assert result[0].id == "shared"
    assert set(result[0].signals) == {"dense", "text"}


def test_rrf_is_scale_agnostic():
    """BM25 scores dwarf cosine scores; RRF must not care."""
    result = fuse(
        {"dense": _hits(("a", 0.02)), "text": _hits(("b", 900.0))},
        strategy="rrf",
        top_k=2,
    )
    assert {h.id for h in result} == {"a", "b"}
    assert abs(result[0].score - result[1].score) < 1e-9


def test_weighted_fusion_respects_weights():
    result = fuse(
        {"dense": _hits(("a", 1.0), ("b", 0.0)), "text": _hits(("b", 1.0), ("a", 0.0))},
        strategy="weighted",
        weights={"dense": 3.0, "text": 1.0},
        top_k=2,
    )
    assert result[0].id == "a"


def test_single_signal_passes_through():
    assert [h.id for h in fuse({"dense": _hits(("a", 1.0))}, top_k=5)] == ["a"]


def test_ttl_stamp_is_a_no_op_without_a_ttl():
    """No TTL means no extra field - $exists in the filter handles absence."""
    assert ttl_mod.stamp([{"_id": "1"}], None) == [{"_id": "1"}]


def test_ttl_stamp_adds_an_epoch_when_asked():
    stamped = ttl_mod.stamp([{"_id": "1"}], 60)
    assert stamped[0][ttl_mod.TTL_FIELD] > ttl_mod.now()


def test_ttl_stamp_respects_a_per_record_override():
    stamped = ttl_mod.stamp([{"_id": "1", ttl_mod.TTL_FIELD: 123}], 60)
    assert stamped[0][ttl_mod.TTL_FIELD] == 123


def test_ttl_field_name_is_not_reserved():
    """Pinecone rejects any field starting with _ or $, failing the whole upsert."""
    assert not ttl_mod.TTL_FIELD.startswith(("_", "$"))
    assert len(ttl_mod.TTL_FIELD.encode()) <= 64


def test_live_filter_does_not_hide_records_without_a_ttl():
    clause = ttl_mod.live_filter(None)
    assert clause["$or"] == [
        {ttl_mod.TTL_FIELD: {"$exists": False}},
        {ttl_mod.TTL_FIELD: {"$gt": ttl_mod.now()}},
    ]


def test_live_filter_combines_with_caller_filter():
    combined = ttl_mod.live_filter({"category": {"$eq": "tech"}})
    assert combined["$and"][0] == {"category": {"$eq": "tech"}}
    assert "$or" in combined["$and"][1]


def test_expired_filter_only_matches_records_that_have_a_ttl():
    clause = ttl_mod.expired_filter()
    assert {ttl_mod.TTL_FIELD: {"$exists": True}} in clause["$and"]
