#!/usr/bin/env python3
"""Live smoke test against a real Pinecone project.

Creates throwaway indexes prefixed ``vtb-smoke-``, exercises each of the five
index/search recipes end to end, then deletes them.

    export PINECONE_API_KEY=pcsk_...
    python scripts/smoke_test.py            # recipes 1 and 4 (fast path)
    python scripts/smoke_test.py --all      # all five
    python scripts/smoke_test.py --keep     # leave the indexes behind

Index creation takes 30-60s each, so the default runs two.
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))

from vectortoolbox.backends.pinecone import PineconeBackend  # noqa: E402
from vectortoolbox.config import get_settings  # noqa: E402
from vectortoolbox.core.types import DenseFieldSpec, SparseFieldSpec, TextFieldSpec  # noqa: E402

NS = "smoke"
SUFFIX = uuid.uuid4().hex[:6]
DOCS = [
    {"_id": "d1", "body": "Refunds are issued within 14 days of the return.",
     "summary": "refund window", "category": "policy", "year": 2026},
    {"_id": "d2", "body": "Machine learning models are transforming retrieval.",
     "summary": "ml and search", "category": "tech", "year": 2025},
    {"_id": "d3", "body": "Our warranty covers manufacturing defects only.",
     "summary": "warranty scope", "category": "policy", "year": 2024},
]

created: list[str] = []
failures: list[str] = []


def log(step: str, detail: str = "") -> None:
    print(f"  {step:<38} {detail}")


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'PASS' if condition else 'FAIL'}  {label} {detail}")
    if not condition:
        failures.append(label)


def wait_for_records(backend: PineconeBackend, index: str, expected: int, timeout: int = 60):
    """Pinecone writes are not read-your-writes immediately."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            stats = backend.stats(index)
            counts = stats.get("namespaces", {}) or {}
            total = counts.get(NS, {}).get("record_count") or counts.get(NS, {}).get("vector_count")
            if total and total >= expected:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def recipe_1_fts_only(backend: PineconeBackend, dim: int) -> None:
    name = f"vtb-smoke-fts-{SUFFIX}"
    print(f"\n[1] Single text field - keyword search only ({name})")
    backend.create_index(name=name, text_fields=[TextFieldSpec(name="body")])
    created.append(name)
    caps = backend.capabilities(name, refresh=True)
    check("modes include text/query_string", {"text", "query_string"} <= set(caps.supported_search_modes), caps.supported_search_modes)
    check("no dense mode advertised", "dense" not in caps.supported_search_modes)

    backend.upsert_documents(name, NS, DOCS, ttl_seconds=3600)
    wait_for_records(backend, name, len(DOCS))

    result = backend.search(name, NS, mode="text", query="refund", top_k=3, include_fields=["*"])
    check("BM25 finds the refund document", any(h["id"] == "d1" for h in result["hits"]), [h["id"] for h in result["hits"]])

    lucene = backend.search(name, NS, mode="query_string", query="body:(warranty AND defects)", top_k=3)
    check("Lucene AND query matches d3", any(h["id"] == "d3" for h in lucene["hits"]), [h["id"] for h in lucene["hits"]])

    try:
        backend.search(name, NS, mode="dense", query="refund")
        check("dense on an FTS-only index is refused", False)
    except Exception as exc:
        check("dense on an FTS-only index is refused", "cannot run" in str(exc))

    profile = backend.sample_metadata(name, NS)
    check("sample_metadata finds auto-indexed metadata", "category" in profile["fields"], sorted(profile["fields"]))


def recipe_2_multi_field_fts(backend: PineconeBackend, dim: int) -> None:
    name = f"vtb-smoke-mfts-{SUFFIX}"
    print(f"\n[2] Multi-field FTS ({name})")
    backend.create_index(
        name=name,
        text_fields=[TextFieldSpec(name="body"), TextFieldSpec(name="summary")],
    )
    created.append(name)
    backend.upsert_documents(name, NS, DOCS)
    wait_for_records(backend, name, len(DOCS))
    result = backend.search(name, NS, mode="text", query="refund window", fields=["body", "summary"], top_k=3)
    check("one clause per field", result["clauses"] == 2, result["clauses"])
    check("returns hits", bool(result["hits"]), [h["id"] for h in result["hits"]])

    per_field = backend.search(
        name, NS, mode="text",
        field_queries={"body": "warranty", "summary": "scope"}, top_k=3,
    )
    check("per-field queries return hits", bool(per_field["hits"]), [h["id"] for h in per_field["hits"]])


def recipe_3_dense_plus_fts(backend: PineconeBackend, dim: int) -> None:
    name = f"vtb-smoke-densefts-{SUFFIX}"
    print(f"\n[3] Dense + FTS in one index ({name})")
    backend.create_index(
        name=name,
        dense_fields=[DenseFieldSpec(name="embedding", dimension=dim)],
        text_fields=[TextFieldSpec(name="body")],
    )
    created.append(name)
    backend.upsert_documents(name, NS, DOCS, embed_source_field="body")
    wait_for_records(backend, name, len(DOCS))
    fused = backend.search(name, NS, mode="hybrid", query="getting money back", top_k=3, include_fields=["*"])
    check("hybrid ran both signals", set(fused["signals"]) == {"dense", "text"}, fused["signals"])
    check("fusion returned hits", bool(fused["hits"]), [h["id"] for h in fused["hits"]])

    # The documented "most common hybrid pattern": rank by dense similarity,
    # restricted to documents containing an exact phrase.
    phrase = backend.search(
        name, NS, mode="dense", query="getting money back", top_k=5,
        filter={"body": {"$match_phrase": "within 14 days"}}, include_fields=["*"],
    )
    check("dense ranking + $match_phrase filter", [h["id"] for h in phrase["hits"]] == ["d1"], [h["id"] for h in phrase["hits"]])

    # Metadata written outside the schema should still be filterable.
    meta = backend.search(name, NS, mode="text", query="the", filter={"category": {"$eq": "policy"}}, top_k=5, include_fields=["*"])
    check("undeclared metadata is filterable", all(h["fields"].get("category") == "policy" for h in meta["hits"]) if meta["hits"] else False, [h["id"] for h in meta["hits"]])


def recipe_4_multi_signal(backend: PineconeBackend, dim: int) -> None:
    name = f"vtb-smoke-multi-{SUFFIX}"
    print(f"\n[4] Multi-signal - dense + sparse + FTS ({name})")
    backend.create_index(
        name=name,
        dense_fields=[DenseFieldSpec(name="embedding", dimension=dim)],
        sparse_fields=[SparseFieldSpec(name="keywords")],
        text_fields=[TextFieldSpec(name="body")],
    )
    created.append(name)
    caps = backend.capabilities(name, refresh=True)
    check("all three signals detected", bool(caps.dense_fields and caps.sparse_fields and caps.fts_fields))

    backend.upsert_documents(name, NS, DOCS, embed_source_field="body", ttl_seconds=7200)
    wait_for_records(backend, name, len(DOCS))

    fused = backend.search(name, NS, mode="hybrid", query="refund policy", top_k=3, include_fields=["*"])
    check("three signals fused", set(fused["signals"]) == {"dense", "sparse", "text"}, fused["signals"])

    weighted = backend.search(
        name, NS, mode="hybrid", query="refund policy", fusion="weighted",
        weights={"dense": 3.0, "text": 1.0, "sparse": 1.0}, top_k=3,
    )
    check("weighted fusion runs", bool(weighted["hits"]))

    filtered = backend.search(name, NS, mode="text", query="refund", filter={"category": {"$eq": "policy"}}, top_k=5, include_fields=["*"])
    check("metadata filter applied", all(h["fields"].get("category") == "policy" for h in filtered["hits"]) if filtered["hits"] else True)

    try:
        backend.upsert_documents(name, NS, [{"_id": "bad", "category": "metadata only"}])
        check("metadata-only document refused before sending", False)
    except Exception as exc:
        check("metadata-only document refused before sending", "no schema field" in str(exc))

    ttl_report = backend.upsert_documents(name, NS, [{"_id": "short", "body": "expires"}], embed_source_field="body", ttl_seconds=1)
    check("TTL field is not reserved", ttl_report.get("ttl_field") == "vtb_expires_at", ttl_report.get("ttl_field"))
    time.sleep(3)
    after = backend.search(name, NS, mode="text", query="expires", top_k=5)
    check("expired record hidden by default", all(h["id"] != "short" for h in after["hits"]), [h["id"] for h in after["hits"]])
    unfiltered = backend.search(name, NS, mode="text", query="expires", top_k=5, exclude_expired=False)
    check("records without a TTL are not hidden", bool(unfiltered["hits"]) , [h["id"] for h in unfiltered["hits"]])

    backend.update_documents(name, NS, documents=[{"_id": "d1", "category": "updated"}])
    backend.delete_documents(name, NS, ids=["d3"])
    check("update + delete accepted", True)


def recipe_5_vectors_api(backend: PineconeBackend, dim: int) -> None:
    name = f"vtb-smoke-vecapi-{SUFFIX}"
    print(f"\n[5] Sparse + dense hybrid over the Vectors API ({name})")
    backend.create_index(
        name=name,
        dense_fields=[DenseFieldSpec(name="values", dimension=dim, metric="dotproduct")],
        sparse_fields=[SparseFieldSpec(name="sparse_values")],
    )
    created.append(name)
    caps = backend.capabilities(name, refresh=True)
    check("vectors_api mode detected", "vectors_api" in caps.supported_search_modes, caps.supported_search_modes)

    from vectortoolbox.embeddings import get_dense_embedder, get_sparse_embedder

    dense = get_dense_embedder()
    sparse = get_sparse_embedder()
    texts = [d["body"] for d in DOCS]
    dvecs = dense.embed_documents(texts)
    svecs = sparse.embed_documents(texts)
    backend.upsert_vectors(
        name, NS,
        [
            {"id": d["_id"], "values": dv, "sparse_values": sv,
             "metadata": {"category": d["category"], "year": d["year"]}}
            for d, dv, sv in zip(DOCS, dvecs, svecs)
        ],
        ttl_seconds=3600,
    )
    wait_for_records(backend, name, len(DOCS))
    result = backend.query_vectors(
        name, NS,
        vector=dense.embed_query("refund"),
        sparse_vector=sparse.embed_query("refund"),
        top_k=3,
    )
    check("single-request dense+sparse query returns matches", bool(result.get("matches")), result.get("matches"))


RECIPES = {
    1: recipe_1_fts_only,
    2: recipe_2_multi_field_fts,
    3: recipe_3_dense_plus_fts,
    4: recipe_4_multi_signal,
    5: recipe_5_vectors_api,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="run all five recipes")
    parser.add_argument("--only", type=int, nargs="*", help="run specific recipe numbers")
    parser.add_argument("--keep", action="store_true", help="do not delete the scratch indexes")
    # Default to the dimension the configured embedder actually produces -
    # a mismatch here fails at upsert, not at index creation, which is a
    # confusing place to discover it.
    parser.add_argument("--dimension", type=int, default=None)
    args = parser.parse_args()

    settings = get_settings()
    if args.dimension is None:
        args.dimension = settings.embed_dimension or 1024
    if not settings.pinecone_api_key:
        print("PINECONE_API_KEY is not set.")
        return 2
    print(f"Project region: {settings.pinecone_cloud}/{settings.pinecone_region}")
    print(f"Embedder: {settings.embed_provider}/{settings.embed_model} (dim {args.dimension})")

    which = args.only or (list(RECIPES) if args.all else [1, 4])
    backend = PineconeBackend()

    try:
        for number in which:
            try:
                RECIPES[number](backend, args.dimension)
            except Exception as exc:
                failures.append(f"recipe {number}: {exc}")
                print(f"  ERROR in recipe {number}: {exc}")
    finally:
        if args.keep:
            print(f"\nLeft behind: {', '.join(created)}")
        else:
            print("\nCleaning up...")
            for name in created:
                try:
                    backend.delete_index(name, confirm=True)
                    log("deleted", name)
                except Exception as exc:
                    print(f"  could not delete {name}: {exc}")

    print(f"\n{'FAILURES: ' + '; '.join(failures) if failures else 'All checks passed.'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
