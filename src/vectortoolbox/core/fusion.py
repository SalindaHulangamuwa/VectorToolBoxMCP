"""Client-side result fusion for hybrid search.

Pinecone's Documents API allows several ``text``/``query_string`` clauses in
one request but requires a ``dense_vector`` or ``sparse_vector`` clause to
stand alone. So a true multi-signal search (dense + sparse + BM25) is run as
several searches and merged here.

Two strategies:

* ``rrf`` - Reciprocal Rank Fusion. Score-scale agnostic, which matters
  because BM25 scores and cosine similarities are not comparable. This is the
  right default.
* ``weighted`` - min/max normalise each signal to [0, 1] then take a weighted
  sum. Use when you want to dial one signal up explicitly.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from .types import Hit


def _normalise(scores: Sequence[float]) -> list[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-12:
        return [1.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def reciprocal_rank_fusion(
    result_sets: dict[str, list[Hit]],
    *,
    k: int = 60,
    weights: dict[str, float] | None = None,
    top_k: int = 10,
) -> list[Hit]:
    weights = weights or {}
    merged: dict[str, Hit] = {}
    totals: dict[str, float] = {}

    for signal, hits in result_sets.items():
        weight = weights.get(signal, 1.0)
        for rank, hit in enumerate(hits, start=1):
            contribution = weight / (k + rank)
            totals[hit.id] = totals.get(hit.id, 0.0) + contribution
            existing = merged.get(hit.id)
            if existing is None:
                merged[hit.id] = Hit(
                    id=hit.id, score=0.0, fields=dict(hit.fields), signals={signal: hit.score}
                )
            else:
                existing.signals[signal] = hit.score
                # Keep the richest field payload we have seen.
                for key, value in hit.fields.items():
                    existing.fields.setdefault(key, value)

    for doc_id, total in totals.items():
        merged[doc_id].score = total

    ordered = sorted(merged.values(), key=lambda h: h.score, reverse=True)
    return ordered[:top_k]


def weighted_fusion(
    result_sets: dict[str, list[Hit]],
    *,
    weights: dict[str, float] | None = None,
    top_k: int = 10,
) -> list[Hit]:
    weights = weights or {}
    merged: dict[str, Hit] = {}
    totals: dict[str, float] = {}

    for signal, hits in result_sets.items():
        weight = weights.get(signal, 1.0)
        normalised = _normalise([h.score for h in hits])
        for hit, value in zip(hits, normalised):
            totals[hit.id] = totals.get(hit.id, 0.0) + weight * value
            existing = merged.get(hit.id)
            if existing is None:
                merged[hit.id] = Hit(
                    id=hit.id, score=0.0, fields=dict(hit.fields), signals={signal: hit.score}
                )
            else:
                existing.signals[signal] = hit.score
                for key, val in hit.fields.items():
                    existing.fields.setdefault(key, val)

    for doc_id, total in totals.items():
        merged[doc_id].score = total

    return sorted(merged.values(), key=lambda h: h.score, reverse=True)[:top_k]


def fuse(
    result_sets: dict[str, list[Hit]],
    *,
    strategy: str = "rrf",
    weights: dict[str, float] | None = None,
    top_k: int = 10,
    rrf_k: int = 60,
) -> list[Hit]:
    non_empty = {k: v for k, v in result_sets.items() if v}
    if not non_empty:
        return []
    if len(non_empty) == 1:
        only = next(iter(non_empty.values()))
        return only[:top_k]
    if strategy == "weighted":
        return weighted_fusion(non_empty, weights=weights, top_k=top_k)
    return reciprocal_rank_fusion(non_empty, k=rrf_k, weights=weights, top_k=top_k)
