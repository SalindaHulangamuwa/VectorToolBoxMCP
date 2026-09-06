"""Client-side TTL.

Pinecone has no server-side record expiry, so the toolbox implements TTL as an
explicit convention rather than pretending the database does it:

* a record written with a TTL carries ``vtb_expires_at``, an epoch-seconds
  integer, as ordinary metadata;
* a record written without one carries nothing extra;
* searches exclude expired records by default, using ``$exists`` so that
  records with no TTL - including records written by something other than this
  server - are never hidden;
* ``pinecone_purge_expired`` actually deletes the lapsed records, and is the
  only thing that reclaims storage.

The field name matters: Pinecone rejects any field whose name starts with
``_`` (reserved for system fields like ``_id`` and ``_score``) or ``$``
(reserved for filter operators), and one invalid document fails the whole
upsert request. Hence ``vtb_expires_at`` rather than ``_expires_at``.
"""

from __future__ import annotations

import time
from typing import Any, Iterable

TTL_FIELD = "vtb_expires_at"


def now() -> int:
    return int(time.time())


def expiry_from_ttl(ttl_seconds: int | None) -> int | None:
    if ttl_seconds is None or ttl_seconds <= 0:
        return None
    return now() + int(ttl_seconds)


def stamp(records: Iterable[dict[str, Any]], ttl_seconds: int | None) -> list[dict[str, Any]]:
    """Add ``vtb_expires_at`` to records, when a TTL was asked for.

    A record that already sets the field keeps its own value, so a per-record
    expiry can be mixed into a batch that shares a default.
    """
    expires_at = expiry_from_ttl(ttl_seconds)
    out = []
    for record in records:
        record = dict(record)
        if expires_at is not None:
            record.setdefault(TTL_FIELD, expires_at)
        out.append(record)
    return out


def live_filter(existing: dict[str, Any] | None = None) -> dict[str, Any]:
    """Combine a caller's filter with 'has not expired'.

    ``$exists: false`` is the important half: without it, every record that
    carries no TTL would be filtered away.
    """
    fresh = {
        "$or": [
            {TTL_FIELD: {"$exists": False}},
            {TTL_FIELD: {"$gt": now()}},
        ]
    }
    if not existing:
        return fresh
    return {"$and": [existing, fresh]}


def expired_filter() -> dict[str, Any]:
    """Records whose TTL has lapsed. Records without a TTL never match."""
    return {"$and": [{TTL_FIELD: {"$exists": True}}, {TTL_FIELD: {"$lte": now()}}]}
