from __future__ import annotations

import os
from typing import Any

import pytest

os.environ.setdefault("PINECONE_API_KEY", "test-key")


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    """Keep the developer's own .env out of the test run.

    config.py loads a .env sitting beside the project so that MCP clients
    (which launch the server with an unpredictable cwd) still pick it up. That
    means a real .env would otherwise leak into these tests - a machine with
    VTB_READ_ONLY=true set would see write tools refuse and half the suite go
    red for no reason.
    """
    from vectortoolbox import config

    for name in (
        "VTB_READ_ONLY",
        "VTB_EMBED_PROVIDER",
        "VTB_EMBED_MODEL",
        "VTB_EMBED_DIMENSION",
        "VTB_SPARSE_PROVIDER",
        "VTB_SPARSE_MODEL",
        "VTB_PINECONE_CLOUD",
        "VTB_PINECONE_REGION",
        "VTB_DEFAULT_BACKEND",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PINECONE_API_KEY", "test-key")
    config.reset_settings_cache()
    yield
    config.reset_settings_cache()


def index_model(
    name: str = "test-index",
    fields: dict[str, Any] | None = None,
    host: str = "https://test.svc.pinecone.io",
) -> dict[str, Any]:
    return {
        "name": name,
        "host": host,
        "deletion_protection": "disabled",
        "status": {"ready": True, "state": "Ready"},
        "schema": {"fields": fields or {}},
    }


DENSE_FIELD = {"type": "dense_vector", "dimension": 8, "metric": "cosine"}
SPARSE_FIELD = {"type": "sparse_vector"}
FTS_FIELD = {"type": "string", "full_text_search": {"language": "en"}}

# The shape every index in a real Vectors-API project reports: system-named
# vector fields, no declared text fields.
LEGACY_VECTOR_FIELDS = {"_values": DENSE_FIELD, "_sparse_values": SPARSE_FIELD}


class FakeDocuments:
    def __init__(self, parent: "FakeIndex"):
        self.parent = parent

    def upsert(self, **kwargs):
        self.parent.calls.append(("documents.upsert", kwargs))
        return {"upserted_count": len(kwargs.get("documents", []))}

    def batch_upsert(self, **kwargs):
        self.parent.calls.append(("documents.batch_upsert", kwargs))
        return {"upserted_count": len(kwargs.get("documents", []))}

    def search(self, **kwargs):
        self.parent.calls.append(("documents.search", kwargs))
        return self.parent.search_response(kwargs)

    def delete(self, **kwargs):
        self.parent.calls.append(("documents.delete", kwargs))
        return {"deleted": True}

    def update(self, **kwargs):
        self.parent.calls.append(("documents.update", kwargs))
        return {"updated": True}

    def fetch(self, **kwargs):
        self.parent.calls.append(("documents.fetch", kwargs))
        return {"documents": self.parent.records}

    def list(self, **kwargs):
        self.parent.calls.append(("documents.list", kwargs))
        return iter([{"id": d["_id"]} for d in self.parent.records])


class FakeIndex:
    def __init__(self, records: list[dict] | None = None):
        self.calls: list[tuple[str, dict]] = []
        self.records = records or []
        self.documents = FakeDocuments(self)

    def search_response(self, kwargs):
        kind = kwargs["score_by"][0]["type"]
        return {
            "matches": [
                {"_id": f"{kind}-1", "_score": 0.9, "body": "one"},
                {"_id": "shared", "_score": 0.5, "body": "two"},
            ]
        }

    def query(self, **kwargs):
        self.calls.append(("query", kwargs))
        return {
            "matches": [
                {"id": "v1", "score": 0.7, "metadata": {"category": "policy"}},
                {"id": "v2", "score": 0.4, "metadata": {"category": "tech"}},
            ]
        }

    def upsert(self, **kwargs):
        self.calls.append(("upsert", kwargs))
        return {"upserted_count": len(kwargs.get("vectors", []))}

    def upsert_records(self, **kwargs):
        self.calls.append(("upsert_records", kwargs))
        return {"upserted_count": len(kwargs.get("records", []))}

    def search(self, **kwargs):
        self.calls.append(("search", kwargs))
        return {"result": {"hits": [{"_id": "r1", "_score": 0.8}]}}

    def describe_index_stats(self, **kwargs):
        self.calls.append(("describe_index_stats", kwargs))
        return {"total_vector_count": len(self.records)}

    def list_namespaces(self, **kwargs):
        return iter([{"namespaces": [{"name": "ns1", "record_count": 3}]}])

    def describe_namespace(self, **kwargs):
        return {"name": kwargs.get("name"), "record_count": 3}

    def delete_namespace(self, **kwargs):
        self.calls.append(("delete_namespace", kwargs))

    def list(self, **kwargs):
        self.calls.append(("list", kwargs))
        return iter([{"vectors": [{"id": r["_id"]} for r in self.records]}])

    def fetch(self, **kwargs):
        self.calls.append(("fetch", kwargs))
        wanted = set(kwargs.get("ids") or [])
        return {
            "vectors": {
                r["_id"]: {
                    "metadata": {k: v for k, v in r.items() if k != "_id"},
                    "values": [0.5, 0.5],
                }
                for r in self.records
                if r["_id"] in wanted
            }
        }

    def fetch_by_metadata(self, **kwargs):
        self.calls.append(("fetch_by_metadata", kwargs))
        return {
            "vectors": {
                r["_id"]: {"metadata": {k: v for k, v in r.items() if k != "_id"}}
                for r in self.records
            }
        }

    def delete(self, **kwargs):
        self.calls.append(("delete", kwargs))


class FakeIndexes:
    def __init__(self, models: dict[str, dict]):
        self.models = models
        self.created: list[dict] = []

    def create(self, **kwargs):
        self.created.append(kwargs)
        model = index_model(kwargs["name"], kwargs["schema"]["fields"])
        self.models[kwargs["name"]] = model
        return model

    def create_for_model(self, **kwargs):
        self.created.append(kwargs)
        text_field = kwargs["embed"]["field_map"]["text"]
        model = index_model(
            kwargs["name"],
            {text_field: {"type": "semantic_text", "dimension": 1024}},
        )
        self.models[kwargs["name"]] = model
        return model

    def describe(self, name: str):
        return self.models[name]

    def list(self):
        return list(self.models.values())

    def delete(self, name: str):
        self.models.pop(name, None)

    def configure(self, **kwargs):
        return self.models[kwargs["name"]]


class FakeClient:
    def __init__(self, models: dict[str, dict], index: FakeIndex):
        self.indexes = FakeIndexes(models)
        self._index = index

    def Index(self, name=None, host=None):
        return self._index

    def describe_index(self, name):
        return self.indexes.models[name]


@pytest.fixture
def wired(monkeypatch):
    """Patch the Pinecone client layer and hand back the fakes."""
    from vectortoolbox.backends.pinecone import backend as backend_mod
    from vectortoolbox.core import registry

    models: dict[str, dict] = {}
    index = FakeIndex()
    client = FakeClient(models, index)

    monkeypatch.setattr(backend_mod, "get_client", lambda: client)
    monkeypatch.setattr(backend_mod, "get_index", lambda name: index)
    backend_mod._CAPS_CACHE.clear()
    registry.reset_backend_cache()

    from vectortoolbox.backends.pinecone.backend import PineconeBackend

    return {
        "backend": PineconeBackend(),
        "client": client,
        "index": index,
        "models": models,
    }
