"""The tool layer: errors come back as readable JSON, not tracebacks."""

from __future__ import annotations

import asyncio

import pytest

from conftest import DENSE_FIELD, FTS_FIELD, index_model


@pytest.fixture
def server(wired, monkeypatch):
    """A live MCPServer whose Pinecone calls hit the fakes."""
    from vectortoolbox.backends.pinecone import tools as tools_mod

    monkeypatch.setattr(tools_mod, "_backend", lambda: wired["backend"])
    from vectortoolbox.server import build_server

    return build_server()


def call(server, name, **kwargs):
    """Unwrap a tool result across MCP SDK versions."""
    import json

    result = asyncio.run(server.call_tool(name, kwargs))
    if isinstance(result, tuple):  # mcp 1.x: (content, structured)
        return result[1]
    structured = getattr(result, "structured_content", None) or getattr(
        result, "structuredContent", None
    )
    if structured is not None:
        return structured.get("result", structured)
    content = result.content[0]
    return json.loads(content.text)


def test_all_tools_are_registered(server):
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert "pinecone_create_index" in names
    assert "pinecone_search" in names
    assert "vectortoolbox_status" in names
    assert len(names) == 28


def test_every_tool_has_a_docstring(server):
    for tool in asyncio.run(server.list_tools()):
        assert tool.description, f"{tool.name} has no description for the model to read"


def test_capability_error_returns_a_readable_payload(server, wired):
    wired["models"]["dense-only"] = index_model("dense-only", {"emb": DENSE_FIELD})
    result = call(server, "pinecone_search", index="dense-only", namespace="ns", mode="text", query="x")
    assert result["error"] == "CapabilityError"
    assert "cannot run a 'text' search" in result["message"]


def test_delete_index_without_confirm_is_refused_not_executed(server, wired):
    wired["models"]["doomed"] = index_model("doomed", {"body": FTS_FIELD})
    result = call(server, "pinecone_delete_index", index="doomed")
    assert result["error"] == "ConfirmationRequired"
    assert "doomed" in wired["models"]


def test_read_only_mode_blocks_writes(server, wired, monkeypatch):
    from vectortoolbox import config
    from vectortoolbox.backends.pinecone import tools as tools_mod

    monkeypatch.setenv("VTB_READ_ONLY", "true")
    config.reset_settings_cache()
    monkeypatch.setattr(tools_mod, "get_settings", config.get_settings)
    try:
        wired["models"]["fts"] = index_model("fts", {"body": FTS_FIELD})
        result = call(
            server, "pinecone_upsert_documents", index="fts", namespace="ns",
            documents=[{"_id": "1", "body": "x"}],
        )
        assert result["error"] == "ReadOnlyError"
        # ... while reads still work.
        assert "error" not in call(server, "pinecone_list_indexes")
    finally:
        monkeypatch.delenv("VTB_READ_ONLY", raising=False)
        config.reset_settings_cache()


def test_status_tool_reports_configuration(server):
    status = call(server, "vectortoolbox_status")
    assert "pinecone" in status["backends"]
    assert status["ttl"]["implementation"] == "client-side"
    assert "openai" in status["dense_embedding_providers"]
