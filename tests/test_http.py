"""HTTP transport: health check, bearer auth, and a real MCP initialize over HTTP."""

from __future__ import annotations

import pytest

pytest.importorskip("httpx")
from starlette.testclient import TestClient  # noqa: E402

INIT = {
    "jsonrpc": "2.0", "id": 1, "method": "initialize",
    "params": {"protocolVersion": "2025-06-18", "capabilities": {},
               "clientInfo": {"name": "pytest", "version": "0"}},
}
HEADERS = {"accept": "application/json, text/event-stream", "content-type": "application/json"}


def _client(token=None, **kw):
    from vectortoolbox.http import build_http_app
    from vectortoolbox.server import build_server

    app = build_http_app(build_server(), host="127.0.0.1", token=token, version="test", **kw)
    return TestClient(app, base_url="http://127.0.0.1:8000")


def test_health_needs_no_token():
    with _client(token="s3cret") as c:
        r = c.get("/health")
        assert r.status_code == 200 and r.json()["status"] == "ok"


def test_missing_or_wrong_token_is_rejected():
    with _client(token="s3cret") as c:
        assert c.post("/mcp", json=INIT, headers=HEADERS).status_code == 401
        bad = c.post("/mcp", json=INIT, headers={**HEADERS, "authorization": "Bearer nope"})
        assert bad.status_code == 401 and bad.headers["www-authenticate"].startswith("Bearer")


def test_correct_token_reaches_the_mcp_server():
    with _client(token="s3cret") as c:
        r = c.post("/mcp", json=INIT, headers={**HEADERS, "authorization": "Bearer s3cret"})
        assert r.status_code == 200
        assert "vector-toolbox" in r.text  # serverInfo.name in the initialize result


def test_no_token_configured_means_open():
    with _client() as c:
        assert c.post("/mcp", json=INIT, headers=HEADERS).status_code == 200


def test_cli_flags_become_settings(monkeypatch):
    from vectortoolbox import config, server

    monkeypatch.setattr(server, "build_server", lambda: (_ for _ in ()).throw(SystemExit(0)))
    monkeypatch.setenv("VTB_READ_ONLY", "false")  # restored after the test
    with pytest.raises(SystemExit):
        server.main(["--read-only", "--transport", "stdio"])
    assert config.get_settings().read_only is True
