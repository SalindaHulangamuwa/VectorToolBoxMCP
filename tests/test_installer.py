"""vector-toolbox-install: merges into existing configs, never drops other servers."""

from __future__ import annotations

import json

import pytest

from vectortoolbox import installer


def test_merges_and_backs_up(tmp_path, monkeypatch):
    monkeypatch.setattr(installer.shutil, "which", lambda name: f"/opt/bin/{name}")
    cfg = tmp_path / "claude_desktop_config.json"
    cfg.write_text(json.dumps({"mcpServers": {"other": {"command": "x"}}, "theme": "dark"}))

    assert installer.main(["claude-desktop", "--config", str(cfg)]) == 0
    data = json.loads(cfg.read_text())
    assert data["theme"] == "dark" and "other" in data["mcpServers"]
    entry = data["mcpServers"]["vector-toolbox"]
    assert entry["command"] == "/opt/bin/uvx"
    assert entry["args"] == ["--from", "vector-toolbox-mcp[chroma]", "vector-toolbox-mcp"]
    assert (tmp_path / "claude_desktop_config.json.bak").exists()

    assert installer.main(["claude-desktop", "--config", str(cfg), "--remove"]) == 0
    assert list(json.loads(cfg.read_text())["mcpServers"]) == ["other"]


def test_antigravity_remote_uses_serverurl(monkeypatch):
    monkeypatch.setenv("TOK", "abc")
    entry = installer.build_entry("antigravity", "http", url="http://h:8000/mcp", token_env="TOK")
    assert entry == {"serverUrl": "http://h:8000/mcp", "headers": {"Authorization": "Bearer abc"}}
    assert "url" in installer.build_entry("cursor", "http", url="http://h:8000/mcp")


def test_claude_desktop_refuses_http_config():
    with pytest.raises(SystemExit, match="custom connector"):
        installer.build_entry("claude-desktop", "http", url="http://h/mcp")


def test_docker_entry_forwards_keys_without_values(monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "pcsk_secret")
    entry = installer.build_entry("antigravity", "docker")
    flat = " ".join(entry["args"])
    assert "VTB_TRANSPORT=stdio" in flat and "-e PINECONE_API_KEY" in flat
    assert "pcsk_secret" not in json.dumps(entry)  # secrets never land in the config


def test_env_file_is_passed_not_copied(tmp_path, monkeypatch):
    monkeypatch.setattr(installer.shutil, "which", lambda name: "/opt/bin/uvx")
    env = tmp_path / ".env"
    env.write_text("PINECONE_API_KEY=pcsk_secret\n")
    entry = installer.build_entry("cursor", "uvx", env_file=str(env))
    assert entry["args"][-2:] == ["--env-file", str(env.resolve())]
    assert "pcsk_secret" not in json.dumps(entry)
