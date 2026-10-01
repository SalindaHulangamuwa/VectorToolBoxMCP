"""Named Chroma clients.

Chroma is client-first: one collection API sits on top of four deployments -
an in-memory store, a local directory, a self-hosted server and Chroma Cloud.
This module keeps a registry of *named* clients so one MCP session can work
against, say, a local persistent store and Chroma Cloud side by side. Every
Chroma tool takes ``client`` (default ``"default"``).

The ``default`` client is built lazily from settings (``VTB_CHROMA_CLIENT``
and friends) the first time a tool needs it, so a server with no Chroma
configuration still starts.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from ...config import get_settings
from ...errors import ConfigurationError, NotFound

ClientKind = Literal["ephemeral", "persistent", "http", "cloud"]
CLIENT_KINDS: tuple[str, ...] = ("ephemeral", "persistent", "http", "cloud")
DEFAULT_CLIENT = "default"


@dataclass
class ClientEntry:
    name: str
    kind: str
    client: Any
    params: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "tenant": getattr(self.client, "tenant", None),
            "database": getattr(self.client, "database", None),
            **self.params,
        }


_CLIENTS: dict[str, ClientEntry] = {}


def import_chromadb():
    try:
        import chromadb
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ConfigurationError(
            "chromadb is not installed. Install with: pip install 'vector-toolbox-mcp[chroma]'"
        ) from exc
    return chromadb


def build_client(
    kind: str,
    *,
    path: str | None = None,
    host: str | None = None,
    port: int | None = None,
    ssl: bool | None = None,
    headers: dict[str, str] | None = None,
    tenant: str | None = None,
    database: str | None = None,
    api_key_env_var: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Construct a chromadb client. Returns ``(client, redacted_params)``."""
    if kind not in CLIENT_KINDS:
        raise ConfigurationError(
            f"Unknown Chroma client kind {kind!r}. Use one of: {', '.join(CLIENT_KINDS)}."
        )
    chromadb = import_chromadb()
    settings = get_settings()

    if kind == "cloud":
        api_key = os.environ.get(api_key_env_var) if api_key_env_var else None
        api_key = api_key or settings.chroma_api_key
        if not api_key:
            where = f"the env var {api_key_env_var!r}" if api_key_env_var else "CHROMA_API_KEY"
            raise ConfigurationError(
                f"A Chroma Cloud client needs an API key, and {where} is empty. Set "
                "CHROMA_API_KEY in .env (or the MCP server env block), or pass "
                "api_key_env_var naming the variable that holds it."
            )
        client = chromadb.CloudClient(
            tenant=tenant or settings.chroma_tenant,
            database=database or settings.chroma_database,
            api_key=api_key,
        )
        return client, {}

    from chromadb.config import DEFAULT_DATABASE, DEFAULT_TENANT

    tenant = tenant or DEFAULT_TENANT
    database = database or DEFAULT_DATABASE

    if kind == "ephemeral":
        client = chromadb.EphemeralClient(tenant=tenant, database=database)
        return client, {
            "note": "In-memory: data is lost when the server stops. Every ephemeral "
            "client in this process shares the same store."
        }

    if kind == "persistent":
        resolved = os.path.abspath(os.path.expanduser(path or settings.chroma_path))
        os.makedirs(resolved, exist_ok=True)
        client = chromadb.PersistentClient(path=resolved, tenant=tenant, database=database)
        return client, {"path": resolved}

    # http - a self-hosted Chroma server (`chroma run`, Docker, ...)
    host = host or settings.chroma_host
    port = port or settings.chroma_port
    ssl = settings.chroma_ssl if ssl is None else ssl
    client = chromadb.HttpClient(
        host=host, port=port, ssl=ssl, headers=headers, tenant=tenant, database=database
    )
    return client, {
        "host": host,
        "port": port,
        "ssl": ssl,
        # Header values often carry tokens; report only the names.
        "headers": sorted(headers) if headers else [],
    }


def register_client(name: str, kind: str, *, replace: bool = False, **kwargs: Any) -> ClientEntry:
    if not name or not name.strip():
        raise ConfigurationError("A client needs a non-empty name.")
    if name in _CLIENTS and not replace:
        raise ConfigurationError(
            f"A client named {name!r} already exists ({_CLIENTS[name].kind}). "
            "Pass replace=true to rebuild it, or pick another name."
        )
    client, params = build_client(kind, **kwargs)
    entry = ClientEntry(name=name, kind=kind, client=client, params=params)
    _CLIENTS[name] = entry
    return entry


def _default_entry() -> ClientEntry:
    settings = get_settings()
    return register_client(DEFAULT_CLIENT, settings.chroma_client, replace=True)


def get_entry(name: str | None = None) -> ClientEntry:
    key = name or DEFAULT_CLIENT
    if key in _CLIENTS:
        return _CLIENTS[key]
    if key == DEFAULT_CLIENT:
        return _default_entry()
    raise NotFound(
        f"No Chroma client named {key!r}. Known clients: "
        f"{', '.join(sorted(_CLIENTS)) or 'none yet'}. Create one with chroma_create_client."
    )


def get_client(name: str | None = None) -> Any:
    return get_entry(name).client


def list_clients() -> list[dict[str, Any]]:
    return [entry.describe() for entry in _CLIENTS.values()]


def remove_client(name: str) -> dict[str, Any]:
    if name not in _CLIENTS:
        raise NotFound(f"No Chroma client named {name!r}.")
    entry = _CLIENTS.pop(name)
    return {"removed": name, "kind": entry.kind}


def reset_clients() -> None:
    """Used by tests."""
    _CLIENTS.clear()
