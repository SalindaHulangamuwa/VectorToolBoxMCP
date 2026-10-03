"""Named Weaviate clients.

Weaviate runs in four shapes and the v4 client has a constructor for each:

* ``local``    - a server on this machine (Docker, ``weaviate`` binary): HTTP + gRPC ports.
* ``custom``   - any self-hosted server: separate HTTP / gRPC hosts, ports and TLS.
* ``cloud``    - Weaviate Cloud: cluster URL + API key.
* ``embedded`` - the client downloads and runs a Weaviate binary itself, with
  data persisted to a local directory (Linux and macOS only).

Every ``weaviate_*`` tool takes ``client`` (default ``"default"``); the default
client is built from VTB_WEAVIATE_* settings the first time it is needed.

Vectorizer and generative modules (text2vec-openai, generative-cohere, ...)
run *inside* Weaviate but need the provider's API key on each request. Those
keys travel as request headers; this module forwards any provider key it finds
in the environment (OPENAI_API_KEY -> X-OpenAI-Api-Key, ...) so a collection
that uses ``text2vec-openai`` works without extra setup. Only header *names*
are ever reported back.
"""

from __future__ import annotations

import atexit
import os
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from ...config import get_settings
from ...errors import ConfigurationError, NotFound

ClientKind = Literal["local", "custom", "cloud", "embedded"]
CLIENT_KINDS: tuple[str, ...] = ("local", "custom", "cloud", "embedded")
DEFAULT_CLIENT = "default"

# env var -> header Weaviate's model-provider modules read.
PROVIDER_HEADERS: dict[str, str] = {
    "OPENAI_API_KEY": "X-OpenAI-Api-Key",
    "AZURE_OPENAI_API_KEY": "X-Azure-Api-Key",
    "COHERE_API_KEY": "X-Cohere-Api-Key",
    "VOYAGEAI_API_KEY": "X-VoyageAI-Api-Key",
    "JINAAI_API_KEY": "X-JinaAI-Api-Key",
    "MISTRAL_API_KEY": "X-Mistral-Api-Key",
    "ANTHROPIC_API_KEY": "X-Anthropic-Api-Key",
    "HUGGINGFACE_API_KEY": "X-HuggingFace-Api-Key",
    "NVIDIA_API_KEY": "X-Nvidia-Api-Key",
    "XAI_API_KEY": "X-Xai-Api-Key",
    "DATABRICKS_TOKEN": "X-Databricks-Token",
}


@dataclass
class ClientEntry:
    name: str
    kind: str
    client: Any
    params: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, **self.params}


_CLIENTS: dict[str, ClientEntry] = {}


def import_weaviate():
    try:
        import weaviate
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ConfigurationError(
            "weaviate-client is not installed. Install with: pip install 'vector-toolbox-mcp[weaviate]'"
        ) from exc
    return weaviate


def provider_headers(extra_env_vars: dict[str, str] | None = None) -> tuple[dict[str, str], list[str]]:
    """Headers carrying model-provider keys found in the environment.

    ``extra_env_vars`` maps header name -> env var name for providers not in
    the built-in table. Returns (headers, header_names).
    """
    headers: dict[str, str] = {}
    for env_name, header in PROVIDER_HEADERS.items():
        value = os.environ.get(env_name)
        if value:
            headers[header] = value
    for header, env_name in (extra_env_vars or {}).items():
        value = os.environ.get(env_name)
        if not value:
            raise ConfigurationError(f"Environment variable {env_name!r} (for header {header}) is not set.")
        headers[header] = value
    return headers, sorted(headers)


def _auth(api_key_env_var: str | None):
    weaviate = import_weaviate()
    key = os.environ.get(api_key_env_var) if api_key_env_var else None
    key = key or get_settings().weaviate_api_key
    return (weaviate.auth.AuthApiKey(key) if key else None), bool(key)


def build_client(
    kind: str,
    *,
    host: str | None = None,
    port: int | None = None,
    secure: bool | None = None,
    grpc_host: str | None = None,
    grpc_port: int | None = None,
    grpc_secure: bool | None = None,
    url: str | None = None,
    api_key_env_var: str | None = None,
    header_env_vars: dict[str, str] | None = None,
    path: str | None = None,
    version: str | None = None,
    timeout_seconds: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Construct and connect a Weaviate v4 client. Returns ``(client, redacted_params)``."""
    if kind not in CLIENT_KINDS:
        raise ConfigurationError(f"Unknown Weaviate client kind {kind!r}. Use one of: {', '.join(CLIENT_KINDS)}.")
    weaviate = import_weaviate()
    from weaviate.classes.init import AdditionalConfig, Timeout

    s = get_settings()
    headers, header_names = provider_headers(header_env_vars)
    extra = AdditionalConfig(timeout=Timeout(init=30, query=timeout_seconds or 60, insert=timeout_seconds or 120))
    params: dict[str, Any] = {"provider_headers": header_names}

    if kind == "cloud":
        url = url or s.weaviate_url
        if not url:
            raise ConfigurationError("A Weaviate Cloud client needs the cluster URL: pass url or set WEAVIATE_URL.")
        auth, has_key = _auth(api_key_env_var)
        if not has_key:
            raise ConfigurationError(
                "A Weaviate Cloud client needs an API key: set WEAVIATE_API_KEY, or pass "
                "api_key_env_var naming the variable that holds it."
            )
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=url, auth_credentials=auth, headers=headers or None, additional_config=extra
        )
        params.update(url=url)
        return client, params

    if kind == "embedded":
        data_path = os.path.abspath(os.path.expanduser(path or s.weaviate_embedded_path))
        os.makedirs(data_path, exist_ok=True)
        kwargs: dict[str, Any] = {
            "persistence_data_path": os.path.join(data_path, "data"),
            "binary_path": os.path.join(data_path, "bin"),
            "headers": headers or None,
            "additional_config": extra,
            # A container or VM often has no private IP, and the default
            # gossip config then refuses to start. Loopback is always there.
            "environment_variables": {"CLUSTER_ADVERTISE_ADDR": "127.0.0.1", "DISABLE_TELEMETRY": "true"},
        }
        if port:
            kwargs["port"] = port
        if grpc_port:
            kwargs["grpc_port"] = grpc_port
        if version or s.weaviate_embedded_version:
            kwargs["version"] = version or s.weaviate_embedded_version
        client = weaviate.connect_to_embedded(**kwargs)
        params.update(path=data_path, port=port or 8079, grpc_port=grpc_port or 50050,
                      note="The client runs Weaviate itself; it stops when this server stops.")
        return client, params

    auth, has_key = _auth(api_key_env_var)
    http_host = host or s.weaviate_http_host
    http_port = port or s.weaviate_http_port
    http_secure = s.weaviate_http_secure if secure is None else secure
    g_port = grpc_port or s.weaviate_grpc_port

    if kind == "local":
        client = weaviate.connect_to_local(
            host=http_host, port=http_port, grpc_port=g_port, headers=headers or None,
            additional_config=extra, auth_credentials=auth,
        )
        params.update(host=http_host, port=http_port, grpc_port=g_port, authenticated=has_key)
        return client, params

    g_host = grpc_host or s.weaviate_grpc_host or http_host
    g_secure = (s.weaviate_grpc_secure if s.weaviate_grpc_secure is not None else http_secure) \
        if grpc_secure is None else grpc_secure
    client = weaviate.connect_to_custom(
        http_host=http_host, http_port=http_port, http_secure=http_secure,
        grpc_host=g_host, grpc_port=g_port, grpc_secure=g_secure,
        headers=headers or None, additional_config=extra, auth_credentials=auth,
    )
    params.update(http=f"{'https' if http_secure else 'http'}://{http_host}:{http_port}",
                  grpc=f"{g_host}:{g_port}{' (tls)' if g_secure else ''}", authenticated=has_key)
    return client, params


def register_client(name: str, kind: str, *, replace: bool = False, **kwargs: Any) -> ClientEntry:
    if not name or not name.strip():
        raise ConfigurationError("A client needs a non-empty name.")
    if name in _CLIENTS and not replace:
        raise ConfigurationError(
            f"A client named {name!r} already exists ({_CLIENTS[name].kind}). "
            "Pass replace=true to rebuild it, or pick another name."
        )
    client, params = build_client(kind, **kwargs)
    if name in _CLIENTS:
        _close(_CLIENTS[name])
    entry = ClientEntry(name=name, kind=kind, client=client, params=params)
    _CLIENTS[name] = entry
    return entry


def get_entry(name: str | None = None) -> ClientEntry:
    key = name or DEFAULT_CLIENT
    if key in _CLIENTS:
        return _CLIENTS[key]
    if key == DEFAULT_CLIENT:
        return register_client(DEFAULT_CLIENT, get_settings().weaviate_client, replace=True)
    raise NotFound(
        f"No Weaviate client named {key!r}. Known clients: "
        f"{', '.join(sorted(_CLIENTS)) or 'none yet'}. Create one with weaviate_create_client."
    )


def get_client(name: str | None = None) -> Any:
    return get_entry(name).client


def list_clients() -> list[dict[str, Any]]:
    return [entry.describe() for entry in _CLIENTS.values()]


def _close(entry: ClientEntry) -> None:
    try:
        entry.client.close()
    except Exception:  # pragma: no cover - best effort
        pass


def remove_client(name: str) -> dict[str, Any]:
    if name not in _CLIENTS:
        raise NotFound(f"No Weaviate client named {name!r}.")
    entry = _CLIENTS.pop(name)
    _close(entry)
    return {"removed": name, "kind": entry.kind}


def reset_clients() -> None:
    for entry in list(_CLIENTS.values()):
        _close(entry)
    _CLIENTS.clear()


atexit.register(reset_clients)
