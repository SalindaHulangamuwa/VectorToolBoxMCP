"""Environment-driven settings for the toolbox."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _load_dotenv() -> None:
    """Find and load a .env file. Variables already set always win.

    Order: ``VTB_ENV_FILE`` (or ``--env-file``) if given; else a .env beside a
    source checkout (MCP clients launch servers with an unpredictable working
    directory - Claude Desktop uses "/" - so a cwd search alone finds nothing);
    else ``~/.config/vector-toolbox/.env``; else the usual search from the cwd.
    Installed from PyPI (uvx / pip) there is usually no .env at all, and
    settings come from the client config's ``env`` block instead.
    """
    try:
        from dotenv import load_dotenv
    except Exception:  # pragma: no cover - dotenv is optional
        return

    explicit = os.environ.get("VTB_ENV_FILE")
    if explicit:
        load_dotenv(os.path.expanduser(explicit))
        return
    here = Path(__file__).resolve()
    for parent in here.parents[:3]:  # src/vectortoolbox -> src -> repo root
        candidate = parent / ".env"
        if candidate.is_file() and (parent / "pyproject.toml").is_file():
            load_dotenv(candidate)
            return
    user_file = Path.home() / ".config" / "vector-toolbox" / ".env"
    if user_file.is_file():
        load_dotenv(user_file)
        return
    load_dotenv()


_load_dotenv()


def _env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value not in (None, ""):
            return value
    return default


def _flag(*names: str, default: bool = False) -> bool:
    raw = _env(*names)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    default_backend: str = "pinecone"

    # Pinecone
    pinecone_api_key: str | None = None
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # Embeddings
    embed_provider: str = "pinecone"
    embed_model: str = "llama-text-embed-v2"
    embed_dimension: int | None = 1024
    sparse_provider: str | None = "pinecone"
    sparse_model: str | None = "pinecone-sparse-english-v0"

    openai_api_key: str | None = None
    cohere_api_key: str | None = None

    # Chroma - the "default" client, built lazily from these settings.
    # kind: ephemeral (in-memory) | persistent (local directory) | http | cloud
    chroma_client: str = "persistent"
    chroma_path: str = "~/.vector-toolbox/chroma"
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_ssl: bool = False
    chroma_api_key: str | None = None
    chroma_tenant: str | None = None
    chroma_database: str | None = None

    # Weaviate - the "default" client, built lazily from these settings.
    # kind: local | custom | cloud | embedded
    weaviate_client: str = "local"
    weaviate_http_host: str = "localhost"
    weaviate_http_port: int = 8080
    weaviate_http_secure: bool = False
    weaviate_grpc_host: str | None = None
    weaviate_grpc_port: int = 50051
    weaviate_grpc_secure: bool | None = None
    weaviate_url: str | None = None
    weaviate_api_key: str | None = None
    weaviate_embedded_path: str = "~/.vector-toolbox/weaviate"
    weaviate_embedded_version: str | None = None

    read_only: bool = False

    # HTTP transport (Docker / remote). stdio ignores these.
    http_host: str = "127.0.0.1"
    http_port: int = 8000
    http_path: str = "/mcp"
    auth_token: str | None = None
    allowed_hosts: list[str] | None = None
    stateless_http: bool = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    dim = _env("VTB_EMBED_DIMENSION")
    return Settings(
        default_backend=_env("VTB_DEFAULT_BACKEND", default="pinecone"),
        pinecone_api_key=_env("PINECONE_API_KEY", "VTB_PINECONE_API_KEY"),
        pinecone_cloud=_env("VTB_PINECONE_CLOUD", default="aws"),
        pinecone_region=_env("VTB_PINECONE_REGION", default="us-east-1"),
        embed_provider=_env("VTB_EMBED_PROVIDER", default="pinecone"),
        embed_model=_env("VTB_EMBED_MODEL", default="llama-text-embed-v2"),
        embed_dimension=int(dim) if dim and dim.isdigit() else None,
        sparse_provider=_env("VTB_SPARSE_PROVIDER", default="pinecone"),
        sparse_model=_env("VTB_SPARSE_MODEL", default="pinecone-sparse-english-v0"),
        openai_api_key=_env("OPENAI_API_KEY"),
        cohere_api_key=_env("COHERE_API_KEY"),
        chroma_client=_env("VTB_CHROMA_CLIENT", default="persistent"),
        chroma_path=_env("VTB_CHROMA_PATH", default="~/.vector-toolbox/chroma"),
        chroma_host=_env("VTB_CHROMA_HOST", default="localhost"),
        chroma_port=int(_env("VTB_CHROMA_PORT", default="8000")),
        chroma_ssl=_flag("VTB_CHROMA_SSL"),
        chroma_api_key=_env("CHROMA_API_KEY", "VTB_CHROMA_API_KEY"),
        chroma_tenant=_env("CHROMA_TENANT", "VTB_CHROMA_TENANT"),
        chroma_database=_env("CHROMA_DATABASE", "VTB_CHROMA_DATABASE"),
        weaviate_client=_env("VTB_WEAVIATE_CLIENT", default="local"),
        weaviate_http_host=_env("VTB_WEAVIATE_HOST", default="localhost"),
        weaviate_http_port=int(_env("VTB_WEAVIATE_PORT", default="8080")),
        weaviate_http_secure=_flag("VTB_WEAVIATE_SECURE"),
        weaviate_grpc_host=_env("VTB_WEAVIATE_GRPC_HOST"),
        weaviate_grpc_port=int(_env("VTB_WEAVIATE_GRPC_PORT", default="50051")),
        weaviate_grpc_secure=_flag("VTB_WEAVIATE_GRPC_SECURE") if _env("VTB_WEAVIATE_GRPC_SECURE") else None,
        weaviate_url=_env("WEAVIATE_URL", "VTB_WEAVIATE_URL"),
        weaviate_api_key=_env("WEAVIATE_API_KEY", "VTB_WEAVIATE_API_KEY"),
        weaviate_embedded_path=_env("VTB_WEAVIATE_EMBEDDED_PATH", default="~/.vector-toolbox/weaviate"),
        weaviate_embedded_version=_env("VTB_WEAVIATE_EMBEDDED_VERSION"),
        read_only=_flag("VTB_READ_ONLY"),
        http_host=_env("VTB_HOST", default="127.0.0.1"),
        http_port=int(_env("VTB_PORT", default="8000")),
        http_path=_env("VTB_HTTP_PATH", default="/mcp"),
        auth_token=_env("VTB_AUTH_TOKEN"),
        allowed_hosts=[h.strip() for h in (_env("VTB_ALLOWED_HOSTS") or "").split(",") if h.strip()] or None,
        stateless_http=_flag("VTB_STATELESS_HTTP"),
    )


def reset_settings_cache() -> None:
    """Re-read settings (tests, and main() after applying CLI flags)."""
    if os.environ.get("VTB_ENV_FILE"):
        _load_dotenv()
    get_settings.cache_clear()
