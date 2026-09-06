"""Environment-driven settings for the toolbox."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache

def _load_dotenv() -> None:
    """Load a .env sitting next to the project, not next to the caller.

    An MCP client launches the server with an unpredictable working directory
    (Claude Desktop uses "/"), so dotenv's default cwd-upwards search finds
    nothing. Look beside the installed package instead, then fall back to the
    normal search for the editable-install and run-from-source cases.
    """
    try:
        from dotenv import load_dotenv
    except Exception:  # pragma: no cover - dotenv is optional
        return

    here = Path(__file__).resolve()
    for parent in here.parents[:4]:
        candidate = parent / ".env"
        if candidate.is_file():
            load_dotenv(candidate)
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

    read_only: bool = False


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
        read_only=_flag("VTB_READ_ONLY"),
    )


def reset_settings_cache() -> None:
    """Used by tests after monkeypatching the environment."""
    get_settings.cache_clear()
