"""Backend registry - how a second vector database gets added."""

from __future__ import annotations

from typing import Callable

from ..errors import BackendNotFound
from .base import VectorStoreBackend

_FACTORIES: dict[str, Callable[[], VectorStoreBackend]] = {}
_INSTANCES: dict[str, VectorStoreBackend] = {}


def register_backend(name: str, factory: Callable[[], VectorStoreBackend]) -> None:
    _FACTORIES[name] = factory


def list_backends() -> list[str]:
    return sorted(_FACTORIES)


def get_backend(name: str | None = None) -> VectorStoreBackend:
    from ..config import get_settings

    key = name or get_settings().default_backend
    if key not in _FACTORIES:
        raise BackendNotFound(
            f"No backend named {key!r}. Registered backends: {', '.join(list_backends()) or 'none'}."
        )
    if key not in _INSTANCES:
        _INSTANCES[key] = _FACTORIES[key]()
    return _INSTANCES[key]


def reset_backend_cache() -> None:
    _INSTANCES.clear()
