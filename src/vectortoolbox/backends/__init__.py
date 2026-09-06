"""Backend implementations.

Importing this package registers every backend that ships with the toolbox.
Adding a second vector database means adding a module here and calling
``register_backend`` from its ``__init__``.
"""

from . import pinecone as _pinecone  # noqa: F401  (registers the backend)

__all__ = ["_pinecone"]
