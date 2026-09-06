"""Error types surfaced to MCP clients as readable messages."""

from __future__ import annotations


class ToolboxError(Exception):
    """Base class for every error the toolbox raises deliberately."""


class ConfigurationError(ToolboxError):
    """Missing API key, unknown provider, unusable settings."""


class BackendNotFound(ToolboxError):
    """No backend registered under the requested name."""


class CapabilityError(ToolboxError):
    """The requested operation is not supported by this index's schema.

    Raised, for example, when a caller asks for full-text search on an index
    whose schema declares no FTS field. The message always names the index,
    what was asked for, and what the index actually supports, so the model
    calling the tool can correct itself without another round trip.
    """


class ReadOnlyError(ToolboxError):
    """A write was attempted while the server is in read-only mode."""


class ConfirmationRequired(ToolboxError):
    """A destructive operation was called without ``confirm=True``."""
