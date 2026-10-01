"""Vector Toolbox MCP server.

One MCP server, many vector databases - the same idea as Google's MCP Toolbox
for databases, applied to vector stores. Pinecone and Chroma are wired up; adding
another means implementing ``VectorStoreBackend`` and registering a tool
module here.
"""

from __future__ import annotations

import os

from ._mcp_compat import MCPServerType

INSTRUCTIONS = """\
Vector Toolbox: tools for working with vector databases. Two backends are
wired up - Pinecone (`pinecone_*`) and Chroma (`chroma_*`).

Pinecone - working order that avoids most errors:

1. `pinecone_list_indexes` - see what exists and which search modes each index
   supports.
2. `pinecone_index_capabilities` - before searching an index you did not just
   create. An index can only answer the searches its schema declared, and a
   schema cannot be changed after creation.
3. `pinecone_sample_metadata` - before writing a metadata filter, to learn the
   field names and types actually present in a namespace.
4. `pinecone_search` with `mode="auto"` unless you need a specific signal.

Creating an index is the decision that matters: dense fields give semantic
search, sparse fields give learned lexical search, string fields with
full-text search give BM25 and Lucene queries. Declare everything you might
need up front - adding a signal later means a new index.

Chroma - every tool takes `client` (default "default", built from VTB_CHROMA_*):

1. `chroma_create_client` if you need more than the default: ephemeral
   (in-memory), persistent (local dir), http (self-hosted) or cloud.
2. `chroma_list_collections` / `chroma_get_collection` - the collection's
   distance space and embedding route decide how to query it.
3. `chroma_sample_metadata` before writing a `where` filter;
   `chroma_validate_filter` to check one.
4. `chroma_query` for similarity, `chroma_get` for lookup by id/filter,
   `chroma_full_text_search` for $contains/$regex document matching
   (optionally ranked by a vector query).
5. `chroma_conditional_transaction` when a write must depend on current state.
"""


def build_server() -> MCPServerType:
    mcp = MCPServerType(
        name="vector-toolbox",
        instructions=INSTRUCTIONS,
    )

    # Importing the backends package registers each backend implementation.
    from . import tools_common
    from .backends.chroma import tools as chroma_tools
    from .backends.pinecone import tools as pinecone_tools

    pinecone_tools.register(mcp)
    chroma_tools.register(mcp)
    tools_common.register(mcp)
    return mcp


def _parse_args(argv: list[str] | None = None):
    import argparse

    from . import __version__

    parser = argparse.ArgumentParser(
        prog="vector-toolbox-mcp",
        description="MCP server for vector databases (Pinecone, Chroma). Every option can also "
        "be set through the environment variable shown.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--transport", choices=["stdio", "http", "sse"],
                        help="stdio (default; for desktop clients) or http (streamable HTTP, for "
                        "Docker/remote). [VTB_TRANSPORT]")
    parser.add_argument("--host", help="HTTP bind address, default 127.0.0.1. [VTB_HOST]")
    parser.add_argument("--port", type=int, help="HTTP port, default 8000. [VTB_PORT]")
    parser.add_argument("--path", help="HTTP endpoint path, default /mcp. [VTB_HTTP_PATH]")
    parser.add_argument("--env-file", help="Load settings from this .env file. [VTB_ENV_FILE]")
    parser.add_argument("--read-only", action="store_true", default=None,
                        help="Disable every write/delete tool. [VTB_READ_ONLY]")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    # Flags become environment variables *before* settings are first read, so
    # there is one source of truth (config.get_settings) for everything else.
    if args.env_file:
        os.environ["VTB_ENV_FILE"] = os.path.abspath(os.path.expanduser(args.env_file))
    if args.read_only:
        os.environ["VTB_READ_ONLY"] = "true"

    from . import __version__
    from .config import get_settings, reset_settings_cache

    reset_settings_cache()
    settings = get_settings()

    transport = (args.transport or os.environ.get("VTB_TRANSPORT") or "stdio").lower()
    if transport == "streamable-http":
        transport = "http"
    mcp = build_server()

    if transport == "stdio":
        mcp.run(transport="stdio")
        return
    if transport == "sse":  # legacy clients only
        mcp.run(transport="sse")
        return
    if transport != "http":
        raise SystemExit(f"Unknown transport {transport!r}: use stdio, http or sse.")

    from .http import serve

    serve(
        mcp,
        host=args.host or settings.http_host,
        port=args.port or settings.http_port,
        path=args.path or settings.http_path,
        token=settings.auth_token,
        allowed_hosts=settings.allowed_hosts,
        stateless=settings.stateless_http,
        version=__version__,
    )


if __name__ == "__main__":
    main()
