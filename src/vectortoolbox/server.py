"""Vector Toolbox MCP server.

One MCP server, many vector databases - the same idea as Google's MCP Toolbox
for databases, applied to vector stores. Pinecone is the first backend; adding
another means implementing ``VectorStoreBackend`` and registering a tool
module here.
"""

from __future__ import annotations

import os

from ._mcp_compat import MCPServerType

INSTRUCTIONS = """\
Vector Toolbox: tools for working with vector databases. Pinecone is the
backend currently wired up.

Working order that avoids most errors:

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
"""


def build_server() -> MCPServerType:
    mcp = MCPServerType(
        name="vector-toolbox",
        instructions=INSTRUCTIONS,
    )

    # Importing the backends package registers each backend implementation.
    from .backends.pinecone import tools as pinecone_tools

    pinecone_tools.register(mcp)
    return mcp


def main() -> None:
    transport = os.environ.get("VTB_TRANSPORT", "stdio")
    build_server().run(transport=transport)


if __name__ == "__main__":
    main()
