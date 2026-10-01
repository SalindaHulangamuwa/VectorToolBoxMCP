# Contributing

Thanks for helping. Bug reports, docs fixes, new tools and whole new backends
are all welcome.

## Setup

```bash
git clone https://github.com/SalindaHulangamuwa/VectorToolBoxMCP.git
cd VectorToolBoxMCP
uv venv && uv pip install -e ".[dev,chroma]"
.venv/bin/pytest -q              # Pinecone is mocked; Chroma runs in-memory for real
.venv/bin/ruff check src tests
```

No API keys are needed for the test suite. `scripts/smoke_test.py` runs against
a real Pinecone project and is optional.

## Ground rules

- **Tool docstrings are the product.** The model calling a tool reads only its
  name, docstring and argument schema. Say what the tool does, what each
  argument expects, and what errors mean.
- **Validate before the network.** If the database would reject a request,
  reject it first with a message that says how to fix it.
- **Report what the database does silently** (skipped ids, ignored config)
  instead of returning a false success.
- **Writes respect read-only mode** (`@_safe(write=True)`); anything that
  deletes data requires `confirm=true`.
- **Never accept raw secrets as tool arguments** - take the *name* of an
  environment variable.
- **Backends never import each other.** Cross-backend code lives in
  `tools_common.py` / `core/`.

## Adding a vector database

1. `src/vectortoolbox/backends/<name>/` with a `VectorStoreBackend`
   implementation and `register_backend("<name>", ...)` in `__init__.py`.
2. `tools.py` with `<name>_*` tools, a `status()` function, and
   `register_status_provider("<name>", status)` inside `register(mcp)`.
3. Call its `register(mcp)` from `server.build_server`.
4. Tests in `tests/test_<name>.py`; an optional extra in `pyproject.toml`.

## Releasing (maintainers)

Bump `version` in `pyproject.toml`, add a CHANGELOG entry, then
`git tag vX.Y.Z && git push --tags`. The release workflow publishes to PyPI
and GHCR.
