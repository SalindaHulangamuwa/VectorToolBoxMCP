# Changelog

All notable changes are listed here. Versions follow [SemVer](https://semver.org/).

## [0.2.0] - unreleased

### Added
- **Chroma backend** - 24 `chroma_*` tools: named clients (ephemeral,
  persistent, HTTP, Chroma Cloud), collections and embedding functions, HNSW /
  SPANN configuration, add / update / upsert / delete with pre-checks,
  conditional transactions, query / get with selectable result shapes,
  metadata filtering with validation, and full-text search.
- **Streamable HTTP transport** (`--transport http`) with a `/health`
  endpoint, optional bearer-token auth (`VTB_AUTH_TOKEN`) and DNS-rebinding
  protection (`VTB_ALLOWED_HOSTS`).
- **CLI options**: `--transport`, `--host`, `--port`, `--path`, `--env-file`,
  `--read-only`, `--version`.
- **`vector-toolbox-install`** - registers the server in Claude Desktop,
  Antigravity or Cursor (uvx, Docker, source or HTTP modes).
- Docker image, `docker-compose.yml`, example client configs, CI and release
  workflows (PyPI + GHCR).

### Changed
- `vectortoolbox_status` moved to `tools_common.py`; backend details are now
  nested per backend (`status["pinecone"]["ttl"]` instead of `status["ttl"]`).
- `.env` lookup: `VTB_ENV_FILE` / `--env-file`, then a source checkout, then
  `~/.config/vector-toolbox/.env`, then the working directory.

## [0.1.0]

- Pinecone backend: schema-first indexes, five search recipes, client-side
  TTL, pluggable embedding providers.
