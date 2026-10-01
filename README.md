# Vector Toolbox MCP

[![CI](https://github.com/SalindaHulangamuwa/VectorToolBoxMCP/actions/workflows/ci.yml/badge.svg)](https://github.com/SalindaHulangamuwa/VectorToolBoxMCP/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/vector-toolbox-mcp)](https://pypi.org/project/vector-toolbox-mcp/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

One MCP server, many vector databases — the same idea as Google's MCP Toolbox for
databases, applied to vector stores. **Pinecone and Chroma** are wired up; the tool
surface, backend contract and embedding layer are built so Qdrant, Weaviate,
Milvus or pgvector slot in behind the same shape.

Built against the **Pinecone Python SDK v10** — the Documents API with declared
field schemas, not the older `dimension`/`metric` index model.

## Quick start

You need [uv](https://docs.astral.sh/uv/getting-started/installation/) (or Docker).
Add this to your MCP client's config and restart it:

```json
{
  "mcpServers": {
    "vector-toolbox": {
      "command": "uvx",
      "args": ["--from", "vector-toolbox-mcp[chroma]", "vector-toolbox-mcp"],
      "env": { "PINECONE_API_KEY": "pcsk_..." }
    }
  }
}
```

Or let the installer find the config file, keep your other servers, and use
absolute paths for you:

```bash
uvx --from vector-toolbox-mcp vector-toolbox-install claude-desktop   # or: antigravity, cursor
```

Chroma needs no account — its default store is a local folder
(`~/.vector-toolbox/chroma`). Pinecone needs `PINECONE_API_KEY`. Full options:
[Install and connect](#install-and-connect).

---

## Install and connect

| Way to run | Best for | Command the client launches |
|---|---|---|
| **uvx** (PyPI) | Most people | `uvx --from vector-toolbox-mcp[chroma] vector-toolbox-mcp` |
| **Docker, stdio** | No Python on the machine | `docker run -i --rm -e VTB_TRANSPORT=stdio ghcr.io/salindahulangamuwa/vector-toolbox-mcp` |
| **Docker, HTTP** | One shared server, several clients or machines | client connects to `http://host:8000/mcp` |
| **Source checkout** | Developing the server | `/path/to/VectorToolBoxMCP/.venv/bin/vector-toolbox-mcp` |

**Extras** pick optional dependencies: `chroma`, `openai`, `cohere`, `local`
(sentence-transformers — large), or `all` (everything but `local`). Pinecone
support is always included. Example: `vector-toolbox-mcp[chroma,openai]`.

### Configuration

Everything is an environment variable — set them in the client config's `env`
block, in a `.env` file, or with `-e` for Docker. The full list with comments is
in [`.env.example`](.env.example). The ones most people touch:

| Variable | Default | Purpose |
|---|---|---|
| `PINECONE_API_KEY` | — | Pinecone backend |
| `VTB_CHROMA_CLIENT` | `persistent` | Chroma default client: `ephemeral`, `persistent`, `http`, `cloud` |
| `VTB_CHROMA_PATH` | `~/.vector-toolbox/chroma` | Where the persistent Chroma store lives |
| `CHROMA_API_KEY` / `CHROMA_TENANT` / `CHROMA_DATABASE` | — | Chroma Cloud |
| `VTB_EMBED_PROVIDER` / `VTB_EMBED_MODEL` | `pinecone` / `llama-text-embed-v2` | Default embedder (`pinecone`, `openai`, `cohere`, `huggingface`) |
| `OPENAI_API_KEY`, `COHERE_API_KEY` | — | Those embedders |
| `VTB_READ_ONLY` | `false` | `true` disables every write and delete tool |
| `VTB_TRANSPORT` | `stdio` | `http` to serve over streamable HTTP |
| `VTB_HOST` / `VTB_PORT` / `VTB_HTTP_PATH` | `127.0.0.1` / `8000` / `/mcp` | HTTP listener |
| `VTB_AUTH_TOKEN` | — | Require `Authorization: Bearer <token>` on HTTP |
| `VTB_ALLOWED_HOSTS` | — | Host headers to accept on HTTP (DNS-rebinding protection) |

A `.env` is found, in order: `--env-file` / `VTB_ENV_FILE`, a source checkout's
root, `~/.config/vector-toolbox/.env`, the working directory. Real environment
variables always win. Command-line flags: `vector-toolbox-mcp --help`.

### Claude Desktop

Config file: macOS `~/Library/Application Support/Claude/claude_desktop_config.json`,
Windows `%APPDATA%\Claude\claude_desktop_config.json` (Settings → Developer →
Edit Config opens it).

```bash
vector-toolbox-install claude-desktop                  # uvx
vector-toolbox-install claude-desktop --mode docker
vector-toolbox-install claude-desktop --mode source    # from a checkout
```

or paste one of [`examples/claude-desktop/`](examples/claude-desktop) — `uvx.json`,
`docker.json`, `source.json`. Then **quit Claude Desktop completely** (Cmd+Q on
macOS, Exit from the tray on Windows) and reopen it.

> **macOS: "spawn uvx ENOENT".** Apps started from the Dock don't get your
> shell's `PATH`. Use the absolute path from `which uvx` as `command` (the
> installer does this for you).

Claude Desktop's config file only launches local servers. To use a remote HTTP
deployment, add its URL as a custom connector in Claude's settings instead.

### Claude Code

```bash
claude mcp add vector-toolbox -e PINECONE_API_KEY=pcsk_... -- uvx --from 'vector-toolbox-mcp[chroma]' vector-toolbox-mcp
claude mcp add --transport http vector-toolbox http://localhost:8000/mcp --header "Authorization: Bearer $VTB_AUTH_TOKEN"
```

### Google Antigravity

Antigravity (IDE, 2.0 and CLI) reads `~/.gemini/config/mcp_config.json`, or
`.agents/mcp_config.json` inside a workspace. In the IDE: agent panel **…** →
**MCP Servers** → **Manage MCP Servers** → **View raw config**.

```bash
vector-toolbox-install antigravity                                  # uvx, local
vector-toolbox-install antigravity --mode http --url http://localhost:8000/mcp --token-env VTB_AUTH_TOKEN
```

Examples: [`examples/antigravity/mcp_config.json`](examples/antigravity/mcp_config.json)
(local) and [`mcp_config.remote.json`](examples/antigravity/mcp_config.remote.json)
(remote — note Antigravity's key is `serverUrl`, not `url`). Refresh the server
list after editing.

### Cursor and VS Code

Cursor: `~/.cursor/mcp.json` or `.cursor/mcp.json` in a project —
`vector-toolbox-install cursor`, or see [`examples/cursor/mcp.json`](examples/cursor/mcp.json).
VS Code: `.vscode/mcp.json` uses a `servers` key and can prompt for secrets —
see [`examples/vscode/mcp.json`](examples/vscode/mcp.json).

### Docker

The image `ghcr.io/salindahulangamuwa/vector-toolbox-mcp` (linux/amd64 and
arm64) runs as a non-root user with `/data` as its volume: the persistent Chroma
store and Chroma's embedding-model cache live there.

**stdio** — the client starts a container per session (see
[`examples/claude-desktop/docker.json`](examples/claude-desktop/docker.json)):

```bash
docker run -i --rm -e VTB_TRANSPORT=stdio -e PINECONE_API_KEY \
  -v vector-toolbox-data:/data ghcr.io/salindahulangamuwa/vector-toolbox-mcp
```

**HTTP** — one long-running server, any number of clients:

```bash
export VTB_AUTH_TOKEN=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
docker run -d --name vector-toolbox -p 127.0.0.1:8000:8000 \
  -e VTB_AUTH_TOKEN -e PINECONE_API_KEY -v vector-toolbox-data:/data \
  ghcr.io/salindahulangamuwa/vector-toolbox-mcp
curl http://127.0.0.1:8000/health          # {"status": "ok", ...}
```

Clients connect to `http://127.0.0.1:8000/mcp` with
`Authorization: Bearer $VTB_AUTH_TOKEN`.

**Compose** — [`docker-compose.yml`](docker-compose.yml) runs the HTTP server
with secrets from `.env`; `--profile chroma` adds a Chroma server alongside it:

```bash
cp .env.example .env        # fill in keys and VTB_AUTH_TOKEN
docker compose up -d
docker compose --profile chroma up -d
```

Build it yourself with `docker build -t vector-toolbox-mcp .`
(`--build-arg EXTRAS=chroma,openai,cohere,local` to include sentence-transformers).

> **Exposing HTTP beyond localhost:** set `VTB_AUTH_TOKEN`, put TLS in front
> (Caddy, nginx, a cloud load balancer), and set `VTB_ALLOWED_HOSTS` to your
> domain. The server prints a warning if it binds to a public address without
> a token. See [SECURITY.md](SECURITY.md).

### From source

```bash
git clone https://github.com/SalindaHulangamuwa/VectorToolBoxMCP.git && cd VectorToolBoxMCP
uv venv && uv pip install -e ".[chroma]"
cp .env.example .env                                  # optional; read automatically from here
python scripts/handshake_check.py .venv/bin/vector-toolbox-mcp   # proves MCP works over stdio
.venv/bin/vector-toolbox-install claude-desktop --mode source
```

`bash scripts/setup_macos.sh` does the venv, install and handshake in one go.

### If it does not show up

| Symptom | Cause |
|---|---|
| Server missing from the client | The client was reloaded, not fully restarted |
| `spawn … ENOENT` | `command` isn't on the GUI app's `PATH` — use an absolute path |
| Every call errors | Keys not reaching the server — call `vectortoolbox_status`, or run the command by hand |
| Only read tools work | `VTB_READ_ONLY=true` |
| `chromadb is not installed` | Add the `chroma` extra: `vector-toolbox-mcp[chroma]` |
| HTTP returns 401 | Missing or wrong `Authorization: Bearer` header |

Running the command by hand shows start-up errors the client hides. It then
waits silently for JSON-RPC on stdin — that is normal.

---

## Why the schema comes first

In current Pinecone an index declares a **schema** of searchable fields, and the
schema decides which searches that index can ever answer:

| Field type | Enables |
|---|---|
| `dense_vector` | semantic search |
| `sparse_vector` | learned lexical search (SPLADE-style) |
| `string` + `full_text_search` | BM25 and Lucene query strings |

**The schema is immutable.** Adding a signal later means creating a new index and
reindexing. Because of that, every search tool in this server checks the request
against the live schema *before* going to the network, and a mismatch comes back
as an explanation ("index X has no FTS field; it supports dense, vectors_api")
rather than an HTTP 400.

`pinecone_index_capabilities` is the tool that reports this.

---

## The five index/search recipes

| # | Setup | Create with | Search with |
|---|---|---|---|
| 1 | Single text field — keyword only (FTS) | `text_fields=[{"name":"body"}]` | `mode="text"` |
| 2 | Multi-field FTS | `text_fields=[{"name":"body"},{"name":"summary"}]` | `mode="text", fields=["body","summary"]` — or `field_queries={...}` for a different query per field |
| 3 | Dense + FTS | `dense_fields=[…]` + `text_fields=[…]` | `mode="hybrid"` (fused) or either alone |
| 4 | Multi-signal — dense + sparse + FTS | all three lists | `mode="hybrid"` |
| 5 | Sparse + dense hybrid, single-vector index | one dense + one sparse field, no text | `pinecone_query_vectors` |

### Multi-field FTS is several clauses, not one

A `text` clause names **exactly one** field. Scoring across `body` and `summary`
means two clauses in a single request, which Pinecone combines with **equal
weight** — there is no per-clause weight parameter. The toolbox builds those
clauses for you from `fields`, and `field_queries` lets each field carry its own
query text. Weighting across text fields is only reachable through
`query_string` boosts (`title:(x)^3 OR body:(x)`) or by running the fields
separately and fusing with `mode="hybrid"`.

### One thing worth knowing about hybrid

Pinecone's Documents API accepts several `text` / `query_string` clauses in one
request, but a `dense_vector` or `sparse_vector` clause **must stand alone**. So a
true multi-signal search is several requests fused client-side. This server does
that with **Reciprocal Rank Fusion** by default (score-scale agnostic — BM25 scores
and cosine similarities are not comparable), with `fusion="weighted"` and
`weights={"dense": 2, "text": 1}` available when you want to bias a signal.

Recipe 5 is the exception: on a single-vector index the Vectors API scores a dense
and a sparse vector together in **one** server-side request. That is what
`pinecone_query_vectors` is for.

---

## TTL — read this before relying on it

**Pinecone has no server-side record expiry.** Rather than pretend otherwise, the
toolbox implements TTL as an explicit convention:

- a record written with `ttl_seconds` carries `vtb_expires_at` (epoch seconds) as
  ordinary metadata; a record written without one carries nothing extra;
- searches exclude lapsed records by default, using
  `{"$or": [{"vtb_expires_at": {"$exists": false}}, {"vtb_expires_at": {"$gt": now}}]}` —
  the `$exists` half is what stops records with no TTL (including any written
  outside this server) from being hidden;
- `pinecone_purge_expired` actually deletes lapsed records — it is the only thing
  that reclaims storage. Run it on a schedule if TTL matters to you.

The field name is not cosmetic: Pinecone **rejects any field starting with `_`
or `$`** — `_` is reserved for `_id` and `_score`, `$` for filter operators — and
one invalid field fails the entire upsert request. `_expires_at` would have
broken every write.

---

## Embeddings are pluggable

Storing vectors is decoupled from producing them. Any tool that needs an embedding
takes `embed_provider` / `embed_model` / `embed_dimension`, falling back to
`VTB_EMBED_PROVIDER` / `VTB_EMBED_MODEL`.

| Provider | Dense | Sparse | Install |
|---|---|---|---|
| `pinecone` (hosted inference) | ✅ | ✅ | included |
| `openai` | ✅ | — | `pip install '.[openai]'` |
| `cohere` | ✅ | — | `pip install '.[cohere]'` |
| `huggingface` (sentence-transformers, local) | ✅ | — | `pip install '.[local]'` |

Documents that already carry a vector are left alone, so pre-computed and
generated vectors can be mixed in one upsert. Vector width is checked against the
schema before anything is sent.

There is also the fully hosted route: `pinecone_create_index_for_model` attaches a
Pinecone embedding model to the index, and `pinecone_search_records` embeds the
query server-side. No provider needed on this side.

---

## Constraints the toolbox checks for you

Each of these fails an entire request server-side, so they are validated before
the call goes out, with a message naming the offender:

| Constraint | Enforced where |
|---|---|
| At most 1 `dense_vector` and 1 `sparse_vector` field per index; up to 100 FTS fields | `pinecone_create_index` |
| Field names unique, ≤64 bytes, never `_`- or `$`-prefixed | create + every upsert |
| Every document needs an `_id` **and** at least one schema field (metadata-only documents are rejected) | `pinecone_upsert_documents` |
| One bad document fails the whole batch — so all offenders are reported at once | `pinecone_upsert_documents` |
| ≤1000 documents per request | batching |
| `top_k` between 1 and 10,000; ≤100 `score_by` clauses | every search |
| A `dense_vector`/`sparse_vector` clause must be alone in its request | every search |

Other documented limits worth designing around: 40 KB metadata per record, 2 MB
per document, 100 KB and 10,000 tokens per FTS field, 10,000 values per `$in`,
128 tokens per `$match_*` operator.

**Filter operators.** Metadata: `$eq $ne $gt $gte $lt $lte $in $nin $exists $and
$or $not`. On FTS fields additionally `$match_phrase`, `$match_all`,
`$match_any`. Lucene (`query_string`) supports boolean operators, phrases, phrase
slop, boosting, phrase prefix and regex — **but not fuzzy matching**.

**Multi-tenancy.** Use one namespace per tenant rather than a metadata filter
over a shared namespace: query cost scales with namespace size, so filtering
`user_id` across 100 GB costs 100× what querying a 1 GB namespace does.

**Eventual consistency.** A read immediately after a write may not see it. The
live smoke test polls rather than assuming.

---

## Tools

**Index lifecycle** — `pinecone_create_index`, `pinecone_create_index_for_model`,
`pinecone_list_indexes`, `pinecone_describe_index`, `pinecone_index_capabilities`,
`pinecone_configure_index`, `pinecone_delete_index`

**Namespaces** — `pinecone_list_namespaces`, `pinecone_describe_namespace`,
`pinecone_create_namespace`, `pinecone_delete_namespace`,
`pinecone_sample_metadata`, `pinecone_describe_index_stats`

**Writing** — `pinecone_upsert_documents`, `pinecone_upsert_vectors`,
`pinecone_update_documents`, `pinecone_update_vector`, `pinecone_delete_records`,
`pinecone_purge_expired`

**Reading** — `pinecone_fetch_records`, `pinecone_list_record_ids`

**Search** — `pinecone_search` (text / query_string / dense / sparse / hybrid / auto),
`pinecone_search_records` (integrated inference), `pinecone_query_vectors`
(Vectors API), `pinecone_rerank`

**Utilities** — `pinecone_embed`, `pinecone_list_models`, `vectortoolbox_status`

**Chroma** — clients: `chroma_create_client`, `chroma_list_clients`,
`chroma_remove_client`, `chroma_heartbeat` · collections:
`chroma_list_embedding_functions`, `chroma_create_collection`,
`chroma_get_collection`, `chroma_list_collections`, `chroma_modify_collection`,
`chroma_configure_collection`, `chroma_delete_collection`, `chroma_count`,
`chroma_peek`, `chroma_fork_collection` · records: `chroma_add`,
`chroma_update`, `chroma_upsert`, `chroma_conditional_transaction`,
`chroma_delete` · reading: `chroma_query`, `chroma_get`,
`chroma_full_text_search`, `chroma_validate_filter`, `chroma_sample_metadata`

Destructive tools (`pinecone_delete_index`, `pinecone_delete_namespace`,
`delete_all`, `pinecone_purge_expired`, `chroma_delete_collection`,
`chroma_delete` with `delete_all`) require `confirm=true`. Setting
`VTB_READ_ONLY=true` disables every write tool.

---

## Chroma

The second backend. Chroma's model is smaller than Pinecone's — a **collection**
is one embedding space, and each record is an id, an embedding, an optional
document string, flat metadata and an optional URI. There is no field schema;
what you choose up front is the **embedding route** and the **index config**.

Install the extra (it is optional so a Pinecone-only setup stays lean):

```bash
uv pip install --python .venv/bin/python -e '.[chroma]'
```

### Clients

Every `chroma_*` tool takes `client` (default `"default"`). Create more with
`chroma_create_client` and use them side by side:

| `kind` | What it is | Settings |
|---|---|---|
| `ephemeral` | In-memory; gone on restart. All ephemeral clients in the process share one store | — |
| `persistent` | Local directory | `path` (default `VTB_CHROMA_PATH` = `~/.vector-toolbox/chroma`) |
| `http` | Self-hosted server (`chroma run`, Docker) | `host`, `port`, `ssl`, `headers` |
| `cloud` | Chroma Cloud | key from `CHROMA_API_KEY` or `api_key_env_var`; `tenant`/`database` |

The `default` client is built from `VTB_CHROMA_CLIENT` (default `persistent`)
the first time a tool needs it. API keys are never accepted as tool arguments —
only the *name* of an environment variable — so they stay out of transcripts.

### Embedding: two routes, chosen per collection

* **Chroma embedding function** — `embedding_function={"name": "openai",
  "kwargs": {"model_name": "text-embedding-3-small", "api_key_env_var":
  "OPENAI_API_KEY"}}`. Chroma embeds documents and `query_texts` itself and
  persists the config with the collection. `chroma_list_embedding_functions`
  lists the ~30 built-ins.
* **Toolbox embedder** — `toolbox_embedder={"provider": "openai", "model": …,
  "dimension": …}`: the same providers the Pinecone tools use. This server
  embeds and sends plain vectors; the choice is recorded in the collection's
  metadata (`vtb_embed_*`) so later writes and queries embed the same way.

With neither, Chroma's `default` function (all-MiniLM-L6-v2, 384 dims) is used
and **downloads an ~80 MB ONNX model on first use**. Records that already carry
an embedding are never re-embedded.

### Index configuration

| | HNSW (single-node: ephemeral, persistent, most self-hosted) | SPANN (Chroma Cloud / distributed) |
|---|---|---|
| Fixed at creation | `space`, `ef_construction`, `max_neighbors` | `space`, `write_nprobe`, `ef_construction`, `max_neighbors`, `reassign_neighbor_count`, `split_threshold`, `merge_threshold` |
| Tunable later (`chroma_configure_collection`) | `ef_search`, `num_threads`, `batch_size`, `sync_threshold`, `resize_factor` | `search_nprobe` (≤128), `ef_search` |

`space` is `l2` (default), `ip` or `cosine` — pick `cosine` for most text
models. Single-node Chroma **silently ignores a SPANN config**, so the toolbox
refuses one on an ephemeral/persistent client rather than let it pass.

### Writes — what Chroma does silently, surfaced

| Chroma behaviour | What the toolbox reports |
|---|---|
| `add` skips ids that already exist, without error | `skipped_existing` (or fail with `on_conflict="error"`) |
| `update` skips ids that do not exist, without error | `skipped_missing` |
| `upsert` doesn't say what it did | `created` / `updated` counts |
| Mixed "some records have embeddings, some don't" fails | caught up front, or gaps filled when `embed_provider` is set |

Records can be passed as Chroma's parallel columns (`ids`, `documents`,
`embeddings`, `metadatas`, `uris`) or as `records=[{"id", "document", …}]`.
Metadata is flat: scalars or **non-empty, single-type arrays** (`{"genres":
["action", "drama"]}`). `update` merges metadata; a key set to `null` is removed.

### Conditional transactions

`chroma_conditional_transaction` is optimistic read-check-write: the checks are
read inside a transaction, the writes are buffered, and the commit only lands if
the records read are unchanged. Checks can require a record to exist / not
exist, hold specific metadata values (compare-and-swap), hold an exact
document, or a filter to match between `min_count` and `max_count` records.
Conflicts are retried up to `max_retries`.

Chroma's limits are validated before anything runs: one collection, at most one
write per id, deletes by explicit id only, reads via `get` only. This needs a
`chromadb` release with `Collection.conditional()` — **1.5.9 (current on PyPI
when this was written) does not have it**, so the tool refuses unless you pass
`allow_non_atomic=true` (same checks then writes, no isolation, reported as
`atomic: false`).

### Query, get and results shape

* `chroma_query` — nearest neighbours; a batch of `query_texts` or
  `query_embeddings`, each with its own `n_results` list.
* `chroma_get` — by id and/or filter, paged with `limit`/`offset`, no ranking.
* `include` picks what comes back: `documents`, `metadatas`, `embeddings`,
  `uris`, plus `distances` for queries. Ids always come back.
* `output="rows"` (default) returns one object per record, nearest first, with
  `similarity = 1 − distance` added on cosine/ip collections. `output="columns"`
  returns Chroma's native column-major shape (`ids[q][i]` for queries,
  `ids[i]` for get); `"both"` returns both.

### Metadata filtering and full-text search

| `where` (metadata) | |
|---|---|
| Comparison | `{"year": 2024}`, `$eq $ne`, numeric `$gt $gte $lt $lte` |
| Inclusion | `{"genre": {"$in": ["a", "b"]}}`, `$nin` |
| Arrays | `{"genres": {"$contains": "action"}}`, `$not_contains` |
| Logical | `$and`, `$or`, nestable |

| `where_document` (document text) | |
|---|---|
| Substring | `{"$contains": "refund"}`, `$not_contains` (case-sensitive) |
| Regex | `{"$regex": "(?i)^invoice \\d+"}`, `$not_regex` — no look-around or backreferences |
| Logical | `$and`, `$or` |

`chroma_validate_filter` shows exactly what will be sent. Forgiving spellings
are rewritten (`{}` → no filter, several keys → `$and`, several operators on one
field → `$and`, one-item `$and` → the item); real mistakes — a bare list
instead of `$in`/`$contains`, `$exists`, `$not`, null — come back as
explanations. `chroma_sample_metadata` reports each key's type (`str[]` marks
array metadata) and the operators that fit.

`chroma_full_text_search` builds `where_document` from term lists
(`contains`, `not_contains`, `regex`, `not_regex`, `match="all"|"any"`). On its
own it is a filter — matches come back unranked. Add `query_text` to **combine
with document search**: the text filter restricts a vector query, so you get
semantic ranking over only the documents that contain your terms.

---

## Tests

```bash
.venv/bin/pytest              # unit tests: Pinecone mocked, Chroma run in-memory for real
.venv/bin/ruff check src tests
.venv/bin/python scripts/smoke_test.py    # live; needs PINECONE_API_KEY
```

The live smoke test creates a scratch index prefixed `vtb-smoke-`, exercises each
of the five recipes, and deletes it again.

---

## Contributing

Adding a backend, a tool or a fix: see [CONTRIBUTING.md](CONTRIBUTING.md).
Security reports: [SECURITY.md](SECURITY.md). Changes: [CHANGELOG.md](CHANGELOG.md).

## License

[MIT](LICENSE)
