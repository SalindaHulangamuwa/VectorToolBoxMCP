# Vector Toolbox MCP

One MCP server, many vector databases — the same idea as Google's MCP Toolbox for
databases, applied to vector stores. **Pinecone is the first backend**; the tool
surface, backend contract and embedding layer are built so Qdrant, Weaviate,
Milvus or pgvector slot in behind the same shape.

Built against the **Pinecone Python SDK v10** — the Documents API with declared
field schemas, not the older `dimension`/`metric` index model.

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

Destructive tools (`pinecone_delete_index`, `pinecone_delete_namespace`,
`delete_all`, `pinecone_purge_expired`) require `confirm=true`. Setting
`VTB_READ_ONLY=true` disables every write tool.

---

## Setup — connecting to Claude Desktop

The virtualenv must be created **on the machine Claude Desktop runs on**, since
the config points at a binary inside it.

```bash
cd ~/VectorToolBoxMCP
bash scripts/setup_macos.sh
```

Optional embedding providers are extras. Note that `uv venv` creates a
virtualenv **without pip**, so install them through uv:

```bash
uv pip install --python .venv/bin/python -e '.[openai]'    # or [cohere], [local], [all]
```

That creates `.venv`, installs the package, verifies the server answers a real
MCP handshake over stdio, and prints the config block with your absolute path
already filled in.

Then edit `~/Library/Application Support/Claude/claude_desktop_config.json`
(create it if absent — Claude Desktop also opens it from
Settings → Developer → Edit Config):

```json
{
  "mcpServers": {
    "vector-toolbox": {
      "command": "/Users/user/VectorToolBoxMCP/.venv/bin/vector-toolbox-mcp"
    }
  }
}
```

**Quit Claude Desktop completely** (Cmd+Q — closing the window is not enough)
and reopen it. The tools appear under the connectors/tools menu.

No `env` block is needed: `config.py` looks for a `.env` beside the project, not
beside the caller, because MCP clients launch servers with an unpredictable
working directory (Claude Desktop uses `/`). If you would rather keep secrets in
the client config, an `env` object on the server entry still works and takes
precedence over nothing — `.env` values are only applied to variables not
already set.

### Claude Code

```bash
claude mcp add vector-toolbox -- /Users/user/VectorToolBoxMCP/.venv/bin/vector-toolbox-mcp
```

### If it does not show up

| Symptom | Cause |
|---|---|
| Server missing from the menu | Claude Desktop was reloaded, not quit and reopened |
| "spawn ENOENT" | The `command` path is wrong, or the venv was built on a different OS |
| Server appears, every call errors | `PINECONE_API_KEY` not reaching it — run the binary by hand and call `vectortoolbox_status` |
| Only read tools work | `VTB_READ_ONLY=true` in `.env` |

Run the binary directly to see startup errors that the client swallows:

```bash
./.venv/bin/vector-toolbox-mcp
```

It will sit waiting for JSON-RPC on stdin; a traceback instead means the import
or config failed.

## Tests

```bash
.venv/bin/pytest              # unit tests, Pinecone fully mocked
.venv/bin/python scripts/smoke_test.py    # live; needs PINECONE_API_KEY
```

The live smoke test creates a scratch index prefixed `vtb-smoke-`, exercises each
of the five recipes, and deletes it again.

---

## Adding a second vector database

1. Implement `vectortoolbox.core.base.VectorStoreBackend`.
2. `register_backend("qdrant", QdrantBackend)` in the backend package's `__init__`.
3. Add a `tools.py` with `qdrant_*` tools and call its `register(mcp)` from
   `server.build_server`.

Capability reporting (`IndexCapabilities`) and result fusion
(`core/fusion.py`) are backend-neutral and get reused as-is.
