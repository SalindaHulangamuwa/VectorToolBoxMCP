# Security

## Reporting a vulnerability

Please do **not** open a public issue. Use GitHub's private vulnerability
reporting (Security tab -> "Report a vulnerability") on this repository.
You should get a reply within a week.

## Deployment notes

This server can read, write and delete data in every vector store its
credentials reach. Treat access to it like access to those credentials.

- **stdio** (the default) is only reachable by the client that launched it.
- **HTTP** listens on `127.0.0.1` by default. Before binding to anything
  else, set `VTB_AUTH_TOKEN` (the server warns at start-up when you don't) and
  put TLS in front of it (a reverse proxy). Set `VTB_ALLOWED_HOSTS` to the
  host names you serve on to enable DNS-rebinding protection.
- `VTB_READ_ONLY=true` disables every write and delete tool - a good default
  when pointing an agent at production data.
- Destructive tools require `confirm=true`, but an agent can pass it. Use
  read-only mode or scoped API keys where that matters.
- Tools never accept raw API keys as arguments; keys come from the
  environment so they stay out of conversation logs.
- Never commit `.env`. The Docker image excludes it; pass secrets at run time.
