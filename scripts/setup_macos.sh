#!/usr/bin/env bash
# Create the macOS virtualenv Claude Desktop will launch, and print the config
# block to paste into claude_desktop_config.json.
#
#   bash scripts/setup_macos.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if command -v uv >/dev/null 2>&1; then
  uv venv .venv
  VIRTUAL_ENV="$ROOT/.venv" uv pip install -e .
else
  python3 -m venv .venv
  ./.venv/bin/pip install --quiet --upgrade pip
  ./.venv/bin/pip install --quiet -e .
fi

BIN="$ROOT/.venv/bin/vector-toolbox-mcp"
test -x "$BIN"

# Prove the server actually speaks MCP over stdio before wiring it into a client.
if ! "$ROOT/.venv/bin/python" "$ROOT/scripts/handshake_check.py" "$BIN"; then
  echo
  echo "Handshake failed. Run '$BIN' by hand to see the startup error."
  exit 1
fi

CONFIG="$HOME/Library/Application Support/Claude/claude_desktop_config.json"
cat <<EOF

Add this to: $CONFIG

{
  "mcpServers": {
    "vector-toolbox": {
      "command": "$BIN"
    }
  }
}

Then quit Claude Desktop completely (Cmd+Q, not just the window) and reopen it.
Credentials are read from $ROOT/.env - no env block needed in the config.
EOF
