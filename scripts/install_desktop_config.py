#!/usr/bin/env python3
"""Add this server to claude_desktop_config.json without clobbering it.

    python scripts/install_desktop_config.py [--remove]

The config file usually already lists other MCP servers, so this merges one
entry in and backs the file up first, rather than replacing it. Re-running is
safe: the entry is just overwritten with the same value.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

NAME = "vector-toolbox"


def config_path() -> Path:
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library/Application Support/Claude/claude_desktop_config.json"
    if sys.platform.startswith("win"):
        return home / "AppData/Roaming/Claude/claude_desktop_config.json"
    return home / ".config/Claude/claude_desktop_config.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remove", action="store_true", help="unregister instead")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    binary = root / ".venv" / "bin" / "vector-toolbox-mcp"
    if not args.remove and not binary.exists():
        print(f"not found: {binary}\nRun scripts/setup_macos.sh first.")
        return 1

    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    if path.exists() and path.read_text().strip():
        try:
            existing = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            print(f"{path} is not valid JSON ({exc}). Fix or move it, then re-run.")
            return 1
        backup = path.with_suffix(".json.bak")
        shutil.copy(path, backup)
        print(f"backed up  {backup}")

    servers = existing.setdefault("mcpServers", {})
    before = sorted(servers)

    if args.remove:
        servers.pop(NAME, None)
    else:
        servers[NAME] = {"command": str(binary)}

    path.write_text(json.dumps(existing, indent=2) + "\n")

    print(f"written    {path}")
    print(f"servers    {', '.join(sorted(servers)) or '(none)'}")
    kept = [s for s in before if s != NAME]
    if kept:
        print(f"preserved  {', '.join(kept)}")
    print("\nNow quit Claude Desktop completely (Cmd+Q) and reopen it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
