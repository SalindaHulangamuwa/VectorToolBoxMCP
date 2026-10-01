"""Register vector-toolbox in an MCP client's config file without clobbering it.

    vector-toolbox-install claude-desktop                 # uvx (default)
    vector-toolbox-install antigravity --mode docker
    vector-toolbox-install cursor --mode source
    vector-toolbox-install antigravity --mode http --url http://localhost:8000/mcp
    vector-toolbox-install claude-desktop --remove
    vector-toolbox-install claude-desktop --print         # show the entry, write nothing

The existing file is backed up (``.bak``) and every other server in it is kept.
Secrets are never copied into the config: point ``--env-file`` at a .env and
the server reads it at start-up, or add an ``env`` block by hand.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

NAME = "vector-toolbox"
PACKAGE = "vector-toolbox-mcp"
IMAGE = "ghcr.io/salindahulangamuwa/vector-toolbox-mcp:latest"
CLIENTS = ("claude-desktop", "antigravity", "cursor")
MODES = ("uvx", "source", "docker", "http")
# Env vars a docker-mode entry forwards from the client's env block / your shell.
DOCKER_FORWARD = ("PINECONE_API_KEY", "OPENAI_API_KEY", "COHERE_API_KEY", "CHROMA_API_KEY")


def config_path(client: str) -> Path:
    home = Path.home()
    if client == "claude-desktop":
        if sys.platform == "darwin":
            return home / "Library/Application Support/Claude/claude_desktop_config.json"
        if sys.platform.startswith("win"):
            return Path(os.environ.get("APPDATA", home / "AppData/Roaming")) / "Claude/claude_desktop_config.json"
        return home / ".config/Claude/claude_desktop_config.json"
    if client == "antigravity":
        return home / ".gemini/config/mcp_config.json"
    if client == "cursor":
        return home / ".cursor/mcp.json"
    raise ValueError(client)


def _source_binary() -> Path:
    root = Path(__file__).resolve().parents[2]
    exe = "Scripts/vector-toolbox-mcp.exe" if sys.platform.startswith("win") else "bin/vector-toolbox-mcp"
    return root / ".venv" / exe


def build_entry(
    client: str,
    mode: str,
    *,
    extras: str = "chroma",
    env_file: str | None = None,
    url: str | None = None,
    token_env: str | None = None,
) -> dict[str, Any]:
    if mode == "http":
        if client == "claude-desktop":
            raise SystemExit(
                "Claude Desktop's config file only launches local (stdio) servers. Use --mode "
                "uvx/docker/source, or add the HTTP URL as a custom connector in Claude's settings."
            )
        if not url:
            raise SystemExit("--mode http needs --url, e.g. http://localhost:8000/mcp")
        key = "serverUrl" if client == "antigravity" else "url"
        entry: dict[str, Any] = {key: url}
        if token_env:
            token = os.environ.get(token_env)
            if not token:
                raise SystemExit(f"{token_env} is not set in this shell.")
            entry["headers"] = {"Authorization": f"Bearer {token}"}
        return entry

    server_args: list[str] = []
    if env_file:
        server_args += ["--env-file", str(Path(env_file).expanduser().resolve())]

    if mode == "uvx":
        uvx = shutil.which("uvx")
        if not uvx:
            raise SystemExit("uvx not found. Install uv: https://docs.astral.sh/uv/getting-started/installation/")
        spec = f"{PACKAGE}[{extras}]" if extras else PACKAGE
        # Absolute path: GUI apps (Claude Desktop on macOS) do not inherit your shell PATH.
        return {"command": uvx, "args": ["--from", spec, PACKAGE, *server_args]}

    if mode == "source":
        binary = _source_binary()
        if not binary.exists():
            raise SystemExit(f"not found: {binary}\nCreate the venv first: uv venv && uv pip install -e '.[{extras}]'")
        return {"command": str(binary), "args": server_args} if server_args else {"command": str(binary)}

    if mode == "docker":
        docker = shutil.which("docker") or "docker"
        args = ["run", "-i", "--rm", "-e", "VTB_TRANSPORT=stdio", "-v", "vector-toolbox-data:/data"]
        if env_file:
            args += ["--env-file", str(Path(env_file).expanduser().resolve())]
        else:
            for name in DOCKER_FORWARD:  # value comes from the entry's env block
                args += ["-e", name]
        return {"command": docker, "args": [*args, IMAGE]}

    raise ValueError(mode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vector-toolbox-install", description=__doc__.split("\n\n")[0])
    parser.add_argument("client", choices=CLIENTS)
    parser.add_argument("--mode", choices=MODES, default="uvx")
    parser.add_argument("--extras", default="chroma",
                        help="Optional dependencies for uvx mode, e.g. chroma,openai (default: chroma).")
    parser.add_argument("--env-file", help="A .env the server should load (keys stay out of the config).")
    parser.add_argument("--url", help="Server URL for --mode http.")
    parser.add_argument("--token-env", help="For --mode http: env var holding the bearer token to embed.")
    parser.add_argument("--config", help="Config file to edit instead of the client's default.")
    parser.add_argument("--remove", action="store_true", help="Unregister instead.")
    parser.add_argument("--print", dest="print_only", action="store_true", help="Print the entry only.")
    args = parser.parse_args(argv)

    entry = None if args.remove else build_entry(
        args.client, args.mode, extras=args.extras, env_file=args.env_file,
        url=args.url, token_env=args.token_env,
    )
    if args.print_only:
        print(json.dumps({"mcpServers": {NAME: entry}}, indent=2))
        return 0

    path = Path(args.config).expanduser() if args.config else config_path(args.client)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if path.exists() and path.read_text().strip():
        try:
            existing = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            print(f"{path} is not valid JSON ({exc}). Fix or move it, then re-run.")
            return 1
        backup = path.with_name(path.name + ".bak")
        shutil.copy(path, backup)
        print(f"backed up  {backup}")

    servers = existing.setdefault("mcpServers", {})
    kept = sorted(s for s in servers if s != NAME)
    if args.remove:
        servers.pop(NAME, None)
    else:
        servers[NAME] = entry
    path.write_text(json.dumps(existing, indent=2) + "\n")

    print(f"written    {path}")
    if kept:
        print(f"preserved  {', '.join(kept)}")
    if not args.remove and args.mode == "docker" and not args.env_file:
        print(f"note       add an \"env\" block with the keys you use ({', '.join(DOCKER_FORWARD)}).")
    restart = {
        "claude-desktop": "Quit Claude Desktop completely (Cmd+Q / exit from the tray) and reopen it.",
        "antigravity": "In Antigravity: agent panel ... > MCP Servers > Manage MCP Servers > refresh.",
        "cursor": "In Cursor: Settings > MCP, toggle the server off and on.",
    }[args.client]
    print(f"\n{restart}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
