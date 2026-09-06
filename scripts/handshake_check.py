#!/usr/bin/env python3
"""Talk to the server the way an MCP client does: JSON-RPC over stdio.

    python scripts/handshake_check.py [path/to/vector-toolbox-mcp]

Runs initialize -> tools/list -> tools/call, from working directory "/" so that
it exercises the same conditions Claude Desktop launches the server under (a
cwd of "/" is why credentials have to come from a .env beside the project
rather than one found by searching upward from the caller).

Exits 0 on success, 1 with the server's stderr on failure.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

TIMEOUT = 30


def main() -> int:
    if len(sys.argv) > 1:
        binary = sys.argv[1]
    else:
        binary = str(Path(__file__).resolve().parents[1] / ".venv" / "bin" / "vector-toolbox-mcp")

    if not Path(binary).exists():
        print(f"not found: {binary}\nRun scripts/setup_macos.sh first.")
        return 1

    proc = subprocess.Popen(
        [binary],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        cwd="/",
    )

    def send(payload: dict) -> None:
        proc.stdin.write(json.dumps(payload) + "\n")
        proc.stdin.flush()

    def read() -> dict:
        while True:
            line = proc.stdout.readline()
            if not line:
                stderr = proc.stderr.read()
                print("server exited without responding.\n")
                print(stderr[:4000] or "(no stderr)")
                sys.exit(1)
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)

    try:
        send({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "handshake-check", "version": "1.0"},
            },
        })
        result = read()["result"]
        print(f"initialize   OK  {result['serverInfo']['name']} "
              f"(protocol {result['protocolVersion']})")

        send({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
        send({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        tools = read()["result"]["tools"]
        print(f"tools/list   OK  {len(tools)} tools")

        send({
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "vectortoolbox_status", "arguments": {}},
        })
        call = read()["result"]
        payload = call.get("structuredContent") or json.loads(call["content"][0]["text"])
        payload = payload.get("result", payload)
        print(f"tools/call   OK  vectortoolbox_status")
        print()
        print(f"  Pinecone key loaded : {payload['pinecone_api_key_set']}")
        print(f"  Region              : {payload['pinecone_region']}")
        print(f"  Dense embedder      : {payload['default_dense_embedder']}")
        print(f"  Sparse embedder     : {payload['default_sparse_embedder']}")
        print(f"  Read-only mode      : {payload['read_only']}")
        if payload["read_only"]:
            print("    -> every write tool will refuse; set VTB_READ_ONLY=false in .env")
        if not payload["pinecone_api_key_set"]:
            print("    -> PINECONE_API_KEY is not reaching the server")
        provider, _, model = payload["default_dense_embedder"].partition("/")
        if provider == "pinecone" and model.startswith(("text-embedding-", "embed-")):
            print(f"    -> {model!r} is not a Pinecone-hosted model; "
                  "set VTB_EMBED_PROVIDER to match")

        # The provider packages are optional extras, and a missing one only
        # surfaces at the first embed call - which is a long way from here.
        needed = {
            "openai": ("openai", "openai"),
            "cohere": ("cohere", "cohere"),
            "huggingface": ("sentence_transformers", "local"),
        }.get(provider)
        if needed:
            module, extra = needed
            probe = subprocess.run(
                [str(Path(binary).parent / "python"), "-c", f"import {module}"],
                capture_output=True,
            )
            if probe.returncode != 0:
                venv = Path(binary).parent.parent
                # `uv venv` does not install pip, so name the installer that
                # actually exists in this virtualenv.
                if (venv / "bin" / "pip").exists():
                    fix = f"{venv}/bin/pip install -e '.[{extra}]'"
                else:
                    fix = f"uv pip install --python {venv}/bin/python -e '.[{extra}]'"
                print(f"    -> the {module!r} package is not installed, so this provider "
                      f"will fail at the first embed call")
                print(f"       fix: {fix}")
            else:
                print(f"    -> {module!r} is installed")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
