#!/usr/bin/env python3
"""Back-compat wrapper: register this checkout's venv in Claude Desktop.

Prefer the packaged installer, which also handles Antigravity, Cursor, uvx and Docker:

    vector-toolbox-install claude-desktop --mode source
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vectortoolbox.installer import main  # noqa: E402

args = sys.argv[1:]
if "--remove" in args:
    raise SystemExit(main(["claude-desktop", "--remove"]))
raise SystemExit(main(["claude-desktop", "--mode", "source", *args]))
