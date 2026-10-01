"""Streamable-HTTP transport: the ASGI app, bearer-token auth and a health check.

stdio is what desktop clients launch; HTTP is for running the server once
(in Docker, on a VM) and pointing one or more clients at its URL. Two things
are added around the SDK's app:

* ``GET /health`` - unauthenticated liveness probe for Docker/Kubernetes.
* Bearer-token auth - when ``VTB_AUTH_TOKEN`` is set, every other request
  must carry ``Authorization: Bearer <token>``. Compared in constant time.
  There is deliberately no way to pass the token on the command line: it
  would show up in ``ps`` output and shell history.
"""

from __future__ import annotations

import hmac
import json
from typing import Any, Awaitable, Callable

Scope = dict[str, Any]
Receive = Callable[[], Awaitable[dict[str, Any]]]
Send = Callable[[dict[str, Any]], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

LOOPBACK = ("127.0.0.1", "localhost", "::1")


async def _json(send: Send, status: int, body: dict[str, Any], extra: list | None = None) -> None:
    payload = json.dumps(body).encode()
    headers = [(b"content-type", b"application/json"), (b"content-length", str(len(payload)).encode())]
    await send({"type": "http.response.start", "status": status, "headers": headers + (extra or [])})
    await send({"type": "http.response.body", "body": payload})


class GuardMiddleware:
    """Health endpoint + optional bearer-token check in front of the MCP app."""

    def __init__(self, app: ASGIApp, *, token: str | None, health_path: str = "/health",
                 version: str = "") -> None:
        self.app = app
        self.token = token.encode() if token else None
        self.health_path = health_path
        self.version = version

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":  # lifespan, websocket: pass straight through
            await self.app(scope, receive, send)
            return
        if scope.get("path") == self.health_path and scope.get("method") in {"GET", "HEAD"}:
            await _json(send, 200, {"status": "ok", "version": self.version})
            return
        if self.token is not None and scope.get("method") != "OPTIONS":
            supplied = b""
            for name, value in scope.get("headers", []):
                if name.lower() == b"authorization":
                    supplied = value
                    break
            scheme, _, credential = supplied.partition(b" ")
            if scheme.lower() != b"bearer" or not hmac.compare_digest(credential.strip(), self.token):
                await _json(
                    send, 401,
                    {"error": "unauthorized", "message": "Send 'Authorization: Bearer <VTB_AUTH_TOKEN>'."},
                    [(b"www-authenticate", b'Bearer realm="vector-toolbox"')],
                )
                return
        await self.app(scope, receive, send)


def build_http_app(
    mcp: Any,
    *,
    host: str,
    path: str = "/mcp",
    token: str | None = None,
    allowed_hosts: list[str] | None = None,
    stateless: bool = False,
    version: str = "",
) -> ASGIApp:
    """Wrap the SDK's streamable-HTTP app. Works with MCP SDK 1.x and 2.x."""
    security = None
    if allowed_hosts:
        from mcp.server.transport_security import TransportSecuritySettings

        security = TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=allowed_hosts,
            allowed_origins=[f"http://{h}" for h in allowed_hosts] + [f"https://{h}" for h in allowed_hosts],
        )
    try:  # SDK 2.x: options are arguments
        app = mcp.streamable_http_app(
            streamable_http_path=path, stateless_http=stateless,
            transport_security=security, host=host,
        )
    except TypeError:  # SDK 1.x: options live on settings
        mcp.settings.host = host
        mcp.settings.streamable_http_path = path
        mcp.settings.stateless_http = stateless
        if security is not None:
            mcp.settings.transport_security = security
        app = mcp.streamable_http_app()
    return GuardMiddleware(app, token=token, version=version)


def serve(mcp: Any, *, host: str, port: int, path: str, token: str | None,
          allowed_hosts: list[str] | None, stateless: bool, version: str, log_level: str = "info") -> None:
    import sys

    import uvicorn

    if token is None and host not in LOOPBACK:
        print(
            f"WARNING: serving on {host}:{port} with no VTB_AUTH_TOKEN - anyone who can reach this "
            "port can read and write your vector stores. Set VTB_AUTH_TOKEN.",
            file=sys.stderr,
        )
    app = build_http_app(mcp, host=host, path=path, token=token, allowed_hosts=allowed_hosts,
                         stateless=stateless, version=version)
    print(f"vector-toolbox {version}: streamable HTTP on http://{host}:{port}{path} "
          f"(auth {'on' if token else 'off'}, health at /health)", file=sys.stderr)
    uvicorn.run(app, host=host, port=port, log_level=log_level)
