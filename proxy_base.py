"""Shared leaf code for proxy.py and embed_proxy.py.

Mechanical extraction — code copied verbatim from proxy.py (canonical source).
A later phase rewrites proxy.py and embed_proxy.py to import from here.
"""
from __future__ import annotations

import asyncio
import logging
import os
import secrets
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from aiohttp import ClientError, ClientSession, ClientTimeout, web
from aiohttp.web_log import AccessLogger

from log_paths import (
    DATE_FMT,
    LocalTzFormatter,
    current_week_dir,
    local_now,
)

if TYPE_CHECKING:
    from router_manager import RouterManager

# Enable ANSI escape sequences on Windows 10+
if sys.platform == "win32":
    os.system("")


ROOT = Path(__file__).resolve().parent


def _load_dotenv(path: Path) -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ (non-overwrite)."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("\"'")
        os.environ.setdefault(key, value)


_load_dotenv(ROOT / ".env")

API_KEY = os.environ.get("LLAMA_API_KEY")

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


class DeadWorkerError(Exception):
    """Raised when the router returns a 500 whose body indicates the worker
    child has died. Caught by proxy_request to trigger a recovery + retry."""

    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self.body = body
        super().__init__(f"dead-worker {status}: {body[:200]!r}")


class ClientGone(Exception):
    """Raised when the *downstream* client hangs up before we finish sending the
    response (e.g. a connection reset during downstream.prepare()). Benign — it is
    not a proxy fault, so it is logged calmly and returned as 499, unlike an
    *upstream* reset from the router which is a real bad gateway (502)."""


# Lowercased substrings that identify a dead-worker 500 from the router.
# The router returns these when its HTTP client can't reach the worker child.
DEAD_WORKER_MARKERS: tuple[str, ...] = (
    "could not establish connection",
    "failed to read connection",
    "failed to write connection",
    "http client error",
)


def _is_dead_worker_response(status: int, body: bytes) -> bool:
    """Return True when *status* ≥ 500 and *body* contains a dead-worker marker."""
    if status < 500:
        return False
    lowered = body.lower()
    return any(m.encode() in lowered for m in DEAD_WORKER_MARKERS)


@dataclass(frozen=True)
class ProxyConfig:
    proxy_host: str
    proxy_port: int
    server_host: str
    server_port: int
    idle_timeout: int
    idle_check_interval: int
    health_poll_interval: float
    boot_timeout: int
    default_model: str
    api_key: str
    embed_host: str | None = None
    embed_port: int | None = None
    chat_log: bool = True

    @property
    def backend_base_url(self) -> str:
        return f"http://{self.server_host}:{self.server_port}"

    @property
    def embed_base_url(self) -> str:
        if self.embed_host is None or self.embed_port is None:
            raise AttributeError("embed_host/embed_port not set on this config")
        host = self.embed_host
        if ":" in host:
            host = f"[{host}]"  # IPv6 needs brackets in URLs
        return f"http://{host}:{self.embed_port}"

def filter_request_headers(headers, api_key: str) -> dict[str, str]:
    forwarded: dict[str, str] = {}
    for name, value in headers.items():
        lowered = name.lower()
        if lowered in ("host", "content-length") or lowered in HOP_BY_HOP_HEADERS:
            continue
        forwarded[name] = value
    if api_key and "Authorization" not in forwarded:
        forwarded["Authorization"] = f"Bearer {api_key}"
    return forwarded


def filter_response_headers(headers) -> dict[str, str]:
    forwarded: dict[str, str] = {}
    for name, value in headers.items():
        lowered = name.lower()
        if lowered == "content-length" or lowered in HOP_BY_HOP_HEADERS:
            continue
        forwarded[name] = value
    return forwarded


# Reverse-proxy hops allowed to set forwarding headers. We listen on
# 0.0.0.0:8001, so a direct LAN client could spoof X-Forwarded-For /
# CF-Connecting-IP; only trust those headers when the TCP peer is Caddy (LAN
# front) or cloudflared (loopback). Override the Caddy IP via $TRUSTED_PROXY.
TRUSTED_PROXIES = {"127.0.0.1", "::1", os.environ.get("TRUSTED_PROXY", "192.168.178.43")}


def client_ip(request: web.Request) -> str:
    """Real client IP. When the request arrives from a trusted reverse proxy
    (Caddy on the LAN, or cloudflared on loopback) we read the forwarded client
    out of CF-Connecting-IP / X-Forwarded-For; otherwise we report the raw TCP
    peer. Untrusted peers can't spoof their way to a fake IP."""
    peer = request.remote or "-"
    if peer in TRUSTED_PROXIES:
        forwarded = (
            request.headers.get("CF-Connecting-IP")
            or request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        )
        if forwarded:
            return forwarded
    return peer


class ForwardedAccessLogger(AccessLogger):
    """aiohttp access logger that resolves the ``%a`` atom through client_ip(),
    so the access line shows the real client instead of the reverse-proxy hop."""

    @staticmethod
    def _format_a(request, response, time):
        if request is None:
            return "-"
        return client_ip(request)


# Paths reachable without an API key. /health is the cloudflared/uptime probe.
PUBLIC_PATHS = {"/health"}


@web.middleware
async def auth_middleware(request: web.Request, handler):
    """Reject any request that doesn't carry the configured API key.

    This proxy is the internet-facing origin for the Cloudflare tunnel, so
    it — not the localhost-only llama-server behind it — is where client
    authentication has to happen. filter_request_headers() still injects the
    key on the *upstream* hop so the backend keeps trusting only this proxy.
    """
    config: ProxyConfig = request.app["config"]
    if not config.api_key or request.path.rstrip("/") in PUBLIC_PATHS:
        return await handler(request)
    header = request.headers.get("Authorization", "")
    token = header[7:].strip() if header[:7].lower() == "bearer " else ""
    if not token or not secrets.compare_digest(token, config.api_key):
        logging.warning(
            "401 unauthorized: %s %s from %s",
            request.method, request.path, client_ip(request),
        )
        return web.json_response({"error": "unauthorized"}, status=401)
    return await handler(request)


def configure_logging(log_stem: str = "proxy") -> None:
    log_dir = ROOT / "logs"
    week_dir = current_week_dir(log_dir)
    log_file = week_dir / f"{log_stem}-{local_now().strftime(DATE_FMT)}.log"
    fmt = LocalTzFormatter("%(asctime)s %(levelname)s %(message)s")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt)
    logging.basicConfig(level=logging.INFO, handlers=[console, file_handler])
    logging.info("Log file: %s", log_file)


async def health_handler(request: web.Request) -> web.Response:
    manager: RouterManager = request.app["manager"]
    return web.json_response({
        "status": "ok",
        "router": "running" if manager.server_running else "down",
        "loaded": manager.model_loaded,
        "active_requests": manager.active_requests,
    })


async def idle_watchdog(manager: RouterManager) -> None:
    while True:
        await asyncio.sleep(manager.config.idle_check_interval)
        try:
            await manager.unload_if_idle()
        except Exception:
            logging.exception("idle watchdog error")
