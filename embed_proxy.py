from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import secrets
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from aiohttp import ClientError, ClientSession, ClientTimeout, web

# Enable ANSI escape sequences on Windows 10+
if sys.platform == "win32":
    os.system("")


ROOT = Path(__file__).resolve().parent
SERVER_EXE = ROOT / "llama.cpp_latest" / "llama-server.exe"
PRESET_PATH = ROOT / "embed-preset.ini"

# Single embedding model. Wrapped in the same router-mode pattern as
# proxy.py so the embedder can be unloaded after idle while the router
# stays up — cloudflared keeps a stable origin.
MODEL_FILE = ROOT / "models" / "Qwen3-Embedding-4B-Q8_0.gguf"
MODEL_ID = "qwen3-embedding-4b-8k"
CTX_SIZE = 8192

PROXY_HOST = "::"
PROXY_PORT = 8003          # cloudflared origin (embed.htk-hrt.cc)
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8004         # internal router
IDLE_TIMEOUT = 600
IDLE_CHECK_INTERVAL = 30
HEALTH_POLL_INTERVAL = 1.0
BOOT_TIMEOUT = 60
LOAD_TIMEOUT = 120
RETRY_AFTER_SECONDS = 30
API_KEY = os.environ.get("LLAMA_API_KEY", "ZXY0UVZt8lbPVj3fSTC4gp0JatpRfOBQqGDAcvaVl3RjmWoq")

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

LOAD_TRIGGER_PATHS = {"/v1/embeddings", "/embeddings", "/v1/rerank", "/rerank"}


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

    @property
    def backend_base_url(self) -> str:
        return f"http://{self.server_host}:{self.server_port}"

    @property
    def server_command(self) -> list[str]:
        log_file = ROOT / "logs" / f"embed-server-{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.log"
        log_file.parent.mkdir(exist_ok=True)
        return [
            str(SERVER_EXE),
            "--log-file", str(log_file),
            "--models-preset", str(PRESET_PATH),
            "--models-max", "1",
            "--no-models-autoload",
            "--host", self.server_host,
            "--port", str(self.server_port),
            "--api-key", self.api_key,
        ]


class ModelManager:
    """Mirror of proxy.py's ModelManager, scoped to the embedding preset."""

    def __init__(self, config: ProxyConfig, session: ClientSession) -> None:
        self.config = config
        self.session = session
        self.process: asyncio.subprocess.Process | None = None
        self._loaded: str | None = None
        self._load_lock = asyncio.Lock()
        self._active = 0
        self._last_activity = time.monotonic()

    @property
    def server_running(self) -> bool:
        return self.process is not None and self.process.returncode is None

    @property
    def model_loaded(self) -> str | None:
        return self._loaded

    @property
    def active_requests(self) -> int:
        return self._active

    def begin_request(self) -> None:
        self._active += 1
        self._last_activity = time.monotonic()

    def end_request(self) -> None:
        self._active = max(0, self._active - 1)
        self._last_activity = time.monotonic()

    async def start_server(self) -> None:
        if self.server_running:
            return
        logging.info(
            "Starting embed router on %s:%s", self.config.server_host, self.config.server_port,
        )
        self.process = await asyncio.create_subprocess_exec(
            *self.config.server_command, cwd=str(ROOT),
        )
        deadline = time.monotonic() + self.config.boot_timeout
        while time.monotonic() < deadline:
            if self.process.returncode is not None:
                raise RuntimeError(f"router exited during boot: {self.process.returncode}")
            try:
                async with self.session.get(
                    f"{self.config.backend_base_url}/health",
                    timeout=ClientTimeout(total=5),
                ) as r:
                    if r.status == 200:
                        logging.info("embed router is ready")
                        return
            except (ClientError, asyncio.TimeoutError):
                pass
            await asyncio.sleep(self.config.health_poll_interval)
        raise TimeoutError(f"embed router did not become healthy in {self.config.boot_timeout}s")

    async def stop_server(self) -> None:
        if not self.server_running:
            return
        logging.info("Stopping embed router")
        try:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self.process.kill()
                await asyncio.wait_for(self.process.wait(), timeout=5)
        except ProcessLookupError:
            pass
        self.process = None
        self._loaded = None

    async def ensure_loaded(self, model: str) -> None:
        async with self._load_lock:
            if self._loaded == model:
                return
            current_status = await self._status(model)
            if current_status == "loaded":
                self._loaded = model
                return
            logging.info("Loading model: %s", model)
            url = f"{self.config.backend_base_url}/models/load"
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            }
            async with self.session.post(url, headers=headers, json={"model": model}) as r:
                if r.status >= 400:
                    body = await r.text()
                    if "already running" in body:
                        self._loaded = model
                        return
                    raise RuntimeError(f"load returned {r.status}: {body}")
            deadline = time.monotonic() + LOAD_TIMEOUT
            while time.monotonic() < deadline:
                status = await self._status(model)
                if status == "loaded":
                    self._loaded = model
                    logging.info("Model loaded: %s", model)
                    return
                if status == "failed":
                    raise RuntimeError(f"model {model} failed to load")
                await asyncio.sleep(0.5)
            raise TimeoutError(f"model {model} did not load in {LOAD_TIMEOUT}s")

    async def unload(self, reason: str) -> None:
        if self._loaded is None:
            return
        model = self._loaded
        logging.info("Unloading %s (%s)", model, reason)
        url = f"{self.config.backend_base_url}/models/unload"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        try:
            async with self.session.post(url, headers=headers, json={"model": model}) as r:
                if r.status >= 400:
                    logging.warning("unload returned %s", r.status)
        except ClientError as e:
            logging.warning("unload error: %s", e)
        self._loaded = None

    async def _status(self, model: str) -> str:
        url = f"{self.config.backend_base_url}/v1/models"
        headers = {"Authorization": f"Bearer {self.config.api_key}"}
        async with self.session.get(url, headers=headers) as r:
            data = await r.json()
        for entry in data.get("data", []):
            if entry.get("id") == model:
                status = entry.get("status") or {}
                if isinstance(status, dict):
                    return status.get("value", "unknown")
                return str(status)
        return "unknown"

    async def unload_if_idle(self) -> None:
        if self._active != 0 or self._loaded is None:
            return
        idle = time.monotonic() - self._last_activity
        if idle >= self.config.idle_timeout:
            await self.unload(f"idle for {int(idle)}s")


def write_preset() -> None:
    content = (
        f"[{MODEL_ID}]\n"
        f"model         = {MODEL_FILE.as_posix()}\n"
        f"ctx-size      = {CTX_SIZE}\n"
        f"embedding     = 1\n"
        f"pooling       = last\n"
        f"n-gpu-layers  = 999\n"
        f"flash-attn    = on\n"
        f"no-mmap       = 1\n"
    )
    PRESET_PATH.write_text(content, encoding="utf-8")


def configure_logging() -> None:
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"embed-proxy-{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.log"
    logging.Formatter.converter = time.gmtime
    fmt = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt)
    logging.basicConfig(level=logging.INFO, handlers=[console, file_handler])
    logging.info("Log file: %s", log_file)


def build_config() -> ProxyConfig:
    p = argparse.ArgumentParser(description="Router-mode proxy for the embedding model")
    p.add_argument("--proxy-host", default=PROXY_HOST)
    p.add_argument("--proxy-port", type=int, default=PROXY_PORT)
    p.add_argument("--server-host", default=SERVER_HOST)
    p.add_argument("--server-port", type=int, default=SERVER_PORT)
    p.add_argument("--idle-timeout", type=int, default=IDLE_TIMEOUT)
    p.add_argument("--idle-check-interval", type=int, default=IDLE_CHECK_INTERVAL)
    p.add_argument("--health-poll-interval", type=float, default=HEALTH_POLL_INTERVAL)
    p.add_argument("--boot-timeout", type=int, default=BOOT_TIMEOUT)
    p.add_argument("--api-key", default=API_KEY)
    args = p.parse_args()

    if not SERVER_EXE.exists():
        raise SystemExit(f"Missing required file: {SERVER_EXE}")
    if not MODEL_FILE.exists():
        raise SystemExit(f"Missing embedding model: {MODEL_FILE}")

    write_preset()
    print(f"Embed proxy default: {MODEL_ID} @ {CTX_SIZE // 1024}k ctx")

    return ProxyConfig(
        proxy_host=args.proxy_host,
        proxy_port=args.proxy_port,
        server_host=args.server_host,
        server_port=args.server_port,
        idle_timeout=args.idle_timeout,
        idle_check_interval=args.idle_check_interval,
        health_poll_interval=args.health_poll_interval,
        boot_timeout=args.boot_timeout,
        default_model=MODEL_ID,
        api_key=args.api_key,
    )


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


def client_ip(request: web.Request) -> str:
    """Real client IP — CF-Connecting-IP when behind the Cloudflare tunnel,
    otherwise the peer address (which is just cloudflared's loopback)."""
    return request.headers.get("CF-Connecting-IP") or request.remote or "-"


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


async def proxy_request(request: web.Request) -> web.StreamResponse:
    manager: ModelManager = request.app["manager"]
    session: ClientSession = request.app["session"]

    req_id = secrets.token_hex(4)
    started = time.monotonic()
    manager.begin_request()

    try:
        body = await request.read() if request.can_read_body else None
        path = request.path.rstrip("/")
        if path in LOAD_TRIGGER_PATHS:
            await manager.ensure_loaded(manager.config.default_model)
    except Exception as exc:
        logging.exception("[req=%s] backend unavailable", req_id)
        manager.end_request()
        return web.json_response(
            {"error": "backend unavailable", "detail": str(exc)},
            status=503,
            headers={"Retry-After": str(RETRY_AFTER_SECONDS), "X-Request-ID": req_id},
        )

    target_url = f"{manager.config.backend_base_url}{request.rel_url}"
    headers = filter_request_headers(request.headers, manager.config.api_key)

    try:
        upstream_resp = await session.request(
            request.method, target_url, headers=headers,
            data=body, allow_redirects=False, timeout=None,
        )
        response_headers = filter_response_headers(upstream_resp.headers)
        response_headers["X-Request-ID"] = req_id
        downstream = web.StreamResponse(
            status=upstream_resp.status, reason=upstream_resp.reason, headers=response_headers,
        )
        await downstream.prepare(request)
        try:
            async for chunk in upstream_resp.content.iter_any():
                await downstream.write(chunk)
        except ConnectionResetError:
            logging.info("[req=%s] client disconnected", req_id)
        finally:
            with contextlib.suppress(ConnectionResetError, RuntimeError):
                await downstream.write_eof()
            duration_ms = int((time.monotonic() - started) * 1000)
            logging.info(
                "[req=%s] %s %s %s -> %s in %dms",
                req_id, client_ip(request), request.method, request.rel_url,
                downstream.status, duration_ms,
            )
            return downstream
    except ClientError as exc:
        logging.exception("[req=%s] proxy failure", req_id)
        return web.json_response(
            {"error": "bad gateway", "detail": str(exc)},
            status=502, headers={"X-Request-ID": req_id},
        )
    finally:
        manager.end_request()


async def health_handler(request: web.Request) -> web.Response:
    manager: ModelManager = request.app["manager"]
    return web.json_response({
        "status": "ok",
        "router": "running" if manager.server_running else "down",
        "loaded": manager.model_loaded,
        "active_requests": manager.active_requests,
    })


async def lifecycle_context(app: web.Application):
    session = ClientSession(timeout=ClientTimeout(total=None))
    manager = ModelManager(app["config"], session)

    app["session"] = session
    app["manager"] = manager

    await manager.start_server()
    watchdog_task = asyncio.create_task(idle_watchdog(manager))

    try:
        yield
    finally:
        watchdog_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watchdog_task
        await manager.stop_server()
        with contextlib.suppress(Exception):
            await session.close()


async def idle_watchdog(manager: ModelManager) -> None:
    while True:
        await asyncio.sleep(manager.config.idle_check_interval)
        try:
            await manager.unload_if_idle()
        except Exception:
            logging.exception("idle watchdog error")


def build_app(config: ProxyConfig) -> web.Application:
    app = web.Application(client_max_size=32 * 1024 * 1024, middlewares=[auth_middleware])
    app["config"] = config
    app.cleanup_ctx.append(lifecycle_context)
    app.router.add_route("GET", "/health", health_handler)
    app.router.add_route("*", "/{tail:.*}", proxy_request)
    return app


def main() -> int:
    configure_logging()
    config = build_config()
    logging.info(
        "Embed proxy %s:%s -> router %s:%s | default=%s | idle=%ss",
        config.proxy_host, config.proxy_port,
        config.server_host, config.server_port,
        config.default_model, config.idle_timeout,
    )
    try:
        web.run_app(
            build_app(config),
            host=config.proxy_host, port=config.proxy_port,
            reuse_address=True,
            access_log_format='%a req=%{X-Request-ID}o "%r" %s %b %Dus',
        )
    except OSError as e:
        if "address already in use" in str(e).lower():
            logging.error("Port %s already in use", config.proxy_port)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
