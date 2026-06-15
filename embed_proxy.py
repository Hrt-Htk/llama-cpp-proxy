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
from pathlib import Path

from aiohttp import ClientError, ClientSession, ClientTimeout, web

from log_paths import (
    DATE_FMT,
    current_week_dir,
    local_now,
)

from proxy_base import (
    API_KEY, ProxyConfig, auth_middleware, client_ip, configure_logging,
    filter_request_headers, filter_response_headers, health_handler, idle_watchdog,
)
from router_manager import RouterManager

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
PROXY_PORT = 8003          # cloudflared origin (embed.example.com)
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8004         # internal router
IDLE_TIMEOUT = 600
IDLE_CHECK_INTERVAL = 30
HEALTH_POLL_INTERVAL = 1.0
BOOT_TIMEOUT = 60
RETRY_AFTER_SECONDS = 30

LOAD_TRIGGER_PATHS = {"/v1/embeddings", "/embeddings", "/v1/rerank", "/rerank"}


class EmbedProxyConfig(ProxyConfig):
    @property
    def server_command(self) -> list[str]:
        log_file = current_week_dir(ROOT / "logs") / f"embed-server-{local_now().strftime(DATE_FMT)}.log"
        return [
            str(SERVER_EXE),
            "--log-file", str(log_file),
            "--log-timestamps",
            "--log-prefix",
            "--models-preset", str(PRESET_PATH),
            "--models-max", "1",
            "--no-models-autoload",
            "--host", self.server_host,
            "--port", str(self.server_port),
            "--api-key", self.api_key,
        ]


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

    return EmbedProxyConfig(
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


async def proxy_request(request: web.Request) -> web.StreamResponse:
    manager: RouterManager = request.app["manager"]
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
            # Close upstream to give the router a clean TCP FIN.
            # Without this, httplib in the router destroys the response object
            # mid-stream and cancels the generation task, crashing the worker child.
            # (See llama.cpp commit 635b70d and PR #23226 for the same pattern.)
            try:
                upstream_resp.close()
            except Exception:
                pass
        finally:
            with contextlib.suppress(ConnectionResetError, RuntimeError):
                await downstream.write_eof()
            # Always close the upstream response to free the router's connection.
            # Prevents zombie connections and worker child crashes on disconnect.
            # close() is idempotent — safe if already closed in the except block.
            try:
                upstream_resp.close()
            except Exception:
                pass
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


async def lifecycle_context(app: web.Application):
    session = ClientSession(timeout=ClientTimeout(total=None))
    manager = RouterManager(app["config"], session)

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


def build_app(config: ProxyConfig) -> web.Application:
    app = web.Application(client_max_size=32 * 1024 * 1024, middlewares=[auth_middleware])
    app["config"] = config
    app.cleanup_ctx.append(lifecycle_context)
    app.router.add_route("GET", "/health", health_handler)
    app.router.add_route("*", "/{tail:.*}", proxy_request)
    return app


def main() -> int:
    configure_logging("embed-proxy")
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
