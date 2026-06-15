"""Slim entry point — app lifecycle and route registration for the chat proxy."""
from __future__ import annotations

import asyncio
import contextlib
import logging

from aiohttp import ClientSession, ClientTimeout, web

from proxy_base import (
    ForwardedAccessLogger, ProxyConfig,
    auth_middleware, configure_logging, health_handler, idle_watchdog,
)
from router_manager import ChatRouterManager
from chat_logger import ChatLogger
from proxy_request_handlers import (
    embed_forward, models_handler, props_handler, proxy_request,
)
from proxy_config import ROOT, build_config


async def lifecycle_context(app: web.Application):
    session = ClientSession(timeout=ClientTimeout(total=None))
    manager = ChatRouterManager(app["config"], session)
    chat_logger = ChatLogger(ROOT / "logs") if app["config"].chat_log else None

    app["session"] = session
    app["manager"] = manager
    app["chat_logger"] = chat_logger

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
        if chat_logger is not None:
            chat_logger.close()


def build_app(config: ProxyConfig) -> web.Application:
    app = web.Application(client_max_size=128 * 1024 * 1024, middlewares=[auth_middleware])
    app["config"] = config
    app.cleanup_ctx.append(lifecycle_context)
    app.router.add_route("GET", "/health", health_handler)
    app.router.add_route("GET", "/models", models_handler)
    app.router.add_route("GET", "/chat/models", models_handler)
    app.router.add_route("GET", "/props", props_handler)
    app.router.add_route("GET", "/chat/props", props_handler)
    # /embedding and /embedding/* are reverse-proxied to embed_proxy on
    # :8003. Must come before the chat catch-all so embed traffic never hits
    # the chat router.
    app.router.add_route("*", "/embedding", embed_forward)
    app.router.add_route("*", "/embedding/{tail:.*}", embed_forward)
    # /chat/* is the public alias for chat completions; bare /v1/... at the
    # root still works for backwards compat. Both go through proxy_request,
    # which strips the /chat prefix before forwarding to the chat router.
    app.router.add_route("*", "/chat", proxy_request)
    app.router.add_route("*", "/chat/{tail:.*}", proxy_request)
    app.router.add_route("*", "/{tail:.*}", proxy_request)
    return app


def main() -> int:
    configure_logging()
    config = build_config()
    logging.info(
        "Proxy %s:%s -> router %s:%s | default=%s | idle=%ss",
        config.proxy_host, config.proxy_port,
        config.server_host, config.server_port,
        config.default_model, config.idle_timeout,
    )
    try:
        web.run_app(
            build_app(config),
            host=config.proxy_host, port=config.proxy_port,
            reuse_address=True,
            access_log_class=ForwardedAccessLogger,
            access_log_format='%a req=%{X-Request-ID}o "%r" %s %b %Dus',
        )
    except OSError as e:
        if "address already in use" in str(e).lower():
            logging.error("Port %s already in use", config.proxy_port)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
