"""HTTP request handlers, streaming, and retry/recovery for the chat proxy."""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import secrets
import time

from aiohttp import ClientError, ClientSession, ClientTimeout, web

from proxy_base import (
    ClientGone, DeadWorkerError, ProxyConfig,
    client_ip, filter_request_headers, filter_response_headers,
    _is_dead_worker_response,
)
from router_manager import ChatRouterManager
from chat_logger import ChatLogger, SSEChunkLogger
from proxy_config import RETRY_AFTER_SECONDS

# Max seconds to wait for the router to send *response headers* before treating
# the worker as hung. The 2026-07-19 incident: the router accepted a completion,
# forwarded it to a dead/hung model-worker child, and never responded, while the
# proxy awaited with timeout=None and hung forever. A hung worker is
# indistinguishable from an infinitely-slow one, so — like nginx's
# proxy_read_timeout — we bound the wait. Generous enough that a cold model load
# (~15-25s) is never mistaken for a hang. Override via $PROXY_FIRST_BYTE_TIMEOUT.
FIRST_BYTE_TIMEOUT = float(os.environ.get("PROXY_FIRST_BYTE_TIMEOUT", "60"))


def _strip_chat_prefix(path: str) -> str:
    """Strip the /chat alias prefix so backend sees plain /v1/... paths."""
    if path == "/chat":
        return "/"
    if path.startswith("/chat/"):
        return path[len("/chat"):]
    return path


def _inject_cache_prompt(body: bytes | None, method: str, path: str) -> bytes | None:
    if method != "POST" or path.rstrip("/") != "/v1/chat/completions" or not body:
        return body
    try:
        payload = json.loads(body)
        if not payload.get("cache_prompt"):
            payload["cache_prompt"] = True
            return json.dumps(payload, separators=(",", ":")).encode()
    except (ValueError, TypeError):
        pass
    return body


def _model_from_body(body: bytes | None, fallback: str) -> str:
    if not body:
        return fallback
    try:
        payload = json.loads(body)
        m = payload.get("model")
        if isinstance(m, str) and m:
            return m
    except (ValueError, TypeError):
        pass
    return fallback


async def _retry_with_recovery(
    manager: ChatRouterManager, model: str, req_id: str, started: float,
    forward_fn,
) -> web.StreamResponse:
    """Retry loop with DeadWorkerError recovery (max 2 attempts)."""
    await manager.guard_after_cancel(model)
    max_attempts = 2
    for attempt in range(1, max_attempts + 1):
        try:
            async with manager.use_model(model):
                return await forward_fn()
        except DeadWorkerError as exc:
            dead_detected_at = time.monotonic()
            logging.warning(
                "[req=%s] dead-worker 500 on attempt %d/%d — body: %r",
                req_id, attempt, max_attempts, exc.body[:200],
            )
            if attempt < max_attempts:
                recovered = await manager.recover_worker(model, dead_detected_at)
                if not recovered:
                    logging.error("[req=%s] worker recovery failed — giving up", req_id)
                    return web.json_response(
                        {"error": "backend unavailable", "detail": "worker recovery failed"},
                        status=503,
                        headers={"Retry-After": str(RETRY_AFTER_SECONDS), "X-Request-ID": req_id},
                    )
                logging.info("[req=%s] worker recovered — retrying request", req_id)
            else:
                logging.error("[req=%s] dead worker persists after recovery — 503", req_id)
                return web.json_response(
                    {"error": "backend unavailable", "detail": "dead worker after recovery"},
                    status=503,
                    headers={"Retry-After": str(RETRY_AFTER_SECONDS), "X-Request-ID": req_id},
                )
        except Exception as exc:
            logging.exception("[req=%s] backend unavailable", req_id)
            return web.json_response(
                {"error": "backend unavailable", "detail": str(exc)},
                status=503,
                headers={"Retry-After": str(RETRY_AFTER_SECONDS), "X-Request-ID": req_id},
            )


async def _stream_response(
    upstream_resp, request: web.Request, is_sse: bool, chat_logger,
    req_id: str, started: float, needs_model: bool, manager: ChatRouterManager,
) -> web.StreamResponse:
    """Prepare downstream, stream content (SSE/non-SSE), handle disconnect, log."""
    response_headers = filter_response_headers(upstream_resp.headers)
    response_headers["X-Request-ID"] = req_id
    downstream = web.StreamResponse(
        status=upstream_resp.status, reason=upstream_resp.reason, headers=response_headers,
    )
    try:
        await downstream.prepare(request)
    except ConnectionResetError:
        raise ClientGone from None
    try:
        if is_sse and chat_logger:
            wrapped = SSEChunkLogger(upstream_resp, chat_logger)
            while True:
                try:
                    chunk = await asyncio.wait_for(wrapped.readany(), timeout=25)
                except asyncio.TimeoutError:
                    await downstream.write(b": keep-alive\n\n")
                    continue
                if not chunk:
                    break
                await downstream.write(chunk)
        elif is_sse:
            while True:
                try:
                    chunk = await asyncio.wait_for(upstream_resp.content.readany(), timeout=25)
                except asyncio.TimeoutError:
                    await downstream.write(b": keep-alive\n\n")
                    continue
                if not chunk:
                    break
                await downstream.write(chunk)
        else:
            async for chunk in upstream_resp.content.iter_any():
                await downstream.write(chunk)
    except ConnectionResetError:
        logging.info("[req=%s] client disconnected", req_id)
        # Mid-stream disconnect on a model request: the router may still be
        # generating. Without guard_after_cancel the next request to this
        # worker can crash (ggml-org/llama.cpp#20921). Mark suspect so the
        # retry loop probes/recovers before the next model request.
        if needs_model:
            manager.mark_worker_suspect()
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


async def proxy_request(request: web.Request) -> web.StreamResponse:
    manager: ChatRouterManager = request.app["manager"]
    session: ClientSession = request.app["session"]
    chat_logger: ChatLogger | None = request.app.get("chat_logger")

    req_id = secrets.token_hex(4)
    started = time.monotonic()
    manager.begin_request()

    effective_path = _strip_chat_prefix(request.path)

    # Determine whether this request requires a model and parse the body up front.
    # Body parsing errors (OOM, malformed) become 503 so clients get Retry-After.
    try:
        body = await request.read() if request.can_read_body else None
        body = _inject_cache_prompt(body, request.method, effective_path)
    except Exception as exc:
        logging.exception("[req=%s] body read failed", req_id)
        manager.end_request()
        return web.json_response(
            {"error": "backend unavailable", "detail": str(exc)},
            status=503,
            headers={"Retry-After": str(RETRY_AFTER_SECONDS), "X-Request-ID": req_id},
        )

    path = effective_path.rstrip("/")
    needs_model = path in (
        "/v1/chat/completions", "/v1/completions", "/v1/embeddings",
        "/chat/completions", "/completions", "/embeddings",
    )
    model: str | None = None
    if needs_model:
        model = _model_from_body(body, manager.config.default_model)

    if chat_logger is not None:
        await chat_logger.log_request(request.method, request.path, body, req_id)

    query = request.rel_url.query_string
    target_url = f"{manager.config.backend_base_url}{effective_path}"
    if query:
        target_url = f"{target_url}?{query}"
    headers = filter_request_headers(request.headers, manager.config.api_key)

    async def _do_forward() -> web.StreamResponse:
        """Inner forward/stream; called from within use_model or directly.

        For non-SSE responses with status >= 500 the body is pre-read so we
        can detect a dead-worker error BEFORE committing the response to the
        client via ``downstream.prepare()``. If a dead-worker marker is found
        a ``DeadWorkerError`` is raised for the caller to handle. All other
        paths (success, 4xx, SSE) stream zero-copy as before.
        """
        try:
            # Bound the wait for *response headers*. A hung worker makes the
            # router accept the request and never respond; without this bound
            # the await hangs forever (timeout=None) and the dead-worker
            # recovery below is never reached. On timeout we synthesize a
            # DeadWorkerError so the retry loop cycles the worker, exactly as it
            # does for a router-reported dead-worker 5xx. Streaming after the
            # headers keeps its own per-chunk keep-alive timeout, so this only
            # guards time-to-first-response, not generation length.
            try:
                upstream_resp = await asyncio.wait_for(
                    session.request(
                        request.method, target_url, headers=headers,
                        data=body, allow_redirects=False, timeout=None,
                    ),
                    timeout=FIRST_BYTE_TIMEOUT,
                )
            except (asyncio.TimeoutError, TimeoutError):
                logging.warning(
                    "[req=%s] no response headers from router within %.0fs — "
                    "treating worker as hung", req_id, FIRST_BYTE_TIMEOUT,
                )
                raise DeadWorkerError(
                    504, b"router sent no response within first-byte timeout "
                         b"(worker hung)",
                )
            is_sse = "text/event-stream" in upstream_resp.headers.get("Content-Type", "")

            # Pre-read non-SSE error bodies BEFORE prepare() so we can inspect
            # them and raise DeadWorkerError without having committed headers.
            if upstream_resp.status >= 500 and not is_sse:
                error_body = await upstream_resp.content.read()
                if _is_dead_worker_response(upstream_resp.status, error_body):
                    raise DeadWorkerError(upstream_resp.status, error_body)
                # Real (non-dead-worker) 5xx — forward as-is.
                response_headers = filter_response_headers(upstream_resp.headers)
                response_headers["X-Request-ID"] = req_id
                downstream = web.StreamResponse(
                    status=upstream_resp.status, reason=upstream_resp.reason,
                    headers=response_headers,
                )
                try:
                    await downstream.prepare(request)
                except ConnectionResetError:
                    raise ClientGone from None
                try:
                    await downstream.write(error_body)
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

            return await _stream_response(
                upstream_resp, request, is_sse, chat_logger,
                req_id, started, needs_model, manager,
            )
        except DeadWorkerError:
            raise  # propagate to retry loop
        except ClientGone:
            # Downstream client (cloudflared / end client) hung up before we finished
            # sending the response — raised when downstream.prepare() hit a closing
            # transport. Same benign disconnect the write loops already handle, just
            # earlier. Not a proxy fault, so log calmly and return 499.
            logging.info("[req=%s] client disconnected before response sent", req_id)
            # Close the upstream response to release the router's connection. The
            # worker had already begun generating (it produced response headers), so
            # this is a mid-generation abort too — flag it for the post-cancel guard.
            try:
                upstream_resp.close()
            except Exception:
                pass
            if needs_model:
                manager.mark_worker_suspect()
            return web.Response(
                status=499, reason="Client Closed Request",
                headers={"X-Request-ID": req_id},
            )
        except ClientError as exc:
            # Genuine upstream failure (router connection reset/refused, etc.) — a
            # real bad gateway. Note: an *upstream* ClientConnectionResetError lands
            # here, while a *downstream* one was already converted to ClientGone above.
            logging.exception("[req=%s] proxy failure", req_id)
            return web.json_response(
                {"error": "bad gateway", "detail": str(exc)},
                status=502, headers={"X-Request-ID": req_id},
            )

    try:
        if needs_model:
            return await _retry_with_recovery(manager, model, req_id, started, _do_forward)
        else:
            # Non-model endpoints (health, /v1/models, props, …) forward directly.
            return await _do_forward()
    except DeadWorkerError as exc:
        # Only reachable on the non-model path (the retry loop handles it for
        # model requests). A hung worker starving a metadata request → 503.
        logging.warning("[req=%s] upstream unresponsive on non-model path: %r",
                        req_id, exc.body[:120])
        return web.json_response(
            {"error": "backend unavailable", "detail": "upstream unresponsive"},
            status=503,
            headers={"Retry-After": str(RETRY_AFTER_SECONDS), "X-Request-ID": req_id},
        )
    finally:
        manager.end_request()


async def models_handler(request: web.Request) -> web.Response:
    """Normalize the router's /models payload so clients see a sane status.

    llama.cpp's router keeps `status.failed = true` (with a stale `exit_code`)
    on presets that have never had a successful load in the current process —
    a residual diagnostic flag rather than a real "this model is broken"
    signal. Newer pi-llama-cpp versions short-circuit to FAILED when they
    see `failed: true`, so a freshly-booted router shows every preset as
    "Retry" in pi. We rewrite the flag to false whenever `value` says the
    preset is simply unloaded — `value` is the source of truth.
    """
    manager: ChatRouterManager = request.app["manager"]
    session: ClientSession = request.app["session"]
    effective_path = _strip_chat_prefix(request.path)
    query = request.rel_url.query_string
    target_url = f"{manager.config.backend_base_url}{effective_path}"
    if query:
        target_url = f"{target_url}?{query}"
    headers = filter_request_headers(request.headers, manager.config.api_key)
    async with session.get(target_url, headers=headers) as upstream:
        body_text = await upstream.text()
        try:
            payload = json.loads(body_text)
        except (ValueError, TypeError):
            return web.Response(status=upstream.status, body=body_text,
                                content_type=upstream.content_type or "application/json")
        for entry in payload.get("data") or []:
            status = entry.get("status")
            if isinstance(status, dict) and status.get("value") == "unloaded":
                status["failed"] = False
                status.pop("exit_code", None)
        return web.json_response(payload, status=upstream.status)


async def props_handler(request: web.Request) -> web.Response:
    """Normalize the router's /props response for unloaded models.

    llama.cpp returns HTTP 400 with `{"error":{"code":400,"message":"model is
    not loaded",...}}` when /props is asked about a non-loaded preset. The
    pi-llama-cpp extension probes /props as a sanity check after seeing a
    model in /models with status "unloaded", and a strict reading of its
    parser can mis-classify that 400 response as FAILED (shows "Retry"
    instead of "Load & switch"). We rewrite to a clean 200 JSON whose
    shape matches the exact equality checks in baseModel.getStatus().
    """
    manager: ChatRouterManager = request.app["manager"]
    session: ClientSession = request.app["session"]
    effective_path = _strip_chat_prefix(request.path)
    query = request.rel_url.query_string
    target_url = f"{manager.config.backend_base_url}{effective_path}"
    if query:
        target_url = f"{target_url}?{query}"
    headers = filter_request_headers(request.headers, manager.config.api_key)
    async with session.get(target_url, headers=headers) as upstream:
        body_text = await upstream.text()
        if upstream.status == 400 and "model is not loaded" in body_text:
            return web.json_response(
                {"error": {"code": 400, "message": "model is not loaded"}}
            )
        return web.Response(
            status=upstream.status,
            body=body_text,
            content_type=upstream.content_type or "application/json",
        )


def _build_embed_target(request: web.Request, config: ProxyConfig) -> str:
    """Build the target URL for an /embedding/* forward.

    Strips the /embedding prefix so embed_proxy sees plain OpenAI-style
    paths (/v1/embeddings, /v1/models, /health, …).
    """
    tail = request.match_info.get("tail", "")
    sub_path = "/" + tail if tail else "/"
    query = request.rel_url.query_string
    target_url = f"{config.embed_base_url}{sub_path}"
    if query:
        target_url = f"{target_url}?{query}"
    return target_url


async def _stream_embed_response(
    upstream_resp,
    request: web.Request,
    req_id: str,
    started: float,
) -> web.StreamResponse:
    """Create downstream, prepare, stream upstream content, and log.

    Handles both SSE (with keep-alive) and non-SSE responses.
    ConnectionResetError on the downstream write is caught and logged.
    """
    response_headers = filter_response_headers(upstream_resp.headers)
    response_headers["X-Request-ID"] = req_id
    downstream = web.StreamResponse(
        status=upstream_resp.status, reason=upstream_resp.reason, headers=response_headers,
    )
    await downstream.prepare(request)
    try:
        is_sse = "text/event-stream" in upstream_resp.headers.get("Content-Type", "")
        if is_sse:
            while True:
                try:
                    chunk = await asyncio.wait_for(upstream_resp.content.readany(), timeout=25)
                except asyncio.TimeoutError:
                    await downstream.write(b": keep-alive\n\n")
                    continue
                if not chunk:
                    break
                await downstream.write(chunk)
        else:
            async for chunk in upstream_resp.content.iter_any():
                await downstream.write(chunk)
    except ConnectionResetError:
        logging.info("[embed req=%s] client disconnected", req_id)
    finally:
        with contextlib.suppress(ConnectionResetError, RuntimeError):
            await downstream.write_eof()
        duration_ms = int((time.monotonic() - started) * 1000)
        logging.info(
            "[embed req=%s] %s %s %s -> %s in %dms",
            req_id, client_ip(request), request.method, request.path,
            downstream.status, duration_ms,
        )
        return downstream


async def embed_forward(request: web.Request) -> web.StreamResponse:
    """Reverse-proxy /embedding/* to the standalone embed_proxy on :8003.

    Strips the prefix so embed_proxy sees plain OpenAI-style paths
    (/v1/embeddings, /v1/models, /health, ...). embed_proxy owns its own
    router, load/unload, and idle timer — this is a dumb HTTP forwarder.
    """
    session: ClientSession = request.app["session"]
    config: ProxyConfig = request.app["config"]
    req_id = secrets.token_hex(4)
    started = time.monotonic()

    target_url = _build_embed_target(request, config)
    body = await request.read() if request.can_read_body else None
    headers = filter_request_headers(request.headers, config.api_key)

    try:
        upstream_resp = await session.request(
            request.method, target_url, headers=headers,
            data=body, allow_redirects=False, timeout=None,
        )
        return await _stream_embed_response(
            upstream_resp, request, req_id, started,
        )
    except ClientError as exc:
        logging.exception("[embed req=%s] forward failure", req_id)
        return web.json_response(
            {"error": "bad gateway", "detail": str(exc)},
            status=502, headers={"X-Request-ID": req_id},
        )
