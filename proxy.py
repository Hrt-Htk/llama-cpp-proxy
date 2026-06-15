from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import secrets
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
    API_KEY, ClientGone, DeadWorkerError, ForwardedAccessLogger, ProxyConfig,
    auth_middleware, client_ip, configure_logging, filter_request_headers,
    filter_response_headers, health_handler, idle_watchdog,
    _is_dead_worker_response,
)
from router_manager import ChatRouterManager
from chat_logger import ChatLogger, SSEChunkLogger


ROOT = Path(__file__).resolve().parent
SERVER_EXE = ROOT / "llama.cpp_latest" / "llama-server.exe"
PRESET_PATH = ROOT / "models-preset.ini"

# Each (model, ctx) pair becomes its own preset section so the router
# exposes them as distinct models that the pi-llama-cpp extension can
# discover and switch between — pi treats ctx changes as model switches,
# which forces a router reload with the new ctx-size baked in.
@dataclass(frozen=True)
class ModelChoice:
    label: str          # menu display
    base_id: str        # e.g. "qwen3.6-35b-q3" — ctx suffix appended at render time
    model_file: Path    # GGUF weights
    mmproj_file: Path   # multimodal projector
    spec_mtp: bool = False  # enable built-in MTP speculative decoding (draft-mtp)

    def preset_id(self, ctx: int) -> str:
        return f"{self.base_id}-{ctx // 1024}k"


MODELS: list[ModelChoice] = [
    ModelChoice(
        "Qwen3.6-35B-A3B Q3",
        "qwen3.6-35b-q3",
        ROOT / "models" / "Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf",
        ROOT / "models" / "_aux" / "mmproj-F16.gguf",
    ),
    ModelChoice(
        "Qwen3.6-35B-A3B Q4",
        "qwen3.6-35b-q4",
        ROOT / "models" / "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        ROOT / "models" / "_aux" / "mmproj-F16.gguf",
    ),
    ModelChoice(
        "Qwen3.6-27B Q4",
        "qwen3.6-27b-q4",
        ROOT / "models" / "Qwen3.6-27B-UD-Q4_K_XL.gguf",
        ROOT / "models" / "_aux" / "mmproj-27b-BF16.gguf",
    ),
    ModelChoice(
        "Qwen3.6-27B Q4 MTP",
        "qwen3.6-27b-q4-mtp",
        ROOT / "models" / "Qwen3.6-27B-UD-Q4_K_XL.mtp.gguf",
        ROOT / "models" / "_aux" / "mmproj-27b-BF16.gguf",
        spec_mtp=True,
    ),
]

CTX_CHOICES: list[int] = [32768, 65536, 98304, 131072]

PROXY_HOST = "0.0.0.0"
PROXY_PORT = 8001
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8002
EMBED_PROXY_HOST = "::1"
EMBED_PROXY_PORT = 8003
IDLE_TIMEOUT = 600  # 10 minutes of inactivity before unload
IDLE_CHECK_INTERVAL = 30  # check every 30s
HEALTH_POLL_INTERVAL = 1.0
BOOT_TIMEOUT = 60
RETRY_AFTER_SECONDS = 30


class ChatProxyConfig(ProxyConfig):
    @property
    def server_command(self) -> list[str]:
        log_file = current_week_dir(ROOT / "logs") / f"llama-server-{local_now().strftime(DATE_FMT)}.log"
        return [
            str(SERVER_EXE),
            "--log-file", str(log_file),
            "--log-timestamps",
            "--log-prefix",
            "--models-preset", str(PRESET_PATH),
            "--models-max", "1",
            "--no-models-autoload",
            # --- perf A/B test: chat-template flags (see chat_template_perf_test.md) ---
            # Variant D: full new config (jinja + custom template + preserve_thinking kwarg)
            "--jinja",
            "--chat-template-file", str(ROOT / "chat_template.jinja"),
            "--chat-template-kwargs", '{"preserve_thinking":true}',
            # ------------------------------------------------------------------------
            "--host", self.server_host,
            "--port", str(self.server_port),
            "--api-key", self.api_key,
        ]


DEFAULT_MODEL: ModelChoice = MODELS[0]
DEFAULT_CTX: int = CTX_CHOICES[-1]


def pick_setup(model_arg: str | None, ctx_arg: int | None) -> tuple[ModelChoice, int]:
    """Resolve (model, context) fallback defaults from CLI args.

    All (model × ctx) combos are exposed as router presets regardless; this
    only picks which one to use when a client doesn't specify a model.
    """
    model = DEFAULT_MODEL
    if model_arg:
        match = next(
            (m for m in MODELS if m.label == model_arg or m.model_file.stem == model_arg),
            None,
        )
        if match is None:
            labels = ", ".join(m.label for m in MODELS)
            raise SystemExit(f"Unknown model {model_arg!r}; available: {labels}")
        model = match

    ctx = ctx_arg if ctx_arg is not None else DEFAULT_CTX
    if ctx_arg is not None and ctx_arg not in CTX_CHOICES:
        logging.warning("--ctx-size %d is outside preset choices %s", ctx_arg, CTX_CHOICES)

    for path in (model.model_file, model.mmproj_file):
        if not path.exists():
            raise SystemExit(f"Missing file for {model.label}: {path}")
    return model, ctx


def _model_preset_section(model: ModelChoice, ctx: int) -> str:
    """Return the INI section text for a single (model, ctx) pair."""
    spec = (
        f"spec-type        = draft-mtp\n"
        f"spec-draft-n-max = 2\n"
        f"spec-draft-p-min = 0.0\n"
        if model.spec_mtp
        else ""
    )
    return (
        f"[{model.preset_id(ctx)}]\n"
        f"model         = {model.model_file.as_posix()}\n"
        f"mmproj        = {model.mmproj_file.as_posix()}\n"
        f"ctx-size      = {ctx}\n"
        f"n-gpu-layers  = 999\n"
        f"flash-attn    = on\n"
        f"cache-type-k  = q4_0\n"
        f"cache-type-v  = q4_0\n"
        f"no-mmap       = 1\n"
        f"parallel      = 1\n"
        f"jinja         = 1\n"
        f"temp          = 0.6\n"
        f"top-p         = 0.95\n"
        f"top-k         = 20\n"
        + spec
    )


def write_preset(models: list[ModelChoice], ctx_choices: list[int]) -> None:
    """Generate models-preset.ini with every (model, ctx) combination.

    Each combo becomes a distinct preset id (e.g. `qwen3.6-35b-q3-128k`),
    so picking a different ctx in pi triggers a router reload with the new
    context size — the only way to "change ctx" without restarting the proxy.
    """
    sections = [
        _model_preset_section(m, ctx)
        for m in models
        for ctx in ctx_choices
    ]
    content = "\n".join(sections) + "\n"
    PRESET_PATH.write_text(content, encoding="utf-8")


def build_config() -> ProxyConfig:
    p = argparse.ArgumentParser(description="Router-mode proxy for llama-server")
    p.add_argument("--proxy-host", default=PROXY_HOST)
    p.add_argument("--proxy-port", type=int, default=PROXY_PORT)
    p.add_argument("--server-host", default=SERVER_HOST)
    p.add_argument("--server-port", type=int, default=SERVER_PORT)
    p.add_argument("--idle-timeout", type=int, default=IDLE_TIMEOUT)
    p.add_argument("--idle-check-interval", type=int, default=IDLE_CHECK_INTERVAL)
    p.add_argument("--health-poll-interval", type=float, default=HEALTH_POLL_INTERVAL)
    p.add_argument("--boot-timeout", type=int, default=BOOT_TIMEOUT)
    p.add_argument("--model", default=None,
                   help="Skip the model picker (use exact label from MODELS)")
    p.add_argument("--ctx-size", type=int, default=None,
                   help="Skip the context picker (any int; menu offers 32k/64k/128k)")
    p.add_argument("--api-key", default=API_KEY)
    p.add_argument("--embed-host", default=EMBED_PROXY_HOST)
    p.add_argument("--embed-port", type=int, default=EMBED_PROXY_PORT)
    p.add_argument("--no-chat-log", action="store_true")
    args = p.parse_args()

    if not SERVER_EXE.exists():
        raise SystemExit(f"Missing required file: {SERVER_EXE}")

    model, ctx = pick_setup(args.model, args.ctx_size)
    write_preset(MODELS, CTX_CHOICES)
    default_id = model.preset_id(ctx)
    print(f"Default: {model.label} @ {ctx // 1024}k ctx (id: {default_id})")
    print(f"Exposed presets: {len(MODELS) * len(CTX_CHOICES)} (one per model×ctx combo)")

    return ChatProxyConfig(
        proxy_host=args.proxy_host,
        proxy_port=args.proxy_port,
        server_host=args.server_host,
        server_port=args.server_port,
        idle_timeout=args.idle_timeout,
        idle_check_interval=args.idle_check_interval,
        health_poll_interval=args.health_poll_interval,
        boot_timeout=args.boot_timeout,
        default_model=default_id,
        api_key=args.api_key,
        embed_host=args.embed_host,
        embed_port=args.embed_port,
        chat_log=not args.no_chat_log,
    )


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
            upstream_resp = await session.request(
                request.method, target_url, headers=headers,
                data=body, allow_redirects=False, timeout=None,
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
                # Closing upstream is the correct way to signal cancellation, but it
                # does NOT prevent the worker crash on its own: llama-server has an
                # unfixed cancel→next-request desync (ggml-org/llama.cpp#20921) that
                # wedges/crashes the worker — worst with reasoning + speculative on a
                # large-context (cancel-during-prefill) request. We can't make the
                # cancel safe here, so we flag the worker suspect and let the NEXT
                # model request probe it first (see guard_after_cancel).
                try:
                    upstream_resp.close()
                except Exception:
                    pass
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
            # Post-cancel guard: if a prior request was aborted mid-generation,
            # probe (and recover if needed) the worker before this request hits it,
            # avoiding the upstream cancel→next-request crash. No-op when not suspect.
            # Runs outside use_model so its recover_worker() doesn't nest _load_lock.
            await manager.guard_after_cancel(model)
            # Wrap the entire forward inside use_model so _end_forward() only
            # fires after the stream is fully drained — no eviction mid-stream.
            # Up to 2 attempts: on DeadWorkerError, recover then retry once.
            max_attempts = 2
            for attempt in range(1, max_attempts + 1):
                try:
                    async with manager.use_model(model):
                        return await _do_forward()
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
        else:
            # Non-model endpoints (health, /v1/models, props, …) forward directly.
            return await _do_forward()
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

    tail = request.match_info.get("tail", "")
    sub_path = "/" + tail if tail else "/"
    query = request.rel_url.query_string
    target_url = f"{config.embed_base_url}{sub_path}"
    if query:
        target_url = f"{target_url}?{query}"

    body = await request.read() if request.can_read_body else None
    headers = filter_request_headers(request.headers, config.api_key)

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
    except ClientError as exc:
        logging.exception("[embed req=%s] forward failure", req_id)
        return web.json_response(
            {"error": "bad gateway", "detail": str(exc)},
            status=502, headers={"X-Request-ID": req_id},
        )


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
