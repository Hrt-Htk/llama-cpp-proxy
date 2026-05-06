from __future__ import annotations

import argparse
import asyncio
import contextlib
from datetime import datetime, timezone
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

from aiohttp import ClientError, ClientSession, ClientTimeout, web


ROOT = Path(__file__).resolve().parent
SERVER_EXE = ROOT / "llama.cpp_setup" / "llama-server.exe"
MMPROJ_35B_PATH = ROOT / "models" / "mmproj-F16.gguf"
MMPROJ_27B_PATH = ROOT / "models" / "mmproj-27b-BF16.gguf"
MODEL_Q3_PATH = ROOT / "models" / "Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf"
MODEL_Q4_PATH = ROOT / "models" / "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
MODEL_27B_PATH = ROOT / "models" / "Qwen3.6-27B-UD-Q4_K_XL.gguf"

PROXY_HOST = "::"
PROXY_PORT = 8001
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8002
IDLE_TIMEOUT = 600
IDLE_CHECK_INTERVAL = 60
HEALTH_POLL_INTERVAL = 1.0
BOOT_TIMEOUT = 120
RETRY_AFTER_SECONDS = 30
API_KEY = os.environ.get("LLAMA_API_KEY", "rRZsSjRvaUuRMr5AeDA14rO9jaSlhSRhRtBI5ZlO")

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
    model_q4: bool
    model_27b: bool
    model_long: bool
    api_key: str

    @property
    def backend_base_url(self) -> str:
        return f"http://{self.server_host}:{self.server_port}"

    @property
    def model_path(self) -> Path:
        if self.model_27b:
            return MODEL_27B_PATH
        return MODEL_Q4_PATH if self.model_q4 else MODEL_Q3_PATH

    @property
    def mmproj_path(self) -> Path:
        if self.model_27b:
            return MMPROJ_27B_PATH
        return MMPROJ_35B_PATH

    @property
    def alias(self) -> str:
        if self.model_27b:
            return "qwen3.6-27b"
        return "qwen3.6-35b-a3b-q4" if self.model_q4 else "qwen3.6-35b-a3b"

    @property
    def context_size(self) -> int:
        return 262144 if self.model_long else 131072

    @property
    def server_command(self) -> list[str]:
        return [
            str(SERVER_EXE),
            "--model",
            str(self.model_path),
            "--mmproj",
            str(self.mmproj_path),
            "--alias",
            self.alias,
            "-ngl",
            "999",
            "--no-mmap",
            "-fa",
            "on",
            "--cache-type-k",
            "q4_0",
            "--cache-type-v",
            "q4_0",
            "-b",
            "8192",
            "-ub",
            "2048",
            "-c",
            str(self.context_size),
            "--parallel",
            "1",
            "--jinja",
            "--temp",
            "0.6",
            "--top-p",
            "0.95",
            "--top-k",
            "20",
            "--min-p",
            "0.0",
            "--presence-penalty",
            "0.0",
            "--port",
            str(self.server_port),
            "--host",
            self.server_host,
            "--api-key",
            self.api_key,
        ]


class ServerManager:
    def __init__(self, config: ProxyConfig, session: ClientSession) -> None:
        self.config = config
        self.session = session
        self.process: asyncio.subprocess.Process | None = None
        self._process_watch_task: asyncio.Task[None] | None = None
        self._boot_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self._active_requests = 0
        self._last_activity = time.monotonic()
        self._adopted = False  # True if we adopted an existing server (no process handle)

    @property
    def state(self) -> str:
        if self._boot_task is not None and not self._boot_task.done():
            return "booting"
        if self._adopted:
            return "running"
        if self.process is not None and self.process.returncode is None:
            return "running"
        return "offline"

    @property
    def active_requests(self) -> int:
        return self._active_requests

    def begin_request(self) -> None:
        self._active_requests += 1
        self._last_activity = time.monotonic()

    def end_request(self) -> None:
        self._active_requests = max(0, self._active_requests - 1)
        self._last_activity = time.monotonic()

    async def ensure_running(self) -> None:
        boot_task: asyncio.Task[None] | None = None

        async with self._lock:
            if self._adopted:
                # Already adopted an existing server, nothing to do
                return
            if self.process is not None and self.process.returncode is None:
                boot_task = self._boot_task
                if boot_task is None or boot_task.done():
                    return
            else:
                if self._boot_task is None or self._boot_task.done():
                    self._boot_task = asyncio.create_task(self._start_process())
                boot_task = self._boot_task

        if boot_task is not None:
            await boot_task

    async def stop(self, reason: str) -> None:
        async with self._lock:
            process = self.process
            adopted = self._adopted

            if process is None and not adopted:
                return

            if process is not None and process.returncode is not None:
                self.process = None
                self._adopted = False
                return

            logging.info("Stopping llama-server (%s)", reason)

            if adopted:
                # We adopted an existing server (from a previous crashed proxy instance)
                # but have no process handle. We can't kill it cleanly.
                # Just mark it as gone — it will keep running until manually stopped
                # or the next proxy instance adopts it again.
                logging.warning(
                    "Cannot stop adopted llama-server (no process handle). "
                    "It will keep running on port %s until manually killed.",
                    self.config.server_port,
                )
                self._adopted = False
            else:
                try:
                    # On Windows, llama-server may not handle terminate() gracefully.
                    # Try terminate first, then kill if needed.
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=5)
                    except asyncio.TimeoutError:
                        process.kill()
                        await asyncio.wait_for(process.wait(), timeout=5)
                except ProcessLookupError:
                    pass  # Already exited
                finally:
                    if self.process is process:
                        self.process = None

    async def stop_if_idle(self) -> None:
        idle_for = time.monotonic() - self._last_activity
        should_stop = (
            self.state == "running"
            and self._active_requests == 0
            and idle_for >= self.config.idle_timeout
        )

        if should_stop:
            await self.stop(f"idle for {int(idle_for)} seconds")

    async def _check_existing_server(self) -> bool:
        """Check if llama-server is already running on the target port (e.g. from a crashed proxy).
        Returns True if a healthy server is found."""
        health_url = f"{self.config.backend_base_url}/health"
        try:
            async with self.session.get(
                health_url,
                timeout=ClientTimeout(total=5),
            ) as response:
                if response.status == 200:
                    self._adopted = True
                    logging.info(
                        "Found existing llama-server on %s:%s — adopting it",
                        self.config.server_host,
                        self.config.server_port,
                    )
                    return True
        except (ClientError, asyncio.TimeoutError):
            pass
        return False

    async def _start_process(self) -> None:
        # Check if llama-server is already running (leftover from a crashed proxy instance)
        if await self._check_existing_server():
            async with self._lock:
                self._last_activity = time.monotonic()
            return

        logging.info(
            "Starting llama-server on %s:%s with model %s",
            self.config.server_host,
            self.config.server_port,
            self.config.model_path.name,
        )

        process = await asyncio.create_subprocess_exec(
            *self.config.server_command,
            cwd=str(ROOT),
        )

        async with self._lock:
            self.process = process
            self._process_watch_task = asyncio.create_task(self._watch_process(process))
            self._last_activity = time.monotonic()

        try:
            await self._wait_until_healthy(process)
            logging.info("llama-server is ready")
        except Exception:
            logging.exception("llama-server failed to start")
            await self.stop("boot failure")
            raise
        finally:
            async with self._lock:
                if self._boot_task is asyncio.current_task():
                    self._boot_task = None

    async def _wait_until_healthy(self, process: asyncio.subprocess.Process) -> None:
        deadline = time.monotonic() + self.config.boot_timeout
        health_url = f"{self.config.backend_base_url}/health"

        while time.monotonic() < deadline:
            if process.returncode is not None:
                raise RuntimeError(
                    f"llama-server exited during boot with code {process.returncode}"
                )

            try:
                async with self.session.get(
                    health_url,
                    timeout=ClientTimeout(total=5),
                ) as response:
                    if response.status == 200:
                        return
            except (ClientError, asyncio.TimeoutError):
                pass

            await asyncio.sleep(self.config.health_poll_interval)

        raise TimeoutError(
            f"llama-server did not become healthy within {self.config.boot_timeout} seconds"
        )

    async def _watch_process(self, process: asyncio.subprocess.Process) -> None:
        exit_code = await process.wait()
        level = logging.INFO if exit_code == 0 else logging.WARNING
        logging.log(level, "llama-server exited with code %s", exit_code)

        async with self._lock:
            if self.process is process:
                self.process = None


def configure_logging() -> None:
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / datetime.now(timezone.utc).strftime("%Y-%m-%d.log")

    # Console handler — live output
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))

    # File handler — daily rotation
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))

    logging.basicConfig(
        level=logging.INFO,
        handlers=[console, file_handler],
    )
    logging.info("Log file: %s", log_file)


def build_config() -> ProxyConfig:
    parser = argparse.ArgumentParser(description="Wake-on-demand proxy for llama-server")
    parser.add_argument("--q4", action="store_true", help="Use the Q4 model (35B-A3B)")
    parser.add_argument("--27b", dest="model_27b", action="store_true", help="Use the Qwen3.6-27B model")
    parser.add_argument("--long", action="store_true", help="Use the 256k context window")
    parser.add_argument("--proxy-host", default=PROXY_HOST)
    parser.add_argument("--proxy-port", type=int, default=PROXY_PORT)
    parser.add_argument("--server-host", default=SERVER_HOST)
    parser.add_argument("--server-port", type=int, default=SERVER_PORT)
    parser.add_argument("--idle-timeout", type=int, default=IDLE_TIMEOUT)
    parser.add_argument("--idle-check-interval", type=int, default=IDLE_CHECK_INTERVAL)
    parser.add_argument("--health-poll-interval", type=float, default=HEALTH_POLL_INTERVAL)
    parser.add_argument("--boot-timeout", type=int, default=BOOT_TIMEOUT)
    parser.add_argument("--api-key", default=API_KEY)
    args = parser.parse_args()

    if args.q4 and args.model_27b:
        raise SystemExit("Cannot use --q4 and --27b together. Pick one.")

    validate_paths(args.q4, args.model_27b)

    return ProxyConfig(
        proxy_host=args.proxy_host,
        proxy_port=args.proxy_port,
        server_host=args.server_host,
        server_port=args.server_port,
        idle_timeout=args.idle_timeout,
        idle_check_interval=args.idle_check_interval,
        health_poll_interval=args.health_poll_interval,
        boot_timeout=args.boot_timeout,
        model_q4=args.q4,
        model_27b=args.model_27b,
        model_long=args.long,
        api_key=args.api_key,
    )


def validate_paths(use_q4: bool, use_27b: bool) -> None:
    required_paths = [SERVER_EXE]
    if use_27b:
        required_paths.extend([MMPROJ_27B_PATH, MODEL_27B_PATH])
    else:
        required_paths.extend([MMPROJ_35B_PATH, MODEL_Q4_PATH if use_q4 else MODEL_Q3_PATH])

    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise SystemExit(f"Missing required files: {', '.join(missing)}")


def filter_request_headers(headers: web.BaseRequest.headers.__class__, api_key: str) -> dict[str, str]:
    forwarded: dict[str, str] = {}
    for name, value in headers.items():
        lowered = name.lower()
        if lowered == "host" or lowered == "content-length" or lowered in HOP_BY_HOP_HEADERS:
            continue
        forwarded[name] = value

    if api_key and "Authorization" not in forwarded:
        forwarded["Authorization"] = f"Bearer {api_key}"

    return forwarded


def filter_response_headers(headers: web.BaseRequest.headers.__class__) -> dict[str, str]:
    forwarded: dict[str, str] = {}
    for name, value in headers.items():
        lowered = name.lower()
        if lowered == "content-length" or lowered in HOP_BY_HOP_HEADERS:
            continue
        forwarded[name] = value
    return forwarded


def synthetic_status_value(manager: ServerManager) -> str:
    if manager.state == "running":
        return "loaded"
    if manager.state == "booting":
        return "loading"
    return "sleeping"


def build_model_details(config: ProxyConfig) -> dict[str, object]:
    model_stat = config.model_path.stat()
    modified_at = datetime.fromtimestamp(model_stat.st_mtime, tz=timezone.utc).isoformat()

    if config.model_27b:
        quantization = "Q4_K_XL"
        parent_model = "Qwen3.6-27B"
        parameter_size = "27B"
    else:
        quantization = "Q4_K_M" if config.model_q4 else "Q3_K_XL"
        parent_model = "Qwen3.6-35B-A3B"
        parameter_size = "35B"

    return {
        "name": config.alias,
        "model": config.model_path.name,
        "modified_at": modified_at,
        "size": str(model_stat.st_size),
        "digest": "",
        "type": "model",
        "description": f"Wake-on-demand proxy for {config.alias}",
        "tags": [],
        "capabilities": ["multimodal"],
        "parameters": str(config.context_size),
        "details": {
            "parent_model": parent_model,
            "format": "gguf",
            "family": "qwen",
            "families": ["qwen"],
            "parameter_size": parameter_size,
            "quantization_level": quantization,
        },
    }


def build_models_payload(manager: ServerManager) -> dict[str, object]:
    config = manager.config
    model_stat = config.model_path.stat()

    return {
        "object": "list",
        "models": [build_model_details(config)],
        "data": [
            {
                "id": config.alias,
                "aliases": [config.alias],
                "tags": [],
                "object": "model",
                "owned_by": "llama.cpp",
                "created": int(model_stat.st_mtime),
                "status": {
                    "value": synthetic_status_value(manager),
                    "args": config.server_command[1:],
                    "preset": "default",
                },
                "meta": {
                    "vocab_type": 0,
                    "n_vocab": 0,
                    "n_ctx_train": config.context_size,
                    "n_embd": 0,
                    "n_params": 0,
                    "size": model_stat.st_size,
                },
            }
        ],
    }


def build_openai_models_payload(manager: ServerManager) -> dict[str, object]:
    config = manager.config
    model_stat = config.model_path.stat()
    return {
        "object": "list",
        "data": [
            {
                "id": config.alias,
                "object": "model",
                "created": int(model_stat.st_mtime),
                "owned_by": "llama.cpp",
            }
        ],
    }


def build_props_payload(manager: ServerManager) -> dict[str, object]:
    return {
        "default_generation_settings": {},
        "total_slots": 1,
        "model_alias": manager.config.alias,
        "model_path": str(manager.config.model_path),
        "modalities": {"vision": True, "audio": False},
        "media_marker": "",
        "endpoint_slots": True,
        "endpoint_props": True,
        "endpoint_metrics": False,
        "webui": False,
        "webui_settings": {},
        "chat_template": "",
        "chat_template_caps": {},
        "bos_token": "",
        "eos_token": "",
        "build_info": "wake-on-demand-proxy",
        "is_sleeping": manager.state != "running",
    }


def build_slots_payload(manager: ServerManager) -> list[dict[str, object]]:
    return [
        {
            "id": 0,
            "n_ctx": manager.config.context_size,
            "speculative": False,
            "is_processing": manager.active_requests > 0,
        }
    ]


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


async def proxy_request(request: web.Request) -> web.StreamResponse:
    manager: ServerManager = request.app["manager"]
    session: ClientSession = request.app["session"]

    manager.begin_request()
    try:
        body = await request.read() if request.can_read_body else None
        body = _inject_cache_prompt(body, request.method, request.path)
        await manager.ensure_running()
    except Exception as exc:
        logging.exception("Backend is unavailable")
        manager.end_request()
        return web.json_response(
            {"error": "backend unavailable", "detail": str(exc)},
            status=503,
            headers={
                "Retry-After": str(RETRY_AFTER_SECONDS),
                "X-Backend": manager.state,
            },
        )

    target_url = f"{manager.config.backend_base_url}{request.rel_url}"
    headers = filter_request_headers(request.headers, manager.config.api_key)

    try:
        async with session.request(
            request.method,
            target_url,
            headers=headers,
            data=body,
            allow_redirects=False,
            timeout=None,
        ) as upstream:
            response_headers = filter_response_headers(upstream.headers)
            response_headers["X-Backend"] = "online"

            downstream = web.StreamResponse(
                status=upstream.status,
                reason=upstream.reason,
                headers=response_headers,
            )
            await downstream.prepare(request)

            try:
                is_sse = "text/event-stream" in upstream.headers.get("Content-Type", "")
                if is_sse:
                    # Poll with timeout so we can inject keep-alives during silent prefill phases
                    # (the async-for approach only fires when chunks arrive, missing the gap)
                    KEEPALIVE_INTERVAL = 25
                    while True:
                        try:
                            chunk = await asyncio.wait_for(
                                upstream.content.readany(), timeout=KEEPALIVE_INTERVAL
                            )
                        except asyncio.TimeoutError:
                            # No data for 25s — send SSE comment to reset Cloudflare's 100s edge timer
                            await downstream.write(b": keep-alive\n\n")
                            continue
                        if not chunk:
                            break
                        await downstream.write(chunk)
                else:
                    async for chunk in upstream.content.iter_any():
                        await downstream.write(chunk)
            except ConnectionResetError:
                logging.info("Client disconnected while streaming %s %s", request.method, request.rel_url)
            finally:
                with contextlib.suppress(ConnectionResetError, RuntimeError):
                    await downstream.write_eof()

            return downstream
    except ClientError as exc:
        logging.exception("Failed to proxy request to backend")
        return web.json_response(
            {"error": "bad gateway", "detail": str(exc)},
            status=502,
            headers={"X-Backend": manager.state},
        )
    finally:
        manager.end_request()


async def health_handler(request: web.Request) -> web.Response:
    manager: ServerManager = request.app["manager"]
    return web.json_response(
        {
            "status": "ok",
            "backend": manager.state,
            "active_requests": manager.active_requests,
        },
        headers={"X-Backend": manager.state},
    )


async def models_handler(request: web.Request) -> web.Response:
    manager: ServerManager = request.app["manager"]
    return web.json_response(
        build_models_payload(manager),
        headers={"X-Backend": manager.state},
    )


async def openai_models_handler(request: web.Request) -> web.Response:
    manager: ServerManager = request.app["manager"]
    return web.json_response(
        build_openai_models_payload(manager),
        headers={"X-Backend": manager.state},
    )


async def props_handler(request: web.Request) -> web.Response:
    manager: ServerManager = request.app["manager"]
    return web.json_response(
        build_props_payload(manager),
        headers={"X-Backend": manager.state},
    )


async def slots_handler(request: web.Request) -> web.Response:
    manager: ServerManager = request.app["manager"]
    return web.json_response(
        build_slots_payload(manager),
        headers={"X-Backend": manager.state},
    )


async def model_load_handler(request: web.Request) -> web.Response:
    manager: ServerManager = request.app["manager"]

    manager.begin_request()
    try:
        await manager.ensure_running()
        return web.json_response(
            {"status": "ok", "model": manager.config.alias},
            headers={"X-Backend": manager.state},
        )
    except Exception as exc:
        logging.exception("Failed to load backend from /models/load")
        return web.json_response(
            {"error": "backend unavailable", "detail": str(exc)},
            status=503,
            headers={
                "Retry-After": str(RETRY_AFTER_SECONDS),
                "X-Backend": manager.state,
            },
        )
    finally:
        manager.end_request()


async def model_unload_handler(request: web.Request) -> web.Response:
    manager: ServerManager = request.app["manager"]
    await manager.stop("remote unload request")
    return web.json_response(
        {"status": "ok", "model": manager.config.alias},
        headers={"X-Backend": manager.state},
    )


async def lifecycle_context(app: web.Application):
    session = ClientSession(timeout=ClientTimeout(total=None))
    manager = ServerManager(app["config"], session)
    watchdog_task = asyncio.create_task(idle_watchdog(manager))

    app["session"] = session
    app["manager"] = manager

    try:
        yield
    finally:
        watchdog_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watchdog_task
        await manager.stop("proxy shutdown")
        if manager._process_watch_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await manager._process_watch_task
        with contextlib.suppress(Exception):
            await session.close()


async def idle_watchdog(manager: ServerManager) -> None:
    while True:
        await asyncio.sleep(manager.config.idle_check_interval)
        await manager.stop_if_idle()


def build_app(config: ProxyConfig) -> web.Application:
    app = web.Application(client_max_size=128 * 1024 * 1024)
    app["config"] = config
    app.cleanup_ctx.append(lifecycle_context)
    app.router.add_route("*", "/health", health_handler)
    app.router.add_route("*", "/models", models_handler)
    app.router.add_route("POST", "/models/load", model_load_handler)
    app.router.add_route("POST", "/models/unload", model_unload_handler)
    app.router.add_route("*", "/props", props_handler)
    app.router.add_route("*", "/slots", slots_handler)
    app.router.add_route("*", "/v1/models", openai_models_handler)
    app.router.add_route("*", "/{tail:.*}", proxy_request)
    return app


def main() -> int:
    configure_logging()
    config = build_config()
    logging.info(
        "Starting proxy on %s:%s -> backend %s:%s",
        config.proxy_host,
        config.proxy_port,
        config.server_host,
        config.server_port,
    )
    logging.info(
        "Proxy config: model=%s context=%s idle_timeout=%ss",
        config.model_path.name,
        config.context_size,
        config.idle_timeout,
    )
    try:
        web.run_app(
            build_app(config),
            host=config.proxy_host,
            port=config.proxy_port,
            reuse_address=True,
        )
    except OSError as e:
        if "address already in use" in str(e).lower():
            logging.error(
                "Port %s is already in use. Another proxy or llama-server may still be running.",
                config.proxy_port,
            )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())