from __future__ import annotations

import argparse
import asyncio
import contextlib
from datetime import datetime, timezone
import json
import logging
import os
import secrets
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from aiohttp import ClientError, ClientSession, ClientTimeout, web

# Enable ANSI escape sequences on Windows 10+
if sys.platform == "win32":
    os.system("")


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
]

CTX_CHOICES: list[int] = [32768, 65536, 131072]

PROXY_HOST = "::"
PROXY_PORT = 8001
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8002
IDLE_TIMEOUT = 600  # 10 minutes of inactivity before unload
IDLE_CHECK_INTERVAL = 30  # check every 30s
HEALTH_POLL_INTERVAL = 1.0
BOOT_TIMEOUT = 60
LOAD_TIMEOUT = 300
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
    default_model: str
    api_key: str
    chat_log: bool = True

    @property
    def backend_base_url(self) -> str:
        return f"http://{self.server_host}:{self.server_port}"

    @property
    def server_command(self) -> list[str]:
        log_file = ROOT / "logs" / f"llama-server-{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.log"
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
    """Owns the long-lived router process and on-demand model load/unload.

    The router process starts with the proxy and dies with it. Models are
    loaded on first request and unloaded by the idle watchdog — the router
    itself stays up so the cloudflared tunnel never breaks.
    """

    def __init__(self, config: ProxyConfig, session: ClientSession) -> None:
        self.config = config
        self.session = session
        self.process: asyncio.subprocess.Process | None = None
        self._loaded: str | None = None  # alias of currently-loaded model, None if nothing
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
            "Starting router on %s:%s", self.config.server_host, self.config.server_port,
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
                        logging.info("router is ready")
                        return
            except (ClientError, asyncio.TimeoutError):
                pass
            await asyncio.sleep(self.config.health_poll_interval)
        raise TimeoutError(f"router did not become healthy in {self.config.boot_timeout}s")

    async def stop_server(self) -> None:
        if not self.server_running:
            return
        logging.info("Stopping router")
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
            # The router may already have the model loaded (e.g. a client called
            # /models/load directly through the proxy). Sync state before issuing
            # another load — otherwise the router returns 400 "already running".
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


RAW_BODY_CAP = 1024 * 1024


class ChatLogger:
    """Rotating chat logger — one file per day in the logs/ directory."""

    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._date: str | None = None
        self._fh = None
        self._raw_fh = None
        self._lock = asyncio.Lock()
        self._open_for_today()

    def _open_for_today(self) -> None:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._date == date and self._fh is not None:
            return
        if self._fh is not None:
            self._fh.close()
        if self._raw_fh is not None:
            self._raw_fh.close()
        self.log_file = self.log_dir / f"chat-{date}.log"
        self.raw_file = self.log_dir / f"chat-{date}.raw.jsonl"
        self._fh = open(self.log_file, "a", encoding="utf-8")
        self._raw_fh = open(self.raw_file, "a", encoding="utf-8")
        self._date = date

    async def log_request(self, method: str, path: str, body: bytes | None, req_id: str) -> None:
        async with self._lock:
            self._open_for_today()
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            self._fh.write(f"=== [{ts}] [req={req_id}] {method} {path} ===\n")
            if body and path.rstrip("/") == "/v1/chat/completions":
                self._write_latest_user_turn(body)
            self._fh.flush()
            self._write_raw(ts, method, path, body, req_id)

    def _write_raw(self, ts: str, method: str, path: str, body: bytes | None, req_id: str) -> None:
        record: dict[str, object] = {"ts": ts, "req_id": req_id, "method": method, "path": path}
        if body is None:
            record["body"] = None
        elif len(body) > RAW_BODY_CAP:
            record["body"] = None
            record["body_truncated"] = body[:RAW_BODY_CAP].decode("utf-8", errors="replace")
            record["original_size"] = len(body)
        else:
            try:
                record["body"] = json.loads(body)
            except (ValueError, TypeError):
                record["body_raw"] = body.decode("utf-8", errors="replace")
        self._raw_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._raw_fh.flush()

    def _write_latest_user_turn(self, body: bytes) -> None:
        try:
            payload = json.loads(body)
        except (ValueError, TypeError):
            return
        messages = payload.get("messages") or []
        if not messages:
            return
        last = messages[-1]
        if last.get("role") != "user":
            return
        text = _stringify_message_content(last.get("content"))
        if text:
            self._fh.write(f"  [user] {text}\n")

    async def log_response(self, data: str, is_done: bool) -> None:
        async with self._lock:
            self._open_for_today()
            ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
            if is_done:
                self._fh.write(f"  [{ts}] [DONE]\n")
            else:
                self._fh.write(f"  [{ts}] {data}\n")
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        if self._raw_fh is not None:
            self._raw_fh.close()
            self._raw_fh = None


def _stringify_message_content(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.replace("\n", " ").strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            ctype = item.get("type")
            if ctype == "text" and item.get("text"):
                parts.append(str(item["text"]).replace("\n", " ").strip())
            elif ctype in ("image_url", "image"):
                parts.append("[image]")
            elif ctype == "input_audio":
                parts.append("[audio]")
        return " ".join(p for p in parts if p)
    return str(content)


class SSEChunkLogger:
    def __init__(self, wrapped, chat_logger: ChatLogger) -> None:
        self._wrapped = wrapped
        self._chat_logger = chat_logger
        self._buffer = b""
        self._current_kind: str | None = None
        self._current_text = ""
        self._tool_calls: dict[int, dict[str, str]] = {}

    async def _flush_text(self) -> None:
        if self._current_kind and self._current_text:
            await self._chat_logger.log_response(
                f"[{self._current_kind}] {self._current_text.strip()}", False
            )
        self._current_kind = None
        self._current_text = ""

    async def _flush_tool_calls(self) -> None:
        if not self._tool_calls:
            return
        for idx in sorted(self._tool_calls):
            tc = self._tool_calls[idx]
            name = tc.get("name") or "?"
            args = tc.get("arguments") or ""
            await self._chat_logger.log_response(f"[tool_call] {name}({args})", False)
        self._tool_calls = {}

    async def _flush_all(self) -> None:
        await self._flush_text()
        await self._flush_tool_calls()

    async def readany(self) -> bytes:
        data = await self._wrapped.content.readany()
        if not data:
            await self._flush_all()
            return data
        self._buffer += data
        while True:
            crlf_idx = self._buffer.find(b"\r\n\r\n")
            lf_idx = self._buffer.find(b"\n\n")
            if crlf_idx == -1 and lf_idx == -1:
                break
            if crlf_idx != -1 and (lf_idx == -1 or crlf_idx <= lf_idx):
                idx, sep_len = crlf_idx, 4
            else:
                idx, sep_len = lf_idx, 2
            event = self._buffer[:idx]
            self._buffer = self._buffer[idx + sep_len:]
            text = event.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            payload = ""
            for line in text.splitlines():
                if line.startswith("data:"):
                    payload = line[5:].strip()
            if not payload:
                continue
            if payload == "[DONE]":
                await self._flush_all()
                await self._chat_logger.log_response("[DONE]", True)
                continue
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            reasoning = delta.get("reasoning_content")
            content = delta.get("content")
            tool_calls = delta.get("tool_calls")
            if reasoning:
                if self._current_kind != "thinking":
                    await self._flush_all()
                    self._current_kind = "thinking"
                self._current_text += reasoning
            if content:
                if self._current_kind != "content":
                    await self._flush_all()
                    self._current_kind = "content"
                self._current_text += content
            if tool_calls:
                await self._flush_text()
                for tc in tool_calls:
                    i = tc.get("index", 0)
                    slot = self._tool_calls.setdefault(i, {"name": "", "arguments": ""})
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["arguments"] += fn["arguments"]
        return data

    def __getattr__(self, name: str) -> object:
        return getattr(self._wrapped, name)


def configure_logging() -> None:
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / datetime.now(timezone.utc).strftime("%Y-%m-%d.log")
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
    p.add_argument("--no-chat-log", action="store_true")
    args = p.parse_args()

    if not SERVER_EXE.exists():
        raise SystemExit(f"Missing required file: {SERVER_EXE}")

    model, ctx = pick_setup(args.model, args.ctx_size)
    write_preset(MODELS, CTX_CHOICES)
    default_id = model.preset_id(ctx)
    print(f"Default: {model.label} @ {ctx // 1024}k ctx (id: {default_id})")
    print(f"Exposed presets: {len(MODELS) * len(CTX_CHOICES)} (one per model×ctx combo)")

    return ProxyConfig(
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
        chat_log=not args.no_chat_log,
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
    manager: ModelManager = request.app["manager"]
    session: ClientSession = request.app["session"]
    chat_logger: ChatLogger | None = request.app.get("chat_logger")

    req_id = secrets.token_hex(4)
    started = time.monotonic()
    manager.begin_request()

    try:
        body = await request.read() if request.can_read_body else None
        body = _inject_cache_prompt(body, request.method, request.path)
        # Only ensure-load for endpoints that need a model
        path = request.path.rstrip("/")
        if path in ("/v1/chat/completions", "/v1/completions", "/v1/embeddings"):
            model = _model_from_body(body, manager.config.default_model)
            await manager.ensure_loaded(model)
    except Exception as exc:
        logging.exception("[req=%s] backend unavailable", req_id)
        manager.end_request()
        return web.json_response(
            {"error": "backend unavailable", "detail": str(exc)},
            status=503,
            headers={"Retry-After": str(RETRY_AFTER_SECONDS), "X-Request-ID": req_id},
        )

    if chat_logger is not None:
        await chat_logger.log_request(request.method, request.path, body, req_id)

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
            is_sse = "text/event-stream" in upstream_resp.headers.get("Content-Type", "")
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
        finally:
            with contextlib.suppress(ConnectionResetError, RuntimeError):
                await downstream.write_eof()
            duration_ms = int((time.monotonic() - started) * 1000)
            logging.info(
                "[req=%s] %s %s -> %s in %dms",
                req_id, request.method, request.rel_url, downstream.status, duration_ms,
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
    manager: ModelManager = request.app["manager"]
    session: ClientSession = request.app["session"]
    target_url = f"{manager.config.backend_base_url}{request.rel_url}"
    headers = filter_request_headers(request.headers, manager.config.api_key)
    async with session.get(target_url, headers=headers) as upstream:
        body_text = await upstream.text()
        try:
            payload = json.loads(body_text)
        except (ValueError, TypeError):
            return web.Response(status=upstream.status, body=body_text,
                                content_type=upstream.headers.get("Content-Type", "application/json"))
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
    manager: ModelManager = request.app["manager"]
    session: ClientSession = request.app["session"]
    target_url = f"{manager.config.backend_base_url}{request.rel_url}"
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
            content_type=upstream.headers.get("Content-Type", "application/json"),
        )


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


async def idle_watchdog(manager: ModelManager) -> None:
    while True:
        await asyncio.sleep(manager.config.idle_check_interval)
        try:
            await manager.unload_if_idle()
        except Exception:
            logging.exception("idle watchdog error")


def build_app(config: ProxyConfig) -> web.Application:
    app = web.Application(client_max_size=128 * 1024 * 1024)
    app["config"] = config
    app.cleanup_ctx.append(lifecycle_context)
    app.router.add_route("GET", "/health", health_handler)
    app.router.add_route("GET", "/models", models_handler)
    app.router.add_route("GET", "/props", props_handler)
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
            access_log_format='%a req=%{X-Request-ID}o "%r" %s %b %Dus',
        )
    except OSError as e:
        if "address already in use" in str(e).lower():
            logging.error("Port %s already in use", config.proxy_port)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
