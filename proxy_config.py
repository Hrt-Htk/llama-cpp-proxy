"""Model metadata, preset generation, and CLI configuration for the chat proxy."""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from log_paths import (
    DATE_FMT,
    current_week_dir,
    local_now,
)

from proxy_base import API_KEY, ProxyConfig


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


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct and return the CLI argument parser for the chat proxy."""
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
    return p


def build_config() -> ProxyConfig:
    args = _build_arg_parser().parse_args()

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
