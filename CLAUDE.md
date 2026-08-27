# Repo Lay of the Land

Two independent Python proxies wrap two `llama-server.exe` router instances.
`proxy.py` listens on `0.0.0.0:8001` and reverse-proxies `/embedding/*` to the
embed stack on :8003. No tunnel runs on this box — the ZBOX's Caddy is the
entry point and forwards to :8001.

```
:8001 proxy.py      → :8002 router (chat)
:8001 /embedding/*  → :8003 embed_proxy.py → :8004 router (embeddings)
```

Bare `/v1/...` at the root also still hits the chat router
(backwards compat); the `/chat` prefix is the preferred alias.

## Code

- `proxy.py` — slim entry point (app lifecycle, route registration). Imports handlers from `proxy_request_handlers` and config from `proxy_config`.
- `proxy_config.py` — model metadata, preset generation, CLI config.
- `proxy_request_handlers.py` — HTTP request handlers, streaming, retry/recovery.
- `embed_proxy.py` — slimmed-down twin of `proxy.py` for the embedder. Single model, no chat logging, no SSE parsing. Same load/unload pattern. Supports embeddings and re-ranking (`/v1/embeddings`, `/v1/rerank`).
- `watchdog.ps1` / `watchdog-embed.ps1` — thin restart-on-crash supervisors. They forward all extra args to the underlying Python script.
- `restart-watchdog.ps1` / `restart-watchdog-embed.ps1` — graceful midnight restarts (WM_CLOSE → cascade shutdown → relaunch). Run via Task Scheduler.
- `create-scheduler-tasks.ps1` — creates daily Task Scheduler entries for the restart scripts (run once as Administrator).
- `log_paths.py` / `log_paths.ps1` — shared log path utilities (weekly buckets, local timestamps).
- `chat_template.jinja` — custom Jinja chat template with `preserve_thinking` kwarg.

## Generated config (don't hand-edit)

- `models-preset.ini` — rewritten by `proxy.py` every startup.
- `embed-preset.ini` — rewritten by `embed_proxy.py` every startup.

## Binaries & data

- `llama.cpp_latest/llama-server.exe` — current router binary (b9209+ for MTP).
- `models/` — GGUF weights. `models/_aux/` holds mmproj projectors.

## External

- `.venv/` — project virtualenv. Python at `.venv\Scripts\python.exe`. Only deps used are `aiohttp` (proxies) and the stdlib.
- ZBOX (`htk-ZBOX-PI336`, LAN 192.168.178.43 / Tailscale 100.80.201.44) — entry point; its Caddy forwards to :8001 on this box. The old cloudflared config is archived at `C:\Users\HTK\.cloudflared.disabled`.

## Logs

All under `logs/<week>/`. Daily-rotated, bucketed by ISO week. Two pairs (`proxy-<date>.log` + `llama-server-<date>.log` for chat, `embed-proxy-<date>.log` + `embed-server-<date>.log` for embed), plus chat traces and watchdog-restart logs.

## Key design points to know before changing things

- **Router stays up across model loads/unloads** — that's the whole point. Don't kill the router process on idle; only call `/models/unload`.
- **`--models-max 1` per router** — loading a different model evicts the current one. The two routers don't share state, so chat and embedder coexist fine.
- **MTP (built-in speculative decoding)** — `spec_mtp=True` on a `ModelChoice` enables it (emits `spec-type=draft-mtp` etc. in the preset; see `proxy_config.py` for the current 27B setup). Requires the b9209+ router binary and a build supporting the model's GGUF arch. Only one model loads at a time, so the extra preset costs no VRAM unless selected.
- **API key** comes from `$env:LLAMA_API_KEY` with a hardcoded fallback in both proxies. The proxies inject `Authorization: Bearer …` if the client omits it.
- **Preset IDs are the API model names** — `<base-id>-<ctx>k`, generated from `MODELS` × `CTX_CHOICES` in `proxy_config.py` (don't hardcode the list here; it rots). Clients pick via the `model` field in the request body.
- **KV cache is quantized** — `cache-type-k = q4_0`, `cache-type-v = q4_0` in all presets.
- **Custom chat template** — router uses `--jinja --chat-template-file chat_template.jinja --chat-template-kwargs '{"preserve_thinking":true}'`.

## Git Workflow

Full rules in `docs/dev/workflow.md`. Summary:

- **Branches:** `type/issueN-description` (e.g. `feat/issue29-workflow-docs`). Types: `feat`, `fix`, `chore`, `docs`, `refactor`, `test`, `perf`, `build`, `ci`, `style`.
- **Commits:** [Conventional Commits](https://www.conventionalcommits.org/) — `type(scope): description`.
- **Issue reference:** every commit or PR body must include `#N` linking to a GitHub issue.
- **Size limits (soft):** 10 files, 300 lines per PR — exceeding triggers a CI warning only.
- **Never commit secrets.** `.env` (API key) is gitignored. Never reference actual keys in code, docs, or commits.
