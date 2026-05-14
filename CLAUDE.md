# Repo Lay of the Land

Two independent Python proxies wrap two `llama-server.exe` router instances.
Cloudflared exposes them under a single hostname; `proxy.py` reverse-proxies
`/embed/*` to the embed stack on :8003.

```
ai.htk-hrt.cc           → :8001 proxy.py        → :8002 router (chat)
ai.htk-hrt.cc/embed/*   → :8001 proxy.py        → :8003 embed_proxy.py → :8004 router (embeddings)
```

## Code

- `proxy.py` — chat proxy. Owns the chat router process, generates `models-preset.ini` at startup from `MODELS × CTX_CHOICES`, load/unload via `/models/load` & `/models/unload`, idle-unloads after 10 min. Has chat logging (`logs/chat-<date>.log`) and SSE chunk capture. Also reverse-proxies `/embed/*` to `embed_proxy.py` on :8003 (strips the `/embed` prefix; embed lifecycle stays owned by `embed_proxy.py`).
- `embed_proxy.py` — slimmed-down twin of `proxy.py` for the embedder. Single model, no chat logging, no SSE parsing. Same load/unload pattern.
- `watchdog.ps1` / `watchdog-embed.ps1` — thin restart-on-crash supervisors. They forward all extra args to the underlying Python script.
- `ping-both.py` — concurrent chat+embedding smoke test. `--local` skips the tunnel.

## Generated config (don't hand-edit)

- `models-preset.ini` — rewritten by `proxy.py` every startup.
- `embed-preset.ini` — rewritten by `embed_proxy.py` every startup.

## Binaries & data

- `llama.cpp_latest/llama-server.exe` — current router binary (b9094).
- `models/` — GGUF weights. `models/_aux/` holds mmproj projectors so `--models-dir`-style discovery wouldn't trip on them (we use presets, but the layout is kept).

## External

- `cloudflared/run-tunnel.ps1` — runs the tunnel; also checks sshd. Tunnel config at `C:\Users\HTK\.cloudflared\config.yml`.
- `.venv/` — project virtualenv. Python at `.venv\Scripts\python.exe`. Only deps used are `aiohttp` (proxies) and the stdlib.

## Logs

All under `logs/`. Daily-rotated. Two pairs (`<proxy>-<date>.log` from the proxy, `<server>-<date>.log` from llama-server) per stack, plus chat traces.

## Key design points to know before changing things

- **Router stays up across model loads/unloads** — that's the whole point. Don't kill the router process on idle; only call `/models/unload`.
- **`--models-max 1` per router** — loading a different model evicts the current one. The two routers don't share state, so chat and embedder coexist fine.
- **API key** comes from `$env:LLAMA_API_KEY` with a hardcoded fallback in both proxies. The proxies inject `Authorization: Bearer …` if the client omits it.
- **Preset IDs are the API model names.** Chat: `qwen3.6-35b-q3-32k`, `…-q4-128k`, etc. Embed: `qwen3-embedding-4b-8k`. Clients pick via the `model` field in the request body.

## Background

`ROUTER_MODE_SETUP.md` documents the migration from the old "kill the server to unload" proxy to router mode. Some sections (test-plan checkboxes, idle-timeout-of-10s) are historical.
