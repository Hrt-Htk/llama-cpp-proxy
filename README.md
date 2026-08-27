# llama.cpp Wake-on-Demand Proxy

Two `llama-server.exe` router instances behind small Python proxies. `proxy.py`
listens on `0.0.0.0:8001` and reverse-proxies `/embedding/*` to the embed stack.
No tunnel runs on this box — the ZBOX's Caddy is the entry point and forwards
to :8001. Each proxy loads its model on first request and unloads after
10 minutes of inactivity — the router process stays up so the front-end
connection never breaks.

```
:8001 proxy.py      → :8002 router (chat)
:8001 /embedding/*  → :8003 embed_proxy.py → :8004 router (embeddings)
```

Bare `/v1/...` at the root also still hits the chat router
(backwards compat); the `/chat` prefix is the preferred alias.
Both stacks run side-by-side as independent processes; both models can be loaded
concurrently.

## Files

| File | Role |
|---|---|
| `proxy.py` | Chat proxy + router supervisor. Generates `models-preset.ini` at startup. Also reverse-proxies `/embedding/*` to the embed stack. |
| `embed_proxy.py` | Embedding/re-ranking proxy + router supervisor. Generates `embed-preset.ini`. |
| `watchdog.ps1` | Restart-on-crash supervisor for `proxy.py`. |
| `watchdog-embed.ps1` | Restart-on-crash supervisor for `embed_proxy.py`. |
| `restart-watchdog.ps1` | Graceful midnight restart for chat watchdog (WM_CLOSE → cascade shutdown → relaunch). |
| `restart-watchdog-embed.ps1` | Same for embed watchdog, staggered 1 min after chat. |
| `create-scheduler-tasks.ps1` | Creates daily Task Scheduler entries (run once as Administrator). |
| `log_paths.py` | Shared log path resolution and formatting utilities. |
| `log_paths.ps1` | PowerShell helper for inspecting log paths. |
| `chat_template.jinja` | Custom Jinja chat template passed to llama-server. |
| `models-preset.ini` | Auto-generated (don't edit). Every `MODELS × CTX_CHOICES` combo as its own preset. |
| `embed-preset.ini` | Auto-generated (don't edit). Single preset for `Qwen3-Embedding-4B`. |
| `models/` | GGUF weights. `models/_aux/` holds mmproj projectors. |
| `llama.cpp_latest/llama-server.exe` | The router binary. |

## Prerequisites

1. **Python 3.x** — create and activate a virtualenv:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```
2. **Install dependencies** — only `aiohttp` is required:
   ```powershell
   pip install aiohttp
   ```
3. **Download `llama-server.exe`** — get a Windows build from [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) (b9209+ for MTP support) and place it in `llama.cpp_latest/`.
4. **Download GGUF models** — place your model files in the `models/` directory.
5. **Set your API key** — export the environment variable (overrides the hardcoded fallback in the proxy code):
   ```powershell
   $env:LLAMA_API_KEY = "your-secret-key"
   ```

## Run it

Two terminals for the proxies (service them however you like — they're independent):

```powershell
H:\llama.cpp\watchdog.ps1                  # chat stack on :8001
H:\llama.cpp\watchdog-embed.ps1            # embed stack on :8003
```

**Daily restart** — create Task Scheduler entries so watchdogs restart cleanly at midnight:

```powershell
H:\llama.cpp\create-scheduler-tasks.ps1    # run once as Administrator
```

**Entry point** — the ZBOX (`htk-ZBOX-PI336`, LAN 192.168.178.43 / Tailscale
100.80.201.44) runs Caddy, which forwards to `:8001` on this box. No tunnel
runs on this machine.

Headless chat variant (skip the interactive picker — sets the *fallback* model
when a client request omits the `model` field; all presets are still routable):

```powershell
H:\llama.cpp\watchdog.ps1 --model "Nail-Qwen3.6-35B-A3B Q4"
```

## Endpoints

- **Chat:** `/chat/v1/chat/completions` (via the ZBOX entry point) — model field selects preset (e.g. `nail-35b-a3b-q4-262k`).
- **Embeddings:** `/embedding/v1/embeddings` — model `qwen3-embedding-4b-8k`.
- **Re-ranking:** `/embedding/v1/rerank` — same embed router.
- Both require `Authorization: Bearer <key>`. Set via `$env:LLAMA_API_KEY` (overrides the hardcoded fallback in the proxy code).

`/health` on either port reports proxy + router state and which model is currently loaded.

## How the load/unload flow works

1. Watchdog starts the proxy. Proxy spawns the router with `--no-models-autoload` and `--models-max 1`. GPU is idle, router is healthy.
2. First chat/embedding request arrives. Proxy POSTs `/models/load` to the router, polls `/v1/models` until `status.value == "loaded"`, then forwards.
3. Idle watchdog (inside each proxy) checks every 30s. After 10 minutes of no activity it POSTs `/models/unload`. VRAM frees; router stays running; front-end connection stays alive.
4. Next request reloads on demand.

Cold-load latency: ~15–20s for the 35B chat model on a 3090 Ti; ~2–3s for the 4B embedder.

## Adding a model

Chat side — append a `ModelChoice(...)` to `MODELS` in `proxy_config.py`. Every entry is crossed with `CTX_CHOICES` (currently a single 262k) to produce one preset per (model × ctx) pair; a model can pin its own `ctx_choices` to override. No INI editing.

Each preset uses KV cache quantization (`cache-type-k = q4_0`, `cache-type-v = q4_0`), flash attention, and a custom Jinja chat template (`chat_template.jinja` with `preserve_thinking` kwarg).

**MTP (speculative decoding)** — set `spec_mtp=True` on a `ModelChoice` to enable built-in MTP speculative decoding. This emits `spec-type=draft-mtp`, `spec-draft-n-max=2`, `spec-draft-p-min=0.0` in the preset. Requires the b9209+ router binary (PR ggml-org/llama.cpp#22673). The 27B model (`qwen3.8-27b-q4-mtp`) uses `models/Qwen3.8-27B-UD-Q4_K_XL.gguf`, which ships the MTP draft head baked in.

Embed side — `embed_proxy.py` is hardcoded to a single model (`MODEL_FILE`, `MODEL_ID`, `CTX_SIZE` constants at the top). Change those if you swap the embedder.

## Logs

- `logs/<week>/proxy-<date>.log` — chat proxy events
- `logs/<week>/embed-proxy-<date>.log` — embed proxy events
- `logs/<week>/llama-server-<date>.log` — chat router output
- `logs/<week>/embed-server-<date>.log` — embed router output
- `logs/<week>/chat-<date>.log` and `chat-<date>.raw.jsonl` — per-request chat traces (proxy.py only)
- `logs/<week>/watchdog-restart-<date>.log` — daily restart logs
- All logs are bucketed by ISO week folder.
