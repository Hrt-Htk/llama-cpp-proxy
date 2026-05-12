# llama.cpp Wake-on-Demand Proxy

Two `llama-server.exe` router instances behind small Python proxies, exposed over a
cloudflared tunnel. Each proxy loads its model on first request and unloads after
10 minutes of inactivity — the router process stays up so the tunnel never breaks.

```
cloudflared
  ├─ ai.htk-hrt.cc      → :8001 proxy.py        → :8002 router (chat models)
  └─ embed.htk-hrt.cc   → :8003 embed_proxy.py  → :8004 router (embedder)
```

Both stacks run side-by-side as independent processes; both models can be loaded
concurrently.

## Files

| File | Role |
|---|---|
| `proxy.py` | Chat proxy + router supervisor. Generates `models-preset.ini` at startup. |
| `embed_proxy.py` | Embedding proxy + router supervisor. Generates `embed-preset.ini`. |
| `watchdog.ps1` | Restart-on-crash supervisor for `proxy.py`. |
| `watchdog-embed.ps1` | Restart-on-crash supervisor for `embed_proxy.py`. |
| `models-preset.ini` | Auto-generated. Every `MODELS × CTX_CHOICES` combo as its own preset. |
| `embed-preset.ini` | Auto-generated. Single preset for `Qwen3-Embedding-4B`. |
| `ping-both.py` | Fires a chat + embedding request concurrently end-to-end. |
| `cloudflared/run-tunnel.ps1` | Launches `cloudflared.exe` with daily log rotation. |
| `models/` | GGUF weights. `models/_aux/` holds mmproj projectors. |
| `llama.cpp_latest/llama-server.exe` | The router binary. |

## Run it

Three terminals (or service them however you like — they're independent):

```powershell
H:\llama.cpp\watchdog.ps1                  # chat stack on :8001
H:\llama.cpp\watchdog-embed.ps1            # embed stack on :8003
H:\cloudflared\run-tunnel.ps1              # public tunnel
```

Headless chat variant (skip the interactive picker — sets the *fallback* model
when a client request omits the `model` field; all combos are still routable):

```powershell
H:\llama.cpp\watchdog.ps1 --model "Qwen3.6-35B-A3B Q3" --ctx-size 32768
```

## Endpoints

- **Chat:** `https://ai.htk-hrt.cc/v1/chat/completions` — model field selects preset (e.g. `qwen3.6-35b-q3-32k`).
- **Embeddings:** `https://embed.htk-hrt.cc/v1/embeddings` — model `qwen3-embedding-4b-8k`.
- Both require `Authorization: Bearer $LLAMA_API_KEY`.

`/health` on either port reports proxy + router state and which model is currently loaded.

## Verify

```powershell
H:\llama.cpp\.venv\Scripts\python.exe H:\llama.cpp\ping-both.py            # through the tunnel
H:\llama.cpp\.venv\Scripts\python.exe H:\llama.cpp\ping-both.py --local    # localhost only
```

## How the load/unload flow works

1. Watchdog starts the proxy. Proxy spawns the router with `--no-models-autoload` and `--models-max 1`. GPU is idle, router is healthy.
2. First chat/embedding request arrives. Proxy POSTs `/models/load` to the router, polls `/v1/models` until `status.value == "loaded"`, then forwards.
3. Idle watchdog (inside each proxy) checks every 30s. After 10 minutes of no activity it POSTs `/models/unload`. VRAM frees; router stays running; tunnel socket stays alive.
4. Next request reloads on demand.

Cold-load latency: ~15–20s for the 35B chat model on a 3090 Ti; ~2–3s for the 4B embedder.

## Adding a model

Chat side — append a `ModelChoice(...)` to `MODELS` in `proxy.py`. Every entry is automatically crossed with `CTX_CHOICES` to produce one preset per (model × ctx) pair. No INI editing.

Embed side — `embed_proxy.py` is hardcoded to a single model (`MODEL_FILE`, `MODEL_ID`, `CTX_SIZE` constants at the top). Change those if you swap the embedder.

## Logs

- `logs/<date>.log` — chat proxy events
- `logs/embed-proxy-<date>.log` — embed proxy events
- `logs/llama-server-<date>.log` — chat router output
- `logs/embed-server-<date>.log` — embed router output
- `logs/chat-<date>.log` and `logs/chat-<date>.raw.jsonl` — per-request chat traces (proxy.py only)
- `cloudflared/logs/<date>.log` — tunnel logs

## Background docs

`ROUTER_MODE_SETUP.md` is the migration write-up from the pre-router proxy to the
current router-mode design. Useful context, not required reading.
