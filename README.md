# llama.cpp Wake-on-Demand Proxy

This workspace runs `llama-server.exe` behind an always-on Python proxy.

- Proxy listens on `8001`
- `llama-server` listens on `8002`
- The proxy starts `llama-server` on the first real request
- The proxy stops `llama-server` after the configured idle timeout

The main launcher is `watchdog_qwen.ps1`. It resolves Python, installs `aiohttp` if needed, starts `proxy.py`, and optionally restarts the proxy if it crashes.

## Files

- `watchdog_qwen.ps1` - main startup script
- `proxy.py` - always-on wake-on-demand proxy
- `watchdog.log` - launcher log output

## Startup Sequence

Open PowerShell in `H:\llama.cpp`, then run one of these commands.

### Normal Start

This keeps the proxy supervised and uses the default 10 minute idle timeout.

```powershell
.\watchdog_qwen.ps1
```

### Manual Test Start

This is useful when validating wake and sleep behavior quickly.

```powershell
.\watchdog_qwen.ps1 -noRestart -idleTimeout 60
```

Notes:

- `-noRestart` runs the proxy once and does not relaunch it if it exits
- `-idleTimeout 60` asks the proxy to stop the backend after 60 idle seconds
- Idle shutdown is checked periodically, so the actual stop can happen a bit later than the exact timeout

### Other Variants

```powershell
.\watchdog_qwen.ps1 -q4
.\watchdog_qwen.ps1 -long
.\watchdog_qwen.ps1 -q4 -long
```

- `-q4` uses `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`
- `-long` uses a `262144` context window instead of `131072`

## What Happens During Boot

1. `watchdog_qwen.ps1` starts `proxy.py`
2. `proxy.py` binds to `0.0.0.0:8001`
3. The proxy answers `/health`, `/models`, `/props`, and `/slots` immediately
4. Pi connects to the proxy on port `8001`
5. The first chat request causes the proxy to launch `llama-server.exe` on `127.0.0.1:8002`
6. The proxy waits for backend health, then forwards the request
7. After the backend stays idle past the timeout, the proxy stops it and frees VRAM

## Pi Boot Sequence

After the proxy is running:

1. Start Pi normally
2. Let Pi connect to the configured llama.cpp URL
3. Send a first message such as `hey there`
4. Expect the first response to be slower because it includes backend startup time
5. Wait for the idle timeout to expire
6. Send another message to confirm the backend wakes again

## Quick Checks

### Check Proxy Health

```powershell
Invoke-WebRequest -Uri 'http://127.0.0.1:8001/health' -UseBasicParsing
```

Expected behavior:

- HTTP `200`
- JSON contains `"status": "ok"`
- Header `X-Backend` is `offline`, `booting`, or `running`

### Check Synthetic Model Discovery

```powershell
Invoke-RestMethod -Uri 'http://127.0.0.1:8001/models' -Headers @{ Authorization = 'Bearer rRZsSjRvaUuRMr5AeDA14rO9jaSlhSRhRtBI5ZlO' }
```

This endpoint lets `pi-llama-cpp` discover the model even while the backend is asleep.

### Check Whether Backend Is Running

```powershell
Get-NetTCPConnection -LocalPort 8002 -State Listen
Get-Process -Name 'llama-server'
```

## Stop Sequence

If you started with `-noRestart`, stop it with `Ctrl+C` in that PowerShell window.

If you started without `-noRestart`, `Ctrl+C` also stops the supervised proxy loop.

## Troubleshooting

- If Pi says no models are available, verify `http://127.0.0.1:8001/health` returns `status: ok`
- If Pi can connect but first inference fails, inspect `watchdog.log` and the active PowerShell window for `llama-server` startup errors
- If `aiohttp` is missing, rerun `watchdog_qwen.ps1`; it will try to install it automatically
- If port `8001` is already in use, stop the conflicting process before starting the proxy