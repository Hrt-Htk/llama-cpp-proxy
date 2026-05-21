# Updating llama.cpp (prebuilt Windows-CUDA release)

Procedure to swap `H:\llama.cpp\llama.cpp_latest\` from **b9094** (May 10) to **b9209** (May 18) so the pi-llama-cpp extension stops crashing on the missing `architecture.input_modalities` field.

## What changes

- Replaces the prebuilt binaries in `llama.cpp_latest\` only.
- Models (`H:\llama.cpp\models\`), presets (`models-preset.ini`), logs, scripts, `.pi\` config — **untouched**.
- API key, server URL, watchdog — **untouched**.

## Prerequisites

- Server processes stopped (`llama-server.exe`, watchdog, embed proxy, any script that holds a model in memory). The current `llama-server` is busy serving an embedded-LLM script — let that finish first.
- ~2 GB free disk for the new zip + extracted files + rollback copy.
- CUDA runtime: your current build uses CUDA 12 (`cudart64_12.dll`, `cublas64_12.dll`), so grab the **12.4** zip, not 13.1.

## Steps

### 1. Download the release zip

```powershell
$url = "https://github.com/ggml-org/llama.cpp/releases/download/b9209/llama-b9209-bin-win-cuda-12.4-x64.zip"
$zip = "H:\llama.cpp\llama-b9209.zip"
Invoke-WebRequest -Uri $url -OutFile $zip
```

Verify the file landed and is ~150–300 MB:

```powershell
Get-Item $zip | Select-Object Name, Length
```

### 2. Stop everything that touches llama-server

In this order:

1. Stop the watchdog (`watchdog.ps1`) — otherwise it will respawn `llama-server.exe` mid-swap.
2. Stop any script using embedded LLM (the one currently running).
3. Stop the embed proxy (`watchdog-embed.ps1` / `embed_proxy.py`) if running.
4. Confirm no `llama-server.exe` processes remain:

```powershell
Get-Process llama-server -ErrorAction SilentlyContinue
```

If any are still alive, stop them gracefully (`Stop-Process -Name llama-server`) only after their callers are down.

### 3. Back up the current build

```powershell
Rename-Item "H:\llama.cpp\llama.cpp_latest" "H:\llama.cpp\llama.cpp_b9094"
```

Keeps a working rollback. Do **not** delete it until the new build is verified.

### 4. Extract the new build

```powershell
Expand-Archive -Path "H:\llama.cpp\llama-b9209.zip" -DestinationPath "H:\llama.cpp\llama.cpp_latest"
```

The zip extracts directly into a flat folder layout (matching the old one). Verify:

```powershell
& "H:\llama.cpp\llama.cpp_latest\llama-server.exe" --version
```

Expected: `version: 9209 (...)`.

### 5. Smoke-test the server standalone

Before re-enabling the watchdog, run the server manually against a small/cheap model to make sure CUDA loads and `/v1/models` returns the new field:

```powershell
# launch in another window
& "H:\llama.cpp\llama.cpp_latest\llama-server.exe" `
    --model "H:\llama.cpp\models\<one-small-gguf>" `
    --host 127.0.0.1 --port 8080 --n-gpu-layers 999
```

In a separate shell:

```powershell
curl http://127.0.0.1:8080/v1/models | Select-String architecture
```

Should print a line containing `"architecture"`. If yes → field is present, pi extension will work.

Stop the test server (Ctrl+C).

### 6. Restart your normal stack

1. Start the watchdog (`watchdog.ps1`) — it will relaunch `llama-server.exe` from the new folder.
2. Start the embed proxy.
3. Restart your embedded-LLM script.
4. Run `pi` — the extension should load without the destructure error and list your models.

### 7. Clean up (after a day or two of confidence)

```powershell
Remove-Item "H:\llama.cpp\llama-b9209.zip"
Remove-Item "H:\llama.cpp\llama.cpp_b9094" -Recurse -Force
```

## Rollback (if b9209 misbehaves)

```powershell
Remove-Item "H:\llama.cpp\llama.cpp_latest" -Recurse -Force
Rename-Item "H:\llama.cpp\llama.cpp_b9094" "H:\llama.cpp\llama.cpp_latest"
```

Restart the watchdog. You're back on b9094.

## Why this fixes pi

- pi-llama-cpp reads `model.architecture.input_modalities` from `/v1/models`.
- b9094 doesn't emit `architecture` at all → the extension's non-null assertion (`model.architecture!`) throws on load.
- b9209 emits `{"architecture": {"input_modalities": [...], "output_modalities": [...]}}` → destructure succeeds.

The field addition is purely additive on the server side, so any other client that talked to b9094 will keep working against b9209.

## Future: automate this

The folder is named `llama.cpp_latest` but only updates when manually swapped. A small `update-llama.ps1` that fetches the newest GitHub release tag, downloads the matching CUDA zip, and runs steps 2–4 would make the name truthful. Worth adding once the current upgrade is verified.
