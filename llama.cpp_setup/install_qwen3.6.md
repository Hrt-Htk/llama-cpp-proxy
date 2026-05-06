# Installing Qwen3.6-35B-A3B on Windows with llama.cpp

A focused guide for an **RTX 3090 (24 GB VRAM)** targeting long-context coding work. No WSL, no compilation — just prebuilt binaries.

---

## ⚠️ Critical warning before you start

**Do NOT use CUDA 13.2** — Unsloth has flagged this version for producing gibberish output with Qwen3.6. NVIDIA is working on a fix. Stick with CUDA 12.x.

Check your CUDA version in PowerShell:
```powershell
nvcc --version
```

If you don't have the CUDA Toolkit installed, that's fine — the prebuilt llama.cpp binaries bundle their own CUDA runtime. You only need an up-to-date NVIDIA GeForce driver (552.xx or newer). Get it from [nvidia.com/Download](https://www.nvidia.com/Download/index.aspx).

---

## Prerequisites

- Windows 10 or 11
- Recent NVIDIA driver (552.xx+)
- ~30 GB free disk space for the model
- Python 3.10+ with pip (only needed once, for downloading the model)

---

## Step 1 — Get llama.cpp prebuilt CUDA binaries

1. Go to https://github.com/ggml-org/llama.cpp/releases
2. On the latest release, download `llama-bXXXX-bin-win-cuda-x64.zip` (XXXX = build number)
3. Unzip to a permanent location, e.g. `C:\llama.cpp\`

You should now have `llama-server.exe`, `llama-cli.exe`, and supporting DLLs.

Open PowerShell and `cd` into that directory:
```powershell
cd C:\llama.cpp
```

---

## Step 2 — Download the model

**UD-Q4_K_XL** is the right pick for a 24 GB card — ~22 GB, leaves room for KV cache, very close to BF16 quality per Unsloth's KL-divergence benchmarks.

```powershell
pip install huggingface_hub hf_transfer
$env:HF_HUB_ENABLE_HF_TRANSFER=1

hf download unsloth/Qwen3.6-35B-A3B-GGUF `
  --local-dir C:\models\Qwen3.6-35B-A3B-GGUF `
  --include "*mmproj-F16*" `
  --include "*UD-Q4_K_XL*"
```

This pulls ~22 GB of model weights plus the multimodal projection file (vision support). Backticks (`` ` ``) are PowerShell line continuations.

---

## Step 3 — Run llama-server

```powershell
.\llama-server.exe `
  --model C:\models\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf `
  --mmproj C:\models\Qwen3.6-35B-A3B-GGUF\mmproj-F16.gguf `
  --alias "qwen3.6-35b-a3b" `
  -ngl 999 `
  -fa on `
  --cache-type-k q8_0 `
  --cache-type-v q8_0 `
  -c 131072 `
  --jinja `
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 `
  --presence-penalty 0.0 `
  --port 8001 `
  --host 127.0.0.1
```

**What the key flags do:**

| Flag | Purpose |
|------|---------|
| `-ngl 999` | Offload all model layers to GPU (without this, you get CPU speeds) |
| `-fa on` | Flash attention — saves memory, faster |
| `--cache-type-k q8_0 --cache-type-v q8_0` | 8-bit KV cache; this is what makes 128k context fit |
| `-c 131072` | 128k context window (push to `262144` once everything works) |
| `--jinja` | Use chat template from GGUF (needed for tool calling) |

Sampling params above are tuned for **precise coding tasks**. For general/exploratory chat, swap to `--temp 1.0 --presence-penalty 1.5`.

Leave this terminal open. Model loads in 30-60 seconds.

---

## Step 4 — Verify it works

The simplest check: open **http://localhost:8001** in a browser. llama-server has a built-in chat UI.

Or from a second PowerShell terminal:
```powershell
curl.exe http://localhost:8001/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{\"model\":\"qwen3.6-35b-a3b\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a Python fizzbuzz.\"}]}'
```

Watch VRAM usage in **Task Manager → Performance → GPU**. You should see ~22-23 GB used on the 3090, growing as context fills.

---

## Thinking mode controls

Qwen3.6 thinks by default. To toggle, add to the server command:

```powershell
# Disable thinking (faster responses, less depth)
--chat-template-kwargs "{\"enable_thinking\":false}"

# Preserve thinking across turns (best for agent loops)
--chat-template-kwargs "{\"preserve_thinking\":true}"
```

Note the Windows escaping: `\"` inside double-quoted strings.

---

## Pointing coding tools at it

Any tool that accepts an OpenAI-compatible endpoint works. Set:
- `OPENAI_BASE_URL` = `http://localhost:8001/v1`
- `OPENAI_API_KEY` = `sk-no-key-required` (any non-empty string)

**Tested compatible:**
- **Aider**: `aider --openai-api-base http://localhost:8001/v1 --openai-api-key dummy --model openai/qwen3.6-35b-a3b`
- **Cline** (VS Code): add as "OpenAI Compatible" provider in settings
- **Continue.dev** (VS Code/JetBrains): same — "OpenAI Compatible"
- **OpenCode**, **Qwen Code**: native support
- **Claude Code**: see https://unsloth.ai/docs/basics/claude-code for env-var setup

---

## Troubleshooting

**Gibberish output** → check CUDA version (must NOT be 13.2). If still bad, try `--cache-type-k bf16 --cache-type-v bf16` to disable KV cache quantization.

**OOM on startup** → drop to UD-Q3_K_XL (~17 GB) or reduce `-c` to 65536.

**Fast at first, then crawls** → context is spilling to system RAM. Reduce `-c` or use a smaller quant.

**Model loading on the wrong GPU** → Windows may put the 2070 as GPU 0. Force the 3090 before running:
```powershell
$env:CUDA_VISIBLE_DEVICES=1
```
Use `nvidia-smi -L` to see which index is the 3090 (it'll usually be 0 or 1).

**Server starts but tools can't connect** → Windows Defender Firewall prompt; allow on first run. Confirm `--host 127.0.0.1` matches what your tool is calling.

**Slow first prompt, fast after** → normal. First prompt processes the system message and warms the KV cache; subsequent turns reuse it.

---

## Quant reference

| Quant | Size | Use case |
|-------|------|----------|
| UD-Q2_K_XL | ~12 GB | Tight VRAM, accept quality loss |
| UD-Q3_K_XL | ~17 GB | Maximum context room, mild quality dip |
| **UD-Q4_K_XL** | **~22 GB** | **Recommended for 24 GB cards** |
| Q4_K_M | 24 GB | Standard quant; fills VRAM, no headroom |
| Q8_0 | ~38 GB | Won't fit on 3090; would need RAM offload |

---

## Useful links

- Unsloth official guide: https://unsloth.ai/docs/models/qwen3.6
- Model on Hugging Face: https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF
- llama.cpp releases (Windows CUDA builds): https://github.com/ggml-org/llama.cpp/releases
- llama-server flag reference: https://github.com/ggml-org/llama.cpp/tree/master/tools/server
- Qwen3.6 model card (architecture, recommended sampling): https://huggingface.co/Qwen/Qwen3.6-35B-A3B
