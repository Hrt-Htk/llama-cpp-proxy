# Custom Proxy + Router Mode vs. llama-swap — Detailed Comparison

> Decision doc for whether to replace the current hand-rolled stack
> (`proxy.py` + `embed_proxy.py` wrapping `llama-server --router`) with
> [llama-swap](https://github.com/mostlygeek/llama-swap).
> Written for **this** deployment: a single Windows box (RTX 3090 Ti), Caddy on a
> separate Linux machine doing dumb LAN ingress, MTP/speculative on the 27B,
> a resident embedder, and chat-trace logging. Last updated 2026-06-09.

---

## TL;DR

| | Verdict |
|---|---|
| **Recommendation** | Move model lifecycle to **llama-swap**; keep a thin (~150–250 line) logging shim in front of it for SSE chat capture; leave Caddy as ingress. |
| **Why** | ~80% of the custom code (lifecycle, dead-worker recovery, preset generation, two-proxy split) duplicates llama-swap and is the source of the recurring dead-worker pain. The recovery code only exists to paper over router mode's unreliable parent↔child health sync. |
| **Biggest win** | On Windows your cancel crash is a *hard process exit* (`0xC0000409`). llama-swap re-launches a dead backend on the next request automatically — the entire `recover_worker` saga becomes unnecessary. |
| **What it does NOT fix** | The underlying `llama-server` cancel-during-decode bug ([#20921](https://github.com/ggml-org/llama.cpp/issues/20921)) is in the binary and survives any orchestrator. But recovery from it gets trivial. |
| **Main cost** | Chat/SSE logging isn't a llama-swap feature — that stays custom. Plus the one-time migration + re-validation effort. |
| **Net LOC** | ~2,142 lines (`proxy.py` 1,605 + `embed_proxy.py` 537) → a YAML config + a few-hundred-line shim. |

---

## Current architecture

```
[outside] → Caddy (Linux, passthrough) → proxy.py     (:8001) → llama-server --router (:8002) → worker child (ephemeral port)
                                        → embed_proxy.py(:8003) → llama-server --router (:8004) → worker child
```

`proxy.py` / `embed_proxy.py` currently own, by hand:

- model load/unload via the router's `/models/load|unload`
- **idle unload after 10 min** (router mode has no time-based unload)
- `models-preset.ini` / `embed-preset.ini` generation from `MODELS × CTX_CHOICES`
- **dead-worker detection + `recover_worker`** (the unload/reload cycle, the queued-exit-kills-next-worker workaround, the `loading→unloaded` race fix)
- **post-cancel guard** (probe + recover before the next request)
- `/embedding/*` reverse-proxy, `/chat` prefix stripping, `Authorization` injection
- chat trace logging + SSE chunk capture

## Proposed architecture

```
[outside] → Caddy (Linux, passthrough) → [thin logging shim] → llama-swap → llama-server (per model, on demand)
```

- llama-swap owns lifecycle, TTL unload, per-model flags, swapping, health checks, restart-on-death.
- One llama-swap config replaces **both** proxies (chat models swap; embedder kept resident via a group).
- The shim keeps only what llama-swap can't do: SSE chat capture → `logs/chat-*.log`. Drop it entirely if you don't need the traces.

---

## Side-by-side

| Dimension | Router mode + custom proxy (current) | llama-swap |
|---|---|---|
| **Maturity** | Router mode is **experimental**; config format changes between builds. Custom proxy is bespoke. | Battle-tested, widely deployed, stable config schema. |
| **Process model** | Parent router + ephemeral **worker child** per model. *Does* isolate, but parent's view of child health is unreliable. | llama-swap manager + one independent `llama-server` process per model. Flat, well-supervised. |
| **Crash recovery** | Hand-rolled `recover_worker`: router reports `loaded` while child is dead → forced unload/reload, queued-exit workaround, multi-attempt poll. ~200 fragile lines. | **On-demand relaunch**: a dead backend is simply re-spawned on the next request, gated by health check. No custom code. |
| **Hang/wedge recovery** | Post-cancel guard probes + recovers. | Health check gates readiness on (re)start; a *mid-life* hang is not auto-detected (same gap) — but on Windows your cancels hard-crash, so this rarely applies. |
| **Idle unload** | Custom 10-min watchdog (router mode has no time-based unload). | Native per-model `ttl` (+ `globalTTL`). Config, not code. |
| **Count-based eviction** | `--models-max 1` (LRU when >max). | Swaps by default; `groups`/`matrix` give explicit control. |
| **Keep embedder resident** | Achieved by running a **second** router + second proxy. | A **group** keeps the embedder loaded while chat models swap — collapses both stacks into one. |
| **Per-model flags (MTP etc.)** | Generated INI presets. | Arbitrary `cmd:` per model — MTP flags (`--spec-type draft-mtp`, …) drop straight in. |
| **Cancellation crash ([#20921](https://github.com/ggml-org/llama.cpp/issues/20921))** | Happens; mitigated by guard + recovery. | **Still happens** (binary bug), but recovery is automatic on next request. |
| **OpenAI/Anthropic API** | Passthrough you maintain. | Native: chat, embeddings, audio, images, token count. |
| **Streaming (SSE)** | Custom loop + keep-alives. | Native SSE handling. |
| **Chat/SSE content logging** | ✅ Custom (`logs/chat-*.log`). | ❌ Not a feature — UI log streaming + Prometheus `/metrics` only. **This is what keeps a shim.** |
| **Routing / auth / path rewrite** | In `proxy.py`. | `stripParams`/`setParams`, aliases, model overrides; or leave in Caddy. |
| **Observability** | Log files. | Web UI w/ live logs + request inspection, `/metrics`, `/logs/stream`. |
| **Windows support** | Works (this is the prod box). | Official Windows releases. |
| **Operational complexity** | High: 2 proxies, 2 routers, generated INIs, bespoke recovery. | Low: one binary, one YAML. |
| **Code you maintain** | ~2,142 lines of Python. | ~a few hundred (shim only) + YAML. |

---

## Deep dive on the dimensions that actually bite you

### 1. Dead-worker recovery — the main reason to switch

Almost all the hairy code in `proxy.py` exists to work around **router mode's parent↔child health-sync being unreliable**: the router keeps reporting a model as `loaded` after its worker child has died, so the proxy can't trust status and must force an unload/reload, then fight the router's queued-exit-signal that kills the *next* freshly-spawned worker. That's the `recover_worker` machinery, the `1.5s` stability check, and the `loading→unloaded` race I just fixed.

llama-swap sidesteps the entire class: it supervises each `llama-server` as a normal child process and **launches one on demand**. If a process has exited (your `0xC0000409` case), the next request just starts a fresh one and waits for its health check. There's nothing to "recover" — there's no second tier reporting stale state.

> Honest caveat: llama-swap does **not** actively monitor a *living-but-hung* backend mid-request and kill it. That wedge case is unsolved on both sides. It matters less here because your Windows cancels produce a hard crash (clean exit), not a Linux-style freeze.

### 2. Cancellation

Neither tool fixes [#20921](https://github.com/ggml-org/llama.cpp/issues/20921) — the cancel→next-request desync is inside `llama-server`. The difference is blast radius and cleanup:

- **Today:** cancel → worker child crashes → router still says `loaded` → proxy must detect + force-cycle + guard the next request.
- **llama-swap:** cancel → process exits → next request spawns a clean process. The guard logic becomes optional/much smaller.

### 3. Embedder coexistence — a structural simplification

You run two whole stacks today *only* so the embedder can stay resident alongside a swapping chat model. llama-swap `groups` express exactly this: put the embedder in a group that stays loaded; let chat models swap within their group. **`embed_proxy.py` (537 lines) disappears.**

### 4. Per-model config & MTP

Today: `MODELS × CTX_CHOICES` → generated `models-preset.ini`. In llama-swap each variant is an explicit model entry with its own `cmd`. More verbose, but explicit and diffable, and arbitrary flags (MTP, quant KV, flash-attn) are first-class. No INI generator to maintain.

### 5. Chat logging — the one real gap

llama-swap intentionally proxies transparently and does **not** parse/store response content. Your `logs/chat-*.log` + SSE chunk capture has no equivalent. Options:

1. **Thin shim** (~150–250 lines) between Caddy and llama-swap that tees SSE → chat log. Keeps the feature; sheds everything else.
2. **Drop it** and rely on llama-swap's UI request inspection + `/metrics` if full traces aren't essential.
3. **Caddy access logs** for request metadata (no bodies).

### 6. Maintenance & failure surface

Current: 2 Python services + 2 routers + generated INIs + bespoke recovery + a cancel guard — every layer a place for the kind of bugs filling these threads. Target: 1 supervised binary + 1 YAML (+ optional shim). Fewer moving parts is the actual goal you flagged.

---

## What you would lose / risks

- **Chat trace logging** unless you keep the shim (see §5).
- **Migration effort + re-validation**: rebuild the model matrix as llama-swap config, confirm MTP works under llama-swap, re-test the embedder-resident group, re-point Caddy.
- **A new config dialect** to learn (groups/macros/ttl), though far smaller than the current code.
- **The cancel crash itself remains** — you're buying simpler *recovery*, not a cure. The real cure is an upstream `llama-server` fix or dropping the amplifiers (reasoning + MTP), which you've chosen to keep.
- **Config verbosity**: explicit per-ctx entries instead of generated presets (mitigable with macros).

## What you keep regardless

- Caddy as LAN ingress (unchanged).
- The RTX 3090 Ti pinning / `CUDA_VISIBLE_DEVICES` (becomes an env in the model `cmd`).
- All your `llama-server` flags — they move verbatim into `cmd:` lines.

---

## Recommendation

1. **Spike, don't cut over.** Write a llama-swap `config.yaml` mirroring today's models (chat variants + MTP + resident embedder group) and run it on alternate ports while the current stack keeps serving.
2. **A/B under the failure mode that matters**: run a cancel-storm (subagent spawn / switch-back / mid-prefill cancel on a large context) against both and compare recovery time and whether requests after a cancel succeed.
3. **Decide on logging**: if traces matter, port only the SSE-capture into a thin shim; otherwise drop it.
4. If the spike wins, cut Caddy over to the shim/llama-swap and retire `proxy.py` + `embed_proxy.py`.

This keeps the current setup as the safety net and tests the exact workload that's been crashing, instead of trusting analysis alone.

---

## Sources

- [llama-swap (GitHub)](https://github.com/mostlygeek/llama-swap) · [config docs](https://github.com/mostlygeek/llama-swap/blob/main/docs/configuration.md) · [config schema](https://github.com/mostlygeek/llama-swap/blob/main/config-schema.json)
- [Router mode vs llama-swap tradeoffs — Glukhov](https://www.glukhov.org/llm-hosting/llama-cpp/llama-server-router-mode/)
- [llama-swap quickstart — Glukhov](https://www.glukhov.org/llm-hosting/llama-swap/)
- [Model management in llama.cpp — HF blog](https://huggingface.co/blog/ggml-org/model-management-in-llamacpp)
- Relevant upstream bugs: [#20921 cancel→crash](https://github.com/ggml-org/llama.cpp/issues/20921) · [#18912 router zombie child](https://github.com/ggml-org/llama.cpp/issues/18912) · [#18170 router image task](https://github.com/ggml-org/llama.cpp/issues/18170) · [#20137 models-max TOCTOU](https://github.com/ggml-org/llama.cpp/issues/20137) · [#10509 cancel during prefill](https://github.com/ggml-org/llama.cpp/issues/10509)
</content>
</invoke>
