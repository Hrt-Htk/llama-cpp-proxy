# New-Model Acceptance Tests

Gate a newly-added router model **before** wiring it into day-to-day use. Run this
for every model added to `proxy_config.py` (chat) — Nail, future Qwen drops, any
re-quant. If a check fails, the model is not ready; fix the preset/binary/template
and re-run.

The philosophy: **fits, loads, talks, thinks, calls tools, remembers long context,
sees images, and stays on the GPU** — proven empirically, not assumed.

---

## 0. Preconditions

- Weights + mmproj present under `models/` (and `models/_aux/`).
- Preset added to `MODELS` in `proxy_config.py`, proxy restarted so
  `models-preset.ini` is regenerated (see `docs/dev/workflow.md` for the safe
  restart). Confirm the new preset id shows in `/v1/models`.
- Chat proxy (`:8001`) and router (`:8002`) both up (`/health` → ok).
- `LLAMA_API_KEY` in env or `.env`.
- `.venv\Scripts\python.exe` available (system `python` fails — tzdata/ZoneInfo).

The router is pinned to the 3090 Ti (`CUDA_VISIBLE_DEVICES=0`,
`router_manager.py`). VRAM ceiling is **24,564 MiB**; treat **> ~24,000 MiB used**
as "no safe headroom" and **any spill to shared/system RAM** (WDDM) as a hard fail —
it silently tanks throughput.

---

## 1. Automated harness

```powershell
.venv\Scripts\python.exe tests\model_acceptance.py --model <preset-id>
# e.g.
.venv\Scripts\python.exe tests\model_acceptance.py --model nail-35b-a3b-q4-256k
```

Useful flags:

| Flag | Default | Meaning |
|------|---------|---------|
| `--model` | *(required)* | Preset id to test (must exist in `/v1/models`). |
| `--proxy` | `http://localhost:8001` | Client-facing proxy base (IPv6-friendly host). |
| `--router` | `http://127.0.0.1:8002` | Router base for load/unload/introspection. |
| `--vram-ceiling` | `24000` | MiB; fail if GPU0 `memory.used` exceeds this after load / during probe. |
| `--ctx-probe-tokens` | `32000` | Filler size for the needle-in-haystack long-context test. |
| `--full-ctx` | off | Push the needle probe to ~95% of the preset ctx (slow; real long-context proof). |
| `--image` | *(auto)* | Path to a test image; if omitted a known solid-color PNG is generated. |
| `--min-decode-tps` | `15` | Fail if decode throughput drops below this (regression floor). |
| `--skip` | *(none)* | Comma list of check ids to skip (e.g. `vision,long_context`). |

The harness prints PASS/FAIL per check and writes a dated report to
`tests/acceptance-results/<model>-<YYYY-MM-DD-HHMM>.md`. Non-zero exit if any
non-skipped check fails.

---

## 2. What each check proves

| id | Check | Method | Pass criteria |
|----|-------|--------|---------------|
| `registered` | Preset is exposed | GET router `/v1/models` | id present; `n_ctx` meta == expected preset ctx |
| `load` | Cold load works | POST router `/models/load`, poll status | reaches `loaded` within timeout; record load seconds |
| `vram_fit` | Fits on 3090 Ti, no spill | `nvidia-smi` GPU0 after load | `memory.used` ≤ `--vram-ceiling`; no shared-RAM spill |
| `basic` | Coherent completion | POST proxy `/v1/chat/completions` (non-stream) | deterministic answer correct (e.g. "2+2" → contains `4`) |
| `streaming` | SSE path healthy | POST proxy `stream=true` | ≥2 events + `[DONE]` |
| `think_integrity` | Reasoning clean (issue #8) | inspect streamed content | reasoning present; **no leaked `<think>`/`</think>` in visible content** |
| `tools` | Tool calling works | POST proxy with a `tools` schema | returns a `tool_calls[0]` with **valid JSON** arguments |
| `long_context` | Long-ctx retrieval + no OOM | needle-in-haystack prompt | needle returned verbatim; no OOM/spill; records **pp** from server timings |
| `vision` | Multimodal + CPU-mmproj | POST proxy image turn, then text follow-up | image described correctly; **turn-2 (text) works and is fast** (no re-encode); GPU stays ≤ ceiling |
| `perf` | Throughput | authoritative llama-server timings | records **pp** and **tg** tok/s; `tg` ≥ `--min-decode-tps` |

Speed is logged as **pp** (prompt processing) and **tg** (token generation) taken
from llama-server's own `timings` (`prompt_per_second` / `predicted_per_second`),
not wall-clock estimates. Both land in the report header and the `perf` row.

Notes on the tricky ones:

- **`think_integrity`** is the issue-#8 guard. The model must emit properly-closed
  `<think>` blocks and the proxy must keep them out of the user-visible `content`.
  A leak here means the SSEChunkLogger reroute regressed or the template's
  `preserve_thinking` handling is wrong for this model.
- **`vision`** proves the earlier design decision: with `no-mmproj-offload` the
  projector runs on CPU. The image is encoded **once**; the follow-up text turn must
  reuse cached KV (fast) and must **not** re-run the encoder. The check times both
  turns — turn-2 should be in the normal text-latency range, not image-encode range.
  It also asserts GPU `memory.used` never exceeds the ceiling during the image turn
  (the encode spike lives in RAM, not VRAM).
- **`long_context`** default is a moderate 32k filler for a fast gate. Run
  `--full-ctx` at least once per model to prove the preset's headline context
  (e.g. 256k) actually retrieves and doesn't OOM.

---

## 3. Manual / judgment checks (not automated)

Automation proves plumbing; a human confirms quality. Spend 5 minutes:

- [ ] **Terseness / system-prompt behavior** — froggeric models should answer
  without preamble. Ask something open-ended; confirm no "Certainly! Here's…".
- [ ] **Thinking length sane** — reasoning shouldn't run away (Nail/Dagger are
  tuned for compressed thinking). Watch a couple of hard questions.
- [ ] **Multi-turn coherence** — a 3–4 turn exchange; confirm it tracks context and
  tools across turns.
- [ ] **Tool-call ergonomics in the real client** — run one real task through pi
  (or your daily agent) and confirm tool calls parse and execute end-to-end.
- [ ] **Vision accuracy on a real screenshot** — describe an actual UI screenshot,
  not just the synthetic color swatch.

---

## 4. Per-model sign-off

Copy into the dated results file and fill in:

```
Model:            <preset-id>
Binary (b###):    <llama-server version>
Weights:          <gguf filename + bytes>
mmproj:           <file> (offload: cpu|gpu)
Preset ctx:       <n>
VRAM used @load:  <MiB> / 24564   (headroom: <MiB>)
pp tok/s:         <n>   tg tok/s: <n>
Automated:        <PASS/FAIL summary>
Manual review:    <initials + notes>
Verdict:          ACCEPTED / REJECTED for daily use
```

Keep every run under `tests/acceptance-results/` so regressions across binary
upgrades and re-quants are visible over time.
</content>
