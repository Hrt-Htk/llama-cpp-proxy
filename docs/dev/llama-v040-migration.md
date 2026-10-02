# llama.cpp v0.4.0 Migration & Side-by-Side Verification

Gate the binary upgrade **before** touching production. Run the candidate on
isolated **ports**, but remember: it shares GPU0 with production, so the two
stacks run **sequentially on the GPU, parallel only in ports**. Verify old vs
new one after another against the same harness and same presets, and only flip
production when the candidate is green on every check.

## 0. Context

- Current binary: `llama.cpp_latest/llama-server.exe` = build **v9670** (commit
  `02810c7aa`, 2026-06-16). Known-good baseline.  
  **STATUS: FLIPPED 2026-09-09 — production now runs v0.4.0 (b10816); v9670
  preserved as `llama.cpp_latest_v9670/`.**
- Candidate: **v0.4.0** (2026-09-04). ~3 months of changes.
- `reasoning_effort` → template forwarding landed 2026-08-14 (PR #26941) — after
  v9670, in v0.4.0. Primary reason to upgrade.
- `--load-mode` replaces deprecated `--mmap/--no-mmap` (PR #26934, 2026-08-15).

**Two binaries, one per router, and they are NOT independent:** both `proxy.py`
and `embed_proxy.py` point at the *same* `llama.cpp_latest/llama-server.exe`
(`proxy_config.py:19`, `embed_proxy.py:27`). The flip upgrades the embedder too —
so embed acceptance is **mandatory**, not optional.

## 1. What we get from the upgrade

Verified against the actual binaries (b10816 vs v9670 `--help`):

**Genuinely new in v0.4.0 (absent from v9670):**
- `--reasoning-effort LEVEL` native flag + top-level `reasoning_effort` forwarding
  to the chat template (PR #26941).
- `--load-mode MODE` (auto/mmap/mmap+mlock) — clean replacement for the now-
  deprecated `--mmap/--no-mmap`.

**Already present in v9670 (NOT upgrade benefits — earlier draft was wrong):**
- `--reasoning-budget N` and `--reasoning-budget-message MESSAGE` both already
  exist in v9670. Only the `reasoning_effort` plumbing is new.

Must be re-verified (not assumed identical): vision/mmproj, MTP decode,
router mode, slot/checkpoint behavior, KV/context budget (re-confirm 224k fits).

## 2. Binary layout

- Create `llama.cpp_v040/`, drop in v0.4.0 `llama-server.exe`.
- Back up the current binary: keep `llama.cpp_latest_b9209_bak`, add
  `llama.cpp_latest_v9670_bak`.
- **Never modify `llama.cpp_latest/` during testing** — production runs on it.

## 3. Hard preconditions (VRAM exclusivity)

Both routers are pinned to **GPU0** (`router_manager.py:83`,
`CUDA_VISIBLE_DEVICES=0`) and each model is ~23 GB on a 24,564 MiB card.
`--models-max 1` bounds one router only; it does not arbitrate across the two
independent routers. Therefore:

- **Before loading the candidate model, guarantee production is unloaded**:
  stop the production proxy, or force `/models/unload`, then confirm GPU0 is
  near-idle via `nvidia-smi` (`memory.used` well under a model footprint).
- `gpu_used_mib` (`model_acceptance.py:93`) reads **total GPU0 `memory.used`**,
  not per-process. Any concurrent production model pollutes the VRAM/fit/vision
  checks. The 224k-fit measurement is only valid with production fully unloaded.
- Run baseline first, unload, then candidate. Never both resident.

## 4. Config threading (two-part change, not one flag)

The proxy hardcodes `SERVER_EXE` and `PRESET_PATH` as module globals
(`proxy_config.py:19-20`) and `server_command` reads the globals, not `self`
(`:91-109`). To run an isolated candidate stack, thread config through:

1. Add `server_exe` and `preset_path` fields to `ProxyConfig`.
2. Add `--server-exe` and `--preset-path` to the parser (`proxy_config.py:192-211`).
3. Rewrite `server_command` to use `self.server_exe` / `self.preset_path`.
4. Make `write_preset` / `build_config` (`:176-241`) write to the configured
   preset path. Defaults preserve production behavior.

**The candidate MUST use its own preset file** (e.g.
`models-preset.candidate.ini`). There is exactly one chat preset path today and
`build_config` overwrites it on every startup — launching the candidate
otherwise clobbers production's live `models-preset.ini`, and any candidate-only
flag change (`load-mode`, etc.) would leak into production on its next restart.
Embed is safe (separate `embed-preset.ini`, `embed_proxy.py:28`).

## 5. Isolated topology

| Stack | Production (v9670) | Candidate (v0.4.0) |
|---|---|---|
| Chat proxy | :8001 | **:8011** |
| Chat router | :8002 | **:8012** |
| Embed proxy | :8003 | :8013 |
| Embed router | :8004 | :8014 |

Launch candidate chat stack:

```powershell
.venv\Scripts\python.exe proxy.py `
  --server-exe "H:\llama.cpp\llama.cpp_v040\llama-server.exe" `
  --preset-path "H:\llama.cpp\models-preset.candidate.ini" `
  --proxy-port 8011 --server-port 8012
```

Launch candidate embed stack the same way via `embed_proxy.py` on :8013/:8014.

**Replicate the real production environment, not the interactive shell.**
The `GGML_CUDA_CUBLAS_COMPUTE_TYPE=fp32` vision workaround is **not set anywhere
in the repo** (grep: only in this doc). Locate where it actually lives (OS/user
env), and pass the same `CUDA_VISIBLE_DEVICES=0`, `LLAMA_API_KEY`, and any vision
env vars to the candidate launch so the two stacks compare on equal footing.

## 6. Test matrix (run against BOTH stacks, one after another)

Acceptance harness takes `--proxy`/`--router` (`model_acceptance.py:461-462`).
Run production first (baseline), then candidate, same model, same GPU state.

```powershell
# Baseline (v9670, production ports)
.venv\Scripts\python.exe tests\model_acceptance.py --model qwen3.8-27b-q4-mtp-224k
# Candidate (v0.4.0, test ports)
.venv\Scripts\python.exe tests\model_acceptance.py `
  --model qwen3.8-27b-q4-mtp-224k --proxy http://localhost:8011 --router http://127.0.0.1:8012
```

Repeat for `nail-35b-a3b-q4-256k` and for the embed stack (embeddings + rerank).

**Tag the reports per stack** — `write_report` names files
`<model>-<stamp>.md` (`:421`) with no stack identifier, so baseline and candidate
for the same model differ only by minute. Run with a `--label baseline|candidate`
(plumb through to `write_report`) or redirect `--` into separate output dirs.

### Harness-covered (auto)
Fits / VRAM / loads / talks / thinks / tools / long-context / vision / full-ctx.

### Manual (NOT harness-covered — observe and record)
These are not in `model_acceptance.py`; record them by hand:

| Check | How | Pass = |
|---|---|---|
| MTP engaged | server log `draft acceptance` | ~0.7+; record decode t/s |
| Decode speed | server log `eval time ... t/s` | within ~10% of baseline |
| Long-context checkpointing | 224k prompt, checkpoint reuse | no crash, `truncated = 0` |
| reasoning_effort | send `low` then `high`, observe thinking length | value now takes effect; length differs |
| reasoning-budget | `--reasoning-budget 8192` + message | fires only past 8192 |
| load-mode | `load-mode = none` in candidate preset | boots, 224k fits, no warning |

`--min-decode-tps` is a floor (default 15), not a delta — it will not catch a
60→30 t/s MTP regression. Perf comparison is a manual two-report diff.

## 7. Verify candidate flags BEFORE editing presets

Run `llama.cpp_v040\llama-server.exe --help` (and `--version`) first:
- Confirm the exact `--load-mode` INI key spelling and enum values
  (`none` vs `mmap`/`load`/…).
- Confirm whether `no-mmap` is **removed** (boot-blocking → must migrate) or
  merely **deprecated** (warning; migrate but not blocking).
- Confirm `--reasoning-budget-message` and `--reasoning-effort` exist.

Only then apply, in the **candidate** preset/config (never production):

**Mandatory for the upgrade** (required for the candidate to boot/run cleanly):
- `no-mmap = 1` → `load-mode = none`.
- `--jinja` stays (it is default now; harmless).

**In scope for this upgrade — make `reasoning_effort` actually work.**
The upgrade enables forwarding; the template still must consume the value:
- Add the `reasoning_effort` branch to both `.jinja` templates
  (`chat_template.jinja`, `chat_template_sharp.jinja`). Map `xhigh`/`high`→
  xhigh-instruction, `medium`→(no extra instruction), `low`→brief-thinking
  instruction, `none`→`enable_thinking=false`. Qwen defines only
  xhigh/medium/low; treat `high`/`max` as aliases of xhigh.
- The client already sends `reasoning_effort: high` — after this change it
  takes effect (agent turns run at xhigh).
- Verify in the test matrix (§6): send `low` and `high` on the candidate and
  confirm thinking length differs.

**Out of scope for now — the user has NOT asked for a thinking limit.**
Leave `--reasoning-budget` at its current value (unlimited). Do NOT set
`--reasoning-budget 8192` / `--reasoning-budget-message` unless asked.

## 8. Tests

The `--server-exe`/`--preset-path` threading touches `proxy_config`/`ProxyConfig`.
Run the repo test suite (stdlib `unittest`, `.venv` python) and add coverage for
the new config plumbing **before** the flip.

## 9. Success criteria / go / no-go

### A/B test results (2026-09-09, run on this machine)

Same model, same GPU0, same harness, sequential. Baseline = v9670 (prod ports),
candidate = b10816 (v0.4.0, :8012).

**Qwen3.8-27B-UD-Q4_K_XL (qwen3.8-27b-q4-mtp-224k)**

| Check | Baseline v9670 | Candidate v0.4.0 | Verdict |
|---|---|---|---|
| Load | 23590 MiB | 23560–23582 MiB | ✓ (30 MiB less) |
| basic / tools / think | pass | pass | ✓ |
| streaming wall tps | 81.3–83.7 | 86.4–87.0 | ✓ +5% |
| decode tg | 77.7–81.5 | 83.9–84.5 | ✓ +8% |
| prefill pp (48k) | 1178 tok/s | 1199 tok/s | ✓ |
| long-context 48k needle | pass (44.4s) | pass (41.7s) | ✓ |
| vision | pass (2.2–5.5s) | pass (1.7–2.0s) | ✓ faster |
| 224k fit | loads, KV=229376 | loads, KV=229376 | ✓ both fit |

**Nail-Qwen3.6-35B-A3B-UD-Q4_K_XL (nail-35b-a3b-q4-256k)**

| Check | Baseline | Candidate | Verdict |
|---|---|---|---|
| decode tg | 122.3 tok/s | **147.9 tok/s** | ✓ **+21%** |
| prefill pp (48k) | 3152.7 | 3284.0 | ✓ |
| vision | pass (2.1s) | pass (1.1s) | ✓ |
| long-context 48k | pass | pass | ✓ |

**Embed (Qwen3-Embedding-4B-Q8_0)**: embeddings OK on both, dims 2560,
vectors near-identical. Rerank returns 501 `Start it with --reranking` on BOTH
(pre-existing gap — rerank was never enabled; not a regression).

**New features verified on candidate:** `--reasoning-effort` and `--load-mode`
present in `--help`; `--mmap/--no-mmap` deprecated (warning only, still works).

**Environment trap (this cost one failed run):** the candidate router must be
launched with `CUDA_VISIBLE_DEVICES=0` exactly like production
(`router_manager.py:83`). Without it, v0.4.0 sees both GPUs and OOMs the MTP
draft on the 2070.

**Harness limitation (not a regression):** `--full-ctx` needle test sends
~339k tokens (the 1.3 tok/word estimate overshoots this model's tokenizer) —
>229k ctx, so it 400s on BOTH stacks identically. The 224k-fit proof is the
load (KV pre-allocated at 229,376).

Verdict: **candidate green on every real check; no regressions; Nail decode
+21%, 27B decode +8%.** Ready to flip.

Go: candidate green on every row, no new warnings, 224k fits, vision works,
MTP acceptance ≈ baseline, embed acceptance green, `reasoning_effort` behaves as
specified (high vs low changes thinking; no reasoning-budget change in scope).
Then flip and re-run the harness once on
production ports before unwiring the test stack.

No-go on: vision regression, 224k no longer fits, MTP/decode regression >10%,
router load/unload breakage, embed regression, new deprecation warnings.

## 10. Flip and rollback (folder rename, never in-place exe overwrite)

Windows locks a running `llama-server.exe` — you cannot overwrite it in place.
The flip and any rollback are **folder renames with all watchdogs killed**:

1. Kill both watchdogs; confirm **no** `llama-server.exe` process remains
   (`Get-Process llama-server` empty). Orphaned elevated workers may require a
   reboot.
2. Rename `llama.cpp_latest` → `llama.cpp_latest_v9670` (rollback source).
3. Rename/copy `llama.cpp_v040` → `llama.cpp_latest`.
4. Relaunch both watchdogs; smoke-test chat + embed.

Rollback: reverse steps 1–4. Presets regenerate on startup; no other state
persists in the binary.

## 11. Other risks

- **Midnight restart during the test window.** `restart-watchdog.ps1` runs on a
  Task Scheduler cadence. If the window straddles it, production regenerates its
  preset and restarts mid-test — interacting badly with the shared preset (now
  avoided by the candidate's own `--preset-path`, but still disruptive). Finish
  the test window within one day or disable the scheduled restart.

## 12. Follow-ups after green

- Commit the config threading, `load-mode`, and template `reasoning_effort`
  branch (Conventional Commits, one issue per PR).
- Update `CLAUDE.md` binary note (b9209+ → v0.4.0) and this doc's §0.
- Rename `llama.cpp_latest/` to reflect v0.4.0 (no longer "latest").
