# MTP Rollout Plan

Goal: offer Qwen MTP (multi-token prediction / built-in speculative decoding) for the 27B on the chat router, for a ~1.5–2x tok/s win on the 3090 Ti.

**APPROACH CHANGED (2026-05-21 session 2):** we are **NOT swapping** the 27B GGUF. Instead the MTP variant is added as a **separate, additional model** (`qwen3.6-27b-q4-mtp`, label "Qwen3.6-27B Q4 MTP", weights `Qwen3.6-27B-UD-Q4_K_XL.mtp.gguf`). All existing models — both 35Bs and the original non-MTP 27B (`qwen3.6-27b-q4`) — are left completely untouched. `--models-max 1` means only one loads at a time, so the extra entry costs no VRAM unless selected. No `.pre-mtp.gguf` backup is needed; the original file stays in place under its original name.

---

## SESSION HAND-OFF (2026-05-21)

Where things stand after the last session — pick up from here:

1. **Phases 1 & 2 are DONE.** The binary is already b9209 and supports MTP. PR #22673 merged 2026-05-16. The correct `--spec-type` value is **`draft-mtp`** (not `mtp`).
2. **Download being resumed (2026-05-21 session 2).** `aria2c` had stalled at 16.68 GB of 17.9 GB (process no longer running, `.mtp.gguf.aria2` control file still present). Resumed with `-c`. Resume command if interrupted again:
   ```powershell
   cd H:\llama.cpp\models
   aria2c -x16 -s16 -c -o "Qwen3.6-27B-UD-Q4_K_XL.mtp.gguf" `
     "https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF/resolve/main/Qwen3.6-27B-UD-Q4_K_XL.gguf"
   ```
   Done when the file is 17.9 GB and the `.mtp.gguf.aria2` control file is gone.
3. **Phase 4 code edits APPLIED (2026-05-21 session 2).** `proxy.py` now has `spec_mtp: bool = False` on `ModelChoice`, `spec_mtp=True` on the 27B entry, and `_model_preset_section` appends the three `spec-*` INI lines when set. Syntax verified. The 35B sections render byte-for-byte unchanged. Not yet exercised against a live router (pending download + swap).

Remaining steps, in order:
- **Swap (do while 27B is NOT loaded):** rename current `Qwen3.6-27B-UD-Q4_K_XL.gguf` → `Qwen3.6-27B-UD-Q4_K_XL.pre-mtp.gguf`, then rename `Qwen3.6-27B-UD-Q4_K_XL.mtp.gguf` → `Qwen3.6-27B-UD-Q4_K_XL.gguf`.
- ~~Wire the proxy (Phase 4 below) — 27B only.~~ DONE.
- **Validate (Phase 5 below).**

---

## Current state

- Build: `llama.cpp_latest/llama-server.exe` @ **b9209** — supports MTP.
- `--spec-type` choices in this build: `none | draft-simple | draft-eagle3 | draft-mtp | ngram-simple | ngram-map-k | ngram-map-k4v | ngram-mod | ngram-cache`. PR [ggml-org/llama.cpp#22673](https://github.com/ggml-org/llama.cpp/pull/22673) **merged 2026-05-16 and is in this build.**
- Confirmed flags: `--spec-type draft-mtp`, `--spec-draft-n-max N` (default 16), `--spec-draft-p-min P` (**default 0.75**).
- Models currently served (from `proxy.py:50`):
  - `Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf` — stays non-MTP, untouched.
  - `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` — stays non-MTP, untouched.
  - `Qwen3.6-27B-UD-Q4_K_XL.gguf` — **target for MTP.**
- The 27B has a direct MTP-variant equivalent at `unsloth/Qwen3.6-27B-MTP-GGUF` with an **identical filename** (17.9 GB vs the current 17 GB — the extra ~0.9 GB is the MTP head tensors).

## What MTP buys us

- ~1.5–2x generation tok/s (unsloth's own claim, matches the 4090 reference figures the perf doc references).
- Especially impactful for the 35B-A3B MoE — only 3B active params per token, so it's heavily bandwidth-bound and benefits most from spec decode.
- No separate draft model in VRAM — the MTP heads are baked into the same GGUF.

---

## Phase 1 — Verify MTP availability upstream ✅ DONE

PR [#22673](https://github.com/ggml-org/llama.cpp/pull/22673) **merged 2026-05-16** into master ("llama + spec: MTP Support"). MTP heads load from the same GGUF; activated via `--spec-type draft-mtp`.

## Phase 2 — Upgrade `llama.cpp_latest` ✅ DONE

Already on **b9209** (the upgrade documented in `update_llama.cpp.md` happened). Confirmed:
```powershell
.\llama.cpp_latest\llama-server.exe --help | Select-String "spec-type"
# → none,draft-simple,draft-eagle3,draft-mtp,ngram-simple,...
```
`draft-mtp` is listed → MTP support present. No `SERVER_EXE` change needed.

## Phase 3 — Download MTP GGUF (27B only)

Pull **one** file into `models/` (filename identical to current → no `MODELS` path edit needed):

| Current file | Source repo | Size |
|---|---|---|
| `Qwen3.6-27B-UD-Q4_K_XL.gguf` | [unsloth/Qwen3.6-27B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF) | 17.9 GB |

The two 35B models are **not** downloaded or changed.

Downloaded to a temp name (`Qwen3.6-27B-UD-Q4_K_XL.mtp.gguf`) so the live 27B keeps serving during the ~17.9 GB pull. Then swap:
1. `Qwen3.6-27B-UD-Q4_K_XL.gguf` → `Qwen3.6-27B-UD-Q4_K_XL.pre-mtp.gguf` (backup)
2. `Qwen3.6-27B-UD-Q4_K_XL.mtp.gguf` → `Qwen3.6-27B-UD-Q4_K_XL.gguf`

Do the swap while the 27B is **not loaded** (Windows locks loaded GGUFs).

**Exit criteria:** MTP 27B GGUF in place; old GGUF preserved as `.pre-mtp.gguf`. Rollback is a single rename.

## Phase 4 — Wire MTP into the preset generator (27B only)

In `proxy.py`:
1. Add an optional field to the `ModelChoice` dataclass (~line 39): `spec_mtp: bool = False`.
2. Set `spec_mtp=True` on the **27B** entry only (`base_id "qwen3.6-27b-q4"`, ~line 63). Leave both 35B entries at the default `False`.
3. In `_model_preset_section` (~line 565), when `model.spec_mtp` is true, append (matching the existing aligned `key = value` style):
   ```python
   f"spec-type        = draft-mtp\n"
   f"spec-draft-n-max = 3\n"
   f"spec-draft-p-min = 0.0\n"
   ```

Notes:
- The INI uses hyphenated CLI-flag-style keys (`flash-attn`, `cache-type-k`, …), so `spec-type` / `spec-draft-n-max` / `spec-draft-p-min` are correct. Verify by checking the router log echoes `draft-mtp` after a 27B load.
- `--spec-draft-n-max 3` is the reference doc's value (build default 16 is too aggressive).
- `--spec-draft-p-min 0.0` matches the reference command (build default is **0.75**). Tighten later if low-confidence drafts hurt accept rate.
- Per-model via the flag — not a global enable — so the 35B presets stay byte-for-byte unchanged.

## Phase 5 — Validate

1. Restart proxy via the watchdog; load the **27B** model through `pi-llama-cpp` or a curl smoke test. Confirm the two 35B models still load and serve normally (regression check).
2. Watch `logs/llama-server-<date>.log` for per-request MTP accept-rate lines on the 27B.
3. Measure tok/s with `ping-both.py --local` (or a longer prompt) before declaring success.
4. Acceptance bar:
   - Accept rate ≥ 60%.
   - Tok/s up by ≥ 1.4x vs. the pre-MTP baseline noted in `3090ti_qwen3_27b_performance.md`.
5. If the 27B misbehaves with MTP, just flip `spec_mtp=False` on its entry and reload — no other model is affected.

Tunable knobs if results are mediocre:
- `--spec-draft-n-max` — try 2 (safer) or 4 (more aggressive). 3 is the reference.
- `--spec-draft-p-min` — raise (e.g. 0.4) to reject low-confidence drafts and lift accept rate at the cost of fewer accepted tokens.

## Phase 6 — Document

- Add a one-liner to `CLAUDE.md` "Key design points" noting MTP is enabled and which build introduced it.
- Update `3090ti_qwen3_27b_performance.md` with the post-MTP measured tok/s figures so the doc reflects the current setup rather than the pre-MTP baseline.

## Rollback

If MTP misbehaves on the 27B:
1. Set `spec_mtp=False` on the 27B `ModelChoice` entry (or revert the `proxy.py` edit) and reload.
2. Rename `Qwen3.6-27B-UD-Q4_K_XL.pre-mtp.gguf` back to `Qwen3.6-27B-UD-Q4_K_XL.gguf`.

The binary (b9209) stays put — it serves the non-MTP GGUFs fine. The two 35B models are never touched, so they need no rollback. No router-architecture or cloudflared changes involved.

## Status

**COMPLETE (2026-05-21 session 2).** All phases done via the add-a-model approach (no swap):
- Phase 3 download finished (`Qwen3.6-27B-UD-Q4_K_XL.mtp.gguf`, 17.9 GB).
- Phase 4: new `qwen3.6-27b-q4-mtp` model entry added to `proxy.py`; originals untouched.
- Phase 5 validated on the live router: `draft-mtp` confirmed in the load log; non-MTP 41.7 tok/s vs MTP 59.5 tok/s eval = **1.43x**; draft acceptance **62–77%**. Both acceptance bars cleared.
- Phase 6: CLAUDE.md + `3090ti_qwen3_27b_performance.md` updated.

Rollback if ever needed: remove the `qwen3.6-27b-q4-mtp` ModelChoice entry and restart the proxy. The `.mtp.gguf` file and the original non-MTP 27B both remain on disk regardless.
