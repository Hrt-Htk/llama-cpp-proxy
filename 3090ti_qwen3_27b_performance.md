# 3090 Ti vs 4090 — Running Qwen3.6 27B with llama.cpp

## Can a 3090 Ti match a 4090 here?

Pretty close, yeah. The 3090 Ti and 4090 have nearly identical memory bandwidth (~1008 GB/s), and LLM token generation is mostly memory-bandwidth-bound, so for a 27B IQ4_NL model the gap is smaller than raw specs suggest.

### Where you'll match the 4090 closely
Plain token generation without speculative decoding. Expect roughly **70–95 tok/s** on a 3090 Ti for the same model and context — same 24 GB VRAM, same bandwidth class.

### Where the 4090 pulls ahead
MTP / speculative decoding leans more on compute than vanilla decoding because the draft+verify cycle does extra work per accepted token. The 4090 has ~50% more CUDA cores and better tensor cores (Ada vs Ampere, plus FP8 support the 3090 Ti lacks). Expect the gap to widen here — probably **75–90 tok/s** with MTP rather than matching 110.

### 128k context
Fits on the 3090 Ti the same way it fits on the 4090 (both 24 GB). The `--cache-type-k q8_0 --cache-type-v q8_0` flags do heavy lifting to keep the KV cache manageable.

### Power / thermals
3090 Ti pulls ~450 W under load and runs hot. Sustained throughput can dip if cooling isn't great. The 4090 is more efficient at similar performance.

**Realistic expectation:** ~70–85% of the 110 tok/s figure with MTP, and basically parity for non-speculative inference.

---

## Why am I only seeing 40 tok/s on my 3090 Ti?

40 tok/s on a 3090 Ti with a Q4 27B model is on the low side — you should be closer to 60–80 tok/s. Likely culprits:

### 1. KV cache eating your bandwidth
At 128k context with full-precision (f16) KV cache, the cache becomes massive and every generated token streams through more memory. Use:

```
--cache-type-k q8_0 --cache-type-v q8_0
```

This roughly halves KV cache size and meaningfully boosts tok/s at long context.

### 2. Flash attention off
Make sure you have:

```
-fa on
```

Without flash attention, attention compute at long context gets expensive fast. Your llama.cpp build should support it on a 3090 Ti.

### 3. Not all layers on GPU
Use `-ngl all` (or `-ngl 99`) to offload everything. If even a few layers spill to CPU, throughput tanks. Check the startup log to confirm all layers loaded to GPU with VRAM headroom remaining.

### 4. Actual context length used vs allocated
This matters a lot:
- *Allocated* 128k context but generating at ~2k tokens of actual conversation → near peak speed.
- Actually deep into a 50k+ token conversation → decode slows significantly because attention scales with context length.

**Check your real prompt length when you measure 40 tok/s.**

### 5. Quant type
"Q4" is vague — Q4_K_M, Q4_K_S, IQ4_NL, Q4_0 all perform differently:
- **IQ-quants** are slightly slower to decode than K-quants on Ampere (more compute per weight).
- **Q4_K_M** is usually the sweet spot for speed on a 3090 Ti.

### 6. Batch / ubatch sizes
The reference command uses `-b 2048 -ub 512`. Defaults can be suboptimal for single-user inference — worth tuning.

### 7. No speculative decoding
The 110 tok/s figure comes from MTP (multi-token prediction / speculative decoding with a draft model). Without `--spec-type mtp` and a draft model, you're doing vanilla decoding and will never hit those numbers regardless of GPU.

---

## Reference command (from the original post)

```bash
exec ./build/bin/llama-server \
  -hf unsloth/Qwen3.6-27B-MTP-GGUF \
  -hff Qwen3.6-27B-IQ4_NL.gguf \
  --alias qwen3.6-27b-mtp-ud \
  -c 128000 \
  --host 0.0.0.0 --port "${PORT:-8080}" \
  -ngl all \
  -fa on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --spec-type mtp --spec-draft-n-max 3 --spec-draft-p-min 0.0 --no-mmproj \
  -b 2048 -ub 512 \
  -t "${THREADS:-24}" \
  -np 1 \
  --temp "${TEMP:-0.4}" --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 ...
```

---

## To diagnose your specific 40 tok/s issue, share:

1. The exact model file (e.g. `Qwen3.6-27B-Q4_K_M.gguf` vs `IQ4_NL.gguf`)
2. Your full launch command
3. Approximate prompt length when you measure 40 tok/s

## Measured on this rig (2026-05-21, post-MTP)

Actual rollout results on this 3090 Ti with `Qwen3.6-27B-UD-Q4_K_XL`, 32k context, a short prompt (~25 tokens), warm cache, and a 300-token generation:

- Non-MTP server eval: **41.7 tok/s**
- MTP (`draft-mtp`, `n-max 3`, `p-min 0.0`) server eval: **59.5 tok/s**
- Measured speedup: **1.43x**
- MTP draft acceptance rate: **62-77%** across two runs

This clears the rollout acceptance bar: acceptance rate **>=60%** and throughput **>=1.4x**.

## n-max sweep (2026-05-21)

| n-max | 2k | 32k | 64k | 96k |
|---|---|---|---|---|
| 2 | 65.1 \| 81% | 58.1 \| 84% | 46.2 \| 69% | 42.8 \| 74% |
| 3 | 63.0 \| 67% | 58.1 \| 72% | 47.8 \| 63% | 35.5 \| 46% |
| 4 | 50.8 \| 45% | 54.6 \| 60% | 40.2 \| 44% | 27.7 \| 28% |
| 6 | 38.0 \| 30% | 41.0 \| 38% | 44.8 \| 50% | 33.6 \| 37% |

With `spec-draft-p-min=0.0` the MTP head always drafts the full `n` tokens, so higher `n-max` drafts deeper, lower-confidence tokens that get rejected (error compounding) — costing verify slots without payoff. Result: `n-max=2` is fastest-or-tied at every depth and degrades most gracefully with context (at 96k, `n=2` gives 42.8 tok/s vs `n=3`'s 35.5, +21%). `n-max` 4 and 6 are strictly worse. Production value was therefore changed from 3 to 2. Caveat: absolute acceptance %s are inflated by synthetic filler text; the `n-max` ranking is the reliable signal.
