# Chat-template perf regression — A/B test plan

## Symptom

Same hardware (3090 Ti), same model (`qwen3.6-35b-q4-128k`), same conversation depth (~80k tokens, ~64% of 128k preset).

| State | TG at low ctx | TG at ~50–65% fill |
|---|---|---|
| Before template change | ~120 t/s | ~90 t/s |
| After template change   | ~120 t/s | **~45 t/s** |

Token generation halved at deep context after enabling the froggeric chat template.

## Change under test

Added to `proxy.py:133-136` (chat router only, embed stack untouched):

```python
"--jinja",
"--chat-template-file", str(ROOT / "chat_template.jinja"),
"--chat-template-kwargs", '{"preserve_thinking":true}',
```

`chat_template.jinja` is the 264-line froggeric Qwen-Fixed-Chat-Templates file from HuggingFace.

## Hypotheses (ranked)

1. **`--jinja` engine cost.** Activating minijinja changes per-request and possibly per-token paths in llama-server b9094. Most likely culprit.
2. **Custom 264-line template file.** Heavier to render than the built-in template; prefill cost grows with messages-array size.
3. **`preserve_thinking:true` cache-layout effect.** Lower probability — affects what's cached, not throughput-per-token.

## Test protocol

For each variant:

1. Stop the chat watchdog (`Ctrl-C` in its window).
2. Edit `proxy.py` `server_command` to match the variant below.
3. Restart watchdog → router boots fresh.
4. **Cold measurement:** send a single ~100-token prompt, record TG t/s.
5. **Warm measurement:** replay (or continue) the same multi-turn conversation that previously hit ~82k tokens. Record TG t/s at the same depth.
6. Record in the results table below.

Use the same client (pi) and the same conversation history file each run so context is identical across variants.

### Variants

| ID | `--jinja` | `--chat-template-file` | `--chat-template-kwargs` | Purpose |
|---|---|---|---|---|
| **A — baseline (full revert)** | off | off | off | Confirm we can reproduce the old 120→90 curve |
| **B — jinja only**             | on  | off (built-in template) | off | Isolate `--jinja` engine cost |
| **C — jinja + custom template** | on | `chat_template.jinja` | off | Add the custom template, no kwargs |
| **D — full new config**        | on  | `chat_template.jinja` | `{"preserve_thinking":true}` | Current state — should reproduce ~45 t/s |

## Results

Model under test: `qwen3.6-35b-q4-128k` for all variants.

| Variant | PP (t/s) | TG (t/s) | Depth | Notes |
|---|---|---|---|---|
| A — baseline                  | ~358 (full prefill), ~2486 (delta) | 95 → 108 | ~22k | Healthy. SWA cache invalidation forced a full 20k prefill on first turn (separate issue). |
| A — baseline                  | ~1701 (delta) | **70** | ~100k | Healthy deep-context curve: ~35% TG drop from 22k → 100k. |
| B — jinja only                | ~2274 (full 99k cold prefill) | **65** | ~99k | ~7% slower than baseline at same depth. `--jinja` alone is near-neutral. |
| C — jinja + custom template   | ~2251 (full 103k cold prefill) | **64** | ~103k | Within margin of B. Custom template file itself adds ~0% cost. |
| D — full new config (prior)   | ~45 | **45** | ~82k | Original reading that triggered the investigation. **Not reproducible** in clean retest — see next row. |
| D — full new config (retest)  | ~2240 (full 103k cold prefill), ~843 (warm 435-tok delta) | **65** | ~103k | Identical to B and C. The 82k → 45 result was a one-off (likely thermal / background process / KV churn from a different conversation shape). Template is **innocent**. |

## Decision matrix

- **A ≈ D** → not the template; something else regressed. Look at binary version, KV cache eviction logs, GPU memory pressure.
- **A ≫ B** → `--jinja` itself is the engine cost. Either drop `--jinja` (lose `chat_template_kwargs` support) or bump the llama.cpp binary and retest.
- **B ≈ A, C ≪ B** → the 264-line custom template is the cost. Try a trimmed version or stay on the built-in template.
- **C ≈ B, D ≪ C** → `preserve_thinking:true` is the cost. Flip default to `false` and use per-request override.

## Notes / observations

(fill in during testing)

-

## Rollback

To restore the pre-test state at any point, delete the three added lines in `proxy.py:133-135` so `server_command` returns to:

```python
"--models-preset", str(PRESET_PATH),
"--models-max", "1",
"--no-models-autoload",
"--host", self.server_host,
"--port", str(self.server_port),
"--api-key", self.api_key,
```

`chat_template.jinja` can stay on disk; it's inert without the flags.
