# Rollback + Concurrency Fix — Plan & Incident Notes

**Status:** planning only (no code changes yet)
**Date:** 2026-06-02
**Decision:** roll `proxy.py` back to commit `e703092` (drop both 2026-06-02 morning
commits), then solve the concurrency problem properly on the clean baseline.

---

## TL;DR

- The chat stack was throwing **500/502 storms**.
- Two commits landed the morning of 2026-06-02 to "fix" it
  (`04dc5e2` parallel-handling refactor, `39feb77` dead-child auto-recovery).
  They targeted the **wrong layer** and added complexity without fixing the
  cause.
- **Real cause:** *overlapping* (concurrent) requests for **different models**
  on a single-model router (`--models-max 1`) cause the in-flight request's
  model to be **evicted mid-stream → 500**. Sequential different-model requests
  are clean (this is why hand-driven use "ping-ponged seamlessly").
- **Trigger:** the `orchestrator` skill (created 2026-06-01 22:05) spawns pi
  subagents **in parallel**, and the `pi-subagent` skill picks **different
  models per task** — so requests started overlapping on mixed models for the
  first time. First 500 of the incident: 2026-06-01 22:33, ~28 min later.
- **Decision:** reset to `e703092` (too many problems in the current version to
  keep iterating on it), then design + build a **fair-queue / never-evict-busy**
  switch so parallel orchestration degrades into clean queuing instead of
  crashing.
- **Caveat:** rollback alone does **not** fix concurrency — `e703092` still has
  the latent mid-stream-eviction bug. It is a clean *reset point*, not the fix.

---

## Symptom

`POST /chat/chat/completions` returning `500`/`502` in bursts, sometimes looping.
Most visible when running the `/orchestrator` skill (multiple pi subagents).

5xx rate by day (chat completions only):

| Day | completions | 5xx | rate | notes |
|---|---|---|---|---|
| 2026-05-31 | 232 | 0 | 0% | clean; loaded both `q4-128k` and `mtp-128k` sequentially |
| 2026-06-01 | 411 | 27 | 6.6% | incident begins 22:33 (no recovery code yet) |
| 2026-06-02 | 1394 | 90 | 6.5% | recovery code active from 12:32 |

Within 2026-06-02: 9.4% before the 12:32 recovery deploy, 4.8% after — i.e. the
recovery commit did **not** make the rate worse (and the issue predates it by
~14 h).

---

## Investigation findings (evidence)

1. **The morning recovery (`39feb77`) was a no-op that lied.** Every "Recovery
   complete" fired in **~34 ms** (a real router restart + multi-GB reload takes
   tens of seconds). The router's own log recorded only **2** genuine
   `model loaded` events all day vs. dozens of claimed recoveries.

2. **Orphaned router processes.** Live snapshot showed three `llama-server.exe`:
   the healthy router on `:8002`, plus an **orphan** (parent proxy dead) still
   holding a model in VRAM. GPU 0 (3090 Ti) sat at **22.3 / 24.5 GB (91%)**.
   Orphans come from non-graceful proxy restarts — on Windows the child router
   is not auto-reaped when the parent dies, and neither the proxy nor
   `watchdog.ps1` kills strays on boot.

3. **Eviction thrash in the storm window.** 15:53–15:57 the router ping-ponged
   `qwen3.6-27b-q4-128k` ↔ `qwen3.6-27b-q4-mtp-128k`, reloading every ~11 s —
   two clients each pinned to a different model on a `--models-max 1` router.

4. **`_request_lock` never existed.** `git log -S'_request_lock' -- proxy.py`
   returns nothing. `04dc5e2`'s message claims it "replaced the stream-wide
   `_request_lock`" — that lock was never in the code. The "fix" was partly
   chasing a ghost.

5. **The pre-morning code (`e703092`) has the latent bug.** `ensure_loaded`
   holds `_load_lock` **only during the load**, then streams with no lock. A
   second request for a different model evicts the first **mid-stream** → 500.
   No drain. So rolling back reintroduces this; it is not, by itself, safe under
   overlap.

6. **Not the binary, not VRAM.** `llama-server.exe` unchanged since 2026-05-18.
   Only CUDA0 (24 GB) was ever enumerated on 05-31, 06-01, 06-02 — the 2070 was
   never used, so `e703092`'s `CUDA_VISIBLE_DEVICES=0` did not reduce headroom.
   Nothing in the server config regressed between the clean day and the broken
   day.

7. **The only thing that changed: request timing.** `orchestrator` skill mtime
   `2026-06-01 22:05`; first incident 500 at `2026-06-01 22:33`. The skill
   spawns pi subagents **in parallel** (`run_in_background`), and `pi-subagent`
   exposes 13 model presets and chooses per task. Parallel + mixed models =
   the first time requests *overlapped* on *different* models.

---

## Corrected root-cause model

The determinant is **temporal overlap × model identity**, not concurrency alone:

- **Sequential** requests, any models → switch happens between completed
  requests → **clean** (matches the remembered seamless ping-pong; 05-31 proves
  it with two models and zero errors).
- **Overlapping** requests on the **same** model → router's single slot queues
  them → **clean**.
- **Overlapping** requests on **different** models → loading model B evicts
  model A while A is still streaming → **A gets a 500** → client retries →
  collides again → **storm**.

The proxy never taught a model switch to *wait* for in-flight work. The morning
commits papered over the resulting crashes (recovery) and refactored adjacent
concurrency (probe cache + drain) without making cross-model switching safe.

---

## Confirmed reproduction (2026-06-02, on rolled-back `e703092`)

Harness: `test_concurrency.py`, run against an isolated stack
(`proxy --proxy-port 8011 --server-port 8012 --no-chat-log`). Two models:
`qwen3.6-27b-q4-32k` (A) and `qwen3.6-35b-q3-32k` (B).

| Test | Result | Verdict |
|---|---|---|
| Sequential different-model (A→B→A) | all **200** | clean — seamless switching confirmed |
| Concurrent different-model, round 0 | A=**500**, B=200 | bug reproduced |
| Concurrent different-model, round 1 | A=**500**, B=200 | bug reproduced again |
| Concurrent same-model | all **200** | clean — single slot queues fine |

The failing request dies with `proxy error: Failed to write connection` at
~13.7 s — exactly when the other request's model finishes loading and **evicts
the first model mid-stream**, killing its upstream connection. The request that
wins the load succeeds; the evicted one 500s. Same-model concurrency is clean
(no eviction).

**Conclusions, now empirical (not inferred):**
- Concurrent requests on *different* models → mid-stream eviction → 500
  (deterministic, 2/2).
- Rollback alone does **not** fix it — this *is* the rolled-back code.
- Sequential different-model is clean — the remembered seamless ping-pong is
  real; the orchestrator only made requests *overlap* for the first time.

This harness is the **acceptance test for the fix**: after Phase 2, every cell
above must be 200, including concurrent different-model (it must *queue*).

---

## Phase 1 — Rollback (reset point)

- Both morning commits touch **only `proxy.py`** and sit directly on top of
  `e703092`.
- `git checkout e703092 -- proxy.py` → commit. Clean removal of both.
- Live proxy keeps running old code until restarted; bounce it (and kill the
  orphan router + worker) so we boot from a clean process state. Server is
  personal — downtime is fine.
- **Known regression accepted:** clean-slate loses `04dc5e2`'s probe cache
  (pi's ~2,400 startup probes/day hit the router directly again) and reinstates
  the latent mid-stream-eviction bug. Both are addressed in Phase 2.

## Phase 2 — The fix (on the clean baseline)

Design goals: **single model at a time** (`--models-max 1` stays), **never
crash under parallel load**, parallel mixed-model traffic degrades into clean
**queuing** instead of mid-stream eviction.

### Root mechanism to eliminate

In `e703092`, `proxy_request` does `await manager.ensure_loaded(model)` and then
forwards the stream **with no lock and no in-flight tracking**. So a second
request's `ensure_loaded(other_model)` takes `_load_lock` and evicts the model
the first request is still streaming against → the first request's upstream
connection is killed → 500. The fix makes a switch **impossible while a request
is in flight**.

### Core change — gate every forward behind a `use_model()` context manager

Add to `ModelManager`:
- `_forwarding: int` — count of requests currently streaming.
- `_idle_forward: asyncio.Event` — *set* when `_forwarding == 0`.
- `_loaded_at: float` — monotonic time the current model finished loading.
- `_begin_forward()` / `_end_forward()` — inc/dec the counter and clear/set the
  event.

Replace the bare `ensure_loaded` + forward with:

```text
async def use_model(model):                 # async context manager
    async with _load_lock:
        if _loaded != model:
            # 1) DRAIN: never evict a model with an in-flight request.
            if _loaded is not None:
                await _idle_forward.wait()   # no new forward can register —
                                             # _begin_forward only runs under _load_lock
            # 2) MIN-RESIDENCY: don't evict a model loaded < MIN_RESIDENCY ago;
            #    wait out the remainder so two agents can't ping-pong-evict.
            wait = MIN_RESIDENCY - (now - _loaded_at)
            if wait > 0: await asyncio.sleep(wait)
            await _switch_and_load(model)    # evict old, load new
            _loaded_at = now
        _begin_forward()                     # register UNDER the lock (atomic vs a switch)
    try:
        yield                                # forward/stream happens here, lock released
    finally:
        _end_forward()
```

`proxy_request` (model-bearing endpoints only) becomes:

```text
async with manager.use_model(model):
    return await forward_to_router(...)
```

### Why this satisfies the requirements

- **Never crashes on a switch:** the switch waits for `_idle_forward`, so no
  request is ever streaming when its model is evicted. The reproduction's
  concurrent-different-model case becomes: B waits for A to finish, then
  switches → both 200.
- **Single model at a time:** unchanged (`--models-max 1`); we only changed
  *when* a switch is allowed.
- **Same-model concurrency stays parallel:** same model ⇒ no switch ⇒ both
  requests register forwards and stream together on the router's slot.
- **No reload thrash:** `MIN_RESIDENCY` (start ~8 s, tune) batches a burst of
  same-model requests and stops the q4↔mtp ping-pong that reloaded every ~11 s.

### Edge cases to handle

- **Queue-wait cap:** a request waiting on drain + residency + load could exceed
  a client timeout. Add an optional max-wait → return `503` + `Retry-After`
  instead of hanging forever (pi retries cleanly). Keep generous by default.
- **Idle unload vs in-flight:** the idle watchdog must also respect
  `_idle_forward` (don't unload mid-stream).
- **`/models/load` direct calls:** route through the same `_load_lock` so an
  explicit client load can't race a switch.

### Companion hardening (separate from the eviction fix)

1. **Orphan reaping on startup:** before `start_server()`, kill any stray
   `llama-server.exe` bound to this stack's `--models-preset`/port. (Today's
   incident showed a stray router on `:8002` we couldn't even kill without
   elevation.)
2. **Windows Job Object:** spawn the router into a Job with
   `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` so it dies with the proxy — no more
   orphans on a hard exit.
3. **(Optional) probe cache:** re-add a minimal TTL/single-flight cache for
   `/models` + `/props` only if pi's probe burst proves to be real load again.

### Acceptance test

Re-run `test_concurrency.py`. **Pass = every cell 200**, including both
concurrent different-model rounds (which must now *queue*, not 500). Confirm
same-model concurrency still runs in parallel (timings overlap, no serial
penalty beyond the single slot).

### Acceptance test result (2026-06-02, PASS)

Implemented via the `use_model()` context manager (drain-before-evict +
`MIN_RESIDENCY = 8.0`). Re-ran `test_concurrency.py` against the fixed proxy on
the isolated `:8011`/`:8012` stack:

| Test | Before (e703092) | After fix |
|---|---|---|
| Sequential different-model | all 200 | all **200** |
| Concurrent different-model, round 0 | A=**500**, B=200 | A=**200**, B=200 |
| Concurrent different-model, round 1 | A=**500**, B=200 | A=**200**, B=200 |
| Concurrent same-model | all 200 | all **200** |

The previously-failing concurrent different-model case now **queues**: in round
0, A completed at 30.5 s and B at 46.6 s — B waited for A to drain, then
switched, then ran. **Zero 5xx.** Same-model concurrency still overlaps. The
mid-stream-eviction 500 is eliminated.

## Phase 3 — Verify empirically (don't assume)

Build a small harness that hits the proxy and records status + timing,
correlated with proxy "Loading model" log lines:

- **Sequential control:** model A → await → model B → await. Expect: clean.
- **Concurrent test:** fire model A and model B simultaneously (overlap).
  Expect on `e703092`: 500s (confirms the latent bug). Expect after Phase 2:
  clean queuing, zero 5xx.

Run against the rolled-back baseline first to confirm the diagnosis, then again
after Phase 2 to confirm the fix. Server can be stopped at will, so test on the
live stack in any window.

---

## Open decisions

1. Min-residency value (default a few seconds; tune to typical agent turn
   length).
2. Re-add probe cache in Phase 2, or wait until flooding is observed?
3. Add the Windows Job Object (router dies with proxy) now or later?
4. Exact reproduction harness shape (Python `asyncio`/`aiohttp` vs parallel
   `curl`).
