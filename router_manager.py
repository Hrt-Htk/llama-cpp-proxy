"""Shared router lifecycle management for proxy.py and embed_proxy.py.

Mechanical extraction from proxy.py (ChatRouterManager) and embed_proxy.py
(RouterManager base). Behaviour-preserving — no logic changes.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time

from aiohttp import ClientError, ClientSession, ClientTimeout

from log_paths import fmt_ts_full
from proxy_base import (
    ProxyConfig,
    ROOT,
    _is_dead_worker_response,
)

# ── Module constants (chat-only — used only by ChatRouterManager methods) ──

MIN_RESIDENCY = 8.0  # minimum seconds a loaded model is kept before an eviction is allowed
# Post-cancel guard: how long to wait for a worker to answer a probe before
# treating it as wedged. Covers a worker still finishing an orphaned prefill;
# if it can't answer in this window we cycle it (the orphan was abandoned anyway).
GUARD_PROBE_TIMEOUT = 30.0


# ── Base class ──────────────────────────────────────────────────────────────

class RouterManager:
    """Mirror of proxy.py's ModelManager, scoped to the embedding preset."""

    LOAD_TIMEOUT = 120
    ROUTER_LABEL = "embed router"  # base IS the embed manager; ChatRouterManager overrides to "router"

    def __init__(self, config: ProxyConfig, session: ClientSession) -> None:
        self.config = config
        self.session = session
        self.process: asyncio.subprocess.Process | None = None
        self._loaded: str | None = None
        self._load_lock = asyncio.Lock()
        self._active = 0
        self._last_activity = time.monotonic()

    @property
    def server_running(self) -> bool:
        return self.process is not None and self.process.returncode is None

    @property
    def model_loaded(self) -> str | None:
        return self._loaded

    @property
    def active_requests(self) -> int:
        return self._active

    def begin_request(self) -> None:
        self._active += 1
        self._last_activity = time.monotonic()

    def end_request(self) -> None:
        self._active = max(0, self._active - 1)
        self._last_activity = time.monotonic()

    async def start_server(self) -> None:
        if self.server_running:
            return
        logging.info(
            "Starting %s on %s:%s | boot_ts=%s",
            self.ROUTER_LABEL,
            self.config.server_host,
            self.config.server_port,
            fmt_ts_full(),
        )
        self.process = await asyncio.create_subprocess_exec(
            *self.config.server_command, cwd=str(ROOT),
            # Pin to the 3090 Ti (GPU 0); hide the 2070 so layers aren't
            # split onto its 8 GB and OOM/slow the embedder.
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
        )
        deadline = time.monotonic() + self.config.boot_timeout
        while time.monotonic() < deadline:
            if self.process.returncode is not None:
                raise RuntimeError(f"router exited during boot: {self.process.returncode}")
            try:
                async with self.session.get(
                    f"{self.config.backend_base_url}/health",
                    timeout=ClientTimeout(total=5),
                ) as r:
                    if r.status == 200:
                        logging.info("%s is ready", self.ROUTER_LABEL)
                        return
            except (ClientError, asyncio.TimeoutError):
                pass
            await asyncio.sleep(self.config.health_poll_interval)
        raise TimeoutError(f"{self.ROUTER_LABEL} did not become healthy in {self.config.boot_timeout}s")

    async def stop_server(self) -> None:
        if not self.server_running:
            return
        logging.info("Stopping %s", self.ROUTER_LABEL)
        try:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self.process.kill()
                await asyncio.wait_for(self.process.wait(), timeout=5)
        except ProcessLookupError:
            pass
        self.process = None
        self._loaded = None

    async def ensure_loaded(self, model: str) -> None:
        async with self._load_lock:
            await self._load_locked(model)

    async def _load_locked(self, model: str) -> None:
        if self._loaded == model:
            return
        current_status = await self._status(model)
        if current_status == "loaded":
            self._loaded = model
            return
        logging.info("Loading model: %s", model)
        async with self._post_json("/models/load", {"model": model}) as r:
            if r.status >= 400:
                body = await r.text()
                if "already running" in body:
                    self._loaded = model
                    return
                raise RuntimeError(f"load returned {r.status}: {body}")
        deadline = time.monotonic() + self.LOAD_TIMEOUT
        if await self._poll_until_loaded(model, deadline):
            self._loaded = model
            logging.info("Model loaded: %s", model)

    async def unload(self, reason: str) -> None:
        if self._loaded is None:
            return
        model = self._loaded
        logging.info("Unloading %s (%s)", model, reason)
        try:
            async with self._post_json("/models/unload", {"model": model}) as r:
                if r.status >= 400:
                    logging.warning("unload returned %s", r.status)
        except ClientError as e:
            logging.warning("unload error: %s", e)
        self._loaded = None

    async def _status(self, model: str) -> str:
        async with self._get_json("/v1/models") as r:
            data = await r.json()
        for entry in data.get("data", []):
            if entry.get("id") == model:
                status = entry.get("status") or {}
                if isinstance(status, dict):
                    return status.get("value", "unknown")
                return str(status)
        return "unknown"

    async def unload_if_idle(self) -> None:
        if self._active != 0 or self._loaded is None:
            return
        idle = time.monotonic() - self._last_activity
        if idle >= self.config.idle_timeout:
            await self.unload(f"idle for {int(idle)}s")

    # ── HTTP helpers ──────────────────────────────────────────────────────

    def _auth_headers(self, content_type: str = "application/json") -> dict:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": content_type,
        }

    def _post_json(self, path: str, json_data: dict,
                   timeout: float | None = None):
        url = f"{self.config.backend_base_url}{path}"
        kwargs: dict = {"headers": self._auth_headers(), "json": json_data}
        if timeout is not None:
            kwargs["timeout"] = ClientTimeout(total=timeout)
        return self.session.post(url, **kwargs)

    def _get_json(self, path: str, timeout: float | None = None):
        url = f"{self.config.backend_base_url}{path}"
        kwargs: dict = {"headers": {"Authorization": f"Bearer {self.config.api_key}"}}
        if timeout is not None:
            kwargs["timeout"] = ClientTimeout(total=timeout)
        return self.session.get(url, **kwargs)

    async def _poll_until_loaded(self, model: str, deadline: float) -> bool:
        """Poll *model* status until loaded, failed, or *deadline* expires.

        Returns True on success (model reached "loaded"). Raises RuntimeError
        on "failed" and TimeoutError on deadline expiry.
        """
        while time.monotonic() < deadline:
            status = await self._status(model)
            if status == "loaded":
                return True
            if status == "failed":
                raise RuntimeError(f"model {model} failed to load")
            await asyncio.sleep(0.5)
        raise TimeoutError(f"model {model} did not load in {self.LOAD_TIMEOUT}s")


# ── Chat subclass ───────────────────────────────────────────────────────────

class ChatRouterManager(RouterManager):
    """Owns the long-lived router process and on-demand model load/unload.

    The router process starts with the proxy and dies with it. Models are
    loaded on first request and unloaded by the idle watchdog — the router
    itself stays up so the cloudflared tunnel never breaks.
    """

    LOAD_TIMEOUT = 300
    ROUTER_LABEL = "router"

    def __init__(self, config: ProxyConfig, session: ClientSession) -> None:
        super().__init__(config, session)
        self._forwarding: int = 0  # count of requests currently streaming
        self._idle_forward: asyncio.Event = asyncio.Event()
        self._idle_forward.set()  # set == 0 in-flight (idle)
        self._loaded_at: float = 0.0  # monotonic time the current model finished loading
        # Set when a request is aborted mid-generation (client cancel/disconnect).
        # llama-server has an unfixed cancel→next-request desync that wedges/crashes
        # the worker (ggml-org/llama.cpp#20921), so the NEXT model request probes the
        # worker first instead of detonating on it. Guarded by _guard_lock.
        self._worker_suspect: bool = False
        self._guard_lock = asyncio.Lock()

    def _begin_forward(self) -> None:
        """Register a new in-flight streaming request."""
        self._forwarding += 1
        self._idle_forward.clear()  # not idle while a request is streaming

    def _end_forward(self) -> None:
        """Deregister a completed streaming request."""
        self._forwarding = max(0, self._forwarding - 1)
        if self._forwarding == 0:
            self._idle_forward.set()  # signal idle to any waiting switcher

    async def _load_locked(self, model: str) -> None:
        """Load *model* into the router. Caller MUST already hold ``_load_lock``.

        Syncs cached ``_loaded`` state from the router before issuing a load
        to avoid spurious "already running" 400s, then polls until the model
        reports ``loaded`` or a timeout expires. Sets ``_loaded_at`` on every
        successful load so the residency window starts fresh.
        """
        if self._loaded == model:
            return
        # Sync state: the router may already have this model loaded (e.g. a
        # direct /models/load call through the proxy). Check before issuing
        # another load — otherwise the router returns 400 "already running".
        current_status = await self._status(model)
        if current_status == "loaded":
            self._loaded = model
            self._loaded_at = time.monotonic()
            return
        logging.info("Loading model: %s", model)
        async with self._post_json("/models/load", {"model": model}) as r:
            if r.status >= 400:
                body = await r.text()
                if "already running" in body:
                    self._loaded = model
                    self._loaded_at = time.monotonic()
                    return
                raise RuntimeError(f"load returned {r.status}: {body}")
        deadline = time.monotonic() + self.LOAD_TIMEOUT
        if await self._poll_until_loaded(model, deadline):
            self._loaded = model
            self._loaded_at = time.monotonic()
            logging.info("Model loaded: %s", model)

    @contextlib.asynccontextmanager
    async def use_model(self, model: str):
        """Async context manager that ensures *model* is loaded for the duration
        of a streaming forward, preventing mid-stream eviction.

        Acquiring ``_load_lock`` for a switch:
        1. DRAIN — waits for all currently-streaming requests to finish before
           evicting the old model (``_idle_forward`` is only set when
           ``_forwarding == 0``).
        2. MIN-RESIDENCY — after a fresh load, waits out the remainder of
           ``MIN_RESIDENCY`` seconds before allowing another eviction, preventing
           rapid ping-pong reloads by concurrent agents on different models.

        ``_begin_forward()`` is called while the lock is still held so the
        counter increment is atomic with respect to any concurrent switcher.
        """
        async with self._load_lock:
            if self._loaded != model:
                # DRAIN: never evict a model that has an in-flight request.
                # _begin_forward() only runs under _load_lock, so no new
                # forward can sneak in while we are deciding to switch.
                if self._loaded is not None:
                    await self._idle_forward.wait()
                # MIN-RESIDENCY: don't evict a model loaded less than
                # MIN_RESIDENCY seconds ago; wait out the remainder.
                wait = MIN_RESIDENCY - (time.monotonic() - self._loaded_at)
                if wait > 0:
                    await asyncio.sleep(wait)
                await self._load_locked(model)
            self._begin_forward()  # register UNDER the lock — atomic vs a switch
        try:
            yield
        finally:
            self._end_forward()

        # ── recover_worker helpers ────────────────────────────────────────────

    async def _recover_check_peer(
        self, model: str, dead_detected_at: float,
    ) -> bool:
        """Return True if a peer coroutine already recovered *model*."""
        if self._loaded == model and self._loaded_at > dead_detected_at:
            logging.info(
                "[recover_worker] %s already recovered by peer (loaded_at=%.3f > detected=%.3f)",
                model, self._loaded_at, dead_detected_at,
            )
            return True
        return False

    async def _recover_unload(
        self, model: str, url_unload: str, auth_headers: dict,
    ) -> None:
        """Best-effort unload POST — force the router to tear down the dead worker entry."""
        try:
            async with self.session.post(
                url_unload, headers=auth_headers, json={"model": model},
                timeout=ClientTimeout(total=10),
            ) as r:
                body_text = await r.text()
                if r.status >= 400:
                    logging.warning("[recover_worker] unload returned %s: %s", r.status, body_text)
                else:
                    logging.info("[recover_worker] unload OK for %s", model)
        except (ClientError, asyncio.TimeoutError) as e:
            logging.warning("[recover_worker] unload error (ignored): %s", e)

    async def _recover_wait_unloaded(self, model: str) -> None:
        """Poll until the router confirms *model* is no longer loaded (15 s deadline)."""
        unload_deadline = time.monotonic() + 15
        while time.monotonic() < unload_deadline:
            try:
                status = await self._status(model)
                if status != "loaded":
                    logging.info("[recover_worker] router confirms %s is %s", model, status)
                    break
            except Exception:
                pass
            await asyncio.sleep(0.5)
        else:
            logging.warning(
                "[recover_worker] router still shows loaded after 15s — proceeding anyway"
            )

    async def _recover_load_once(
        self, model: str, load_attempt: int, max_load_attempts: int,
        load_url: str, auth_headers: dict,
    ) -> bool | str:
        """Execute one load attempt with rapid-poll and stability check.

        Returns True on success (worker stable), False on terminal failure
        (POST error or status "failed"), "retry_now" on queued-exit flash
        or mid-load death (retry immediately, no unload/sleep),
        "retry_after_unload" on poll timeout (caller should unload + sleep).
        """
        logging.info(
            "[recover_worker] load attempt %d/%d for %s",
            load_attempt, max_load_attempts, model,
        )
        try:
            async with self.session.post(
                load_url, headers=auth_headers, json={"model": model},
                timeout=ClientTimeout(total=10),
            ) as r:
                body_text = await r.text()
                if r.status >= 400 and "already running" not in body_text:
                    logging.warning("[recover_worker] load returned %s: %s", r.status, body_text)
        except (ClientError, asyncio.TimeoutError) as e:
            logging.error("[recover_worker] load POST failed: %s", e)
            return False

        # Poll until status leaves "loading" (either loaded or unloaded)
        poll_deadline = time.monotonic() + 120  # per_load_timeout
        prev_status = ""
        seen_loading = False
        while time.monotonic() < poll_deadline:
            try:
                status = await self._status(model)
            except Exception:
                await asyncio.sleep(0.5)
                continue
            if status != prev_status:
                logging.info("[recover_worker] %s status: %s", model, status)
                prev_status = status
            if status == "loading":
                seen_loading = True
            if status == "loaded":
                # Give the router 1.5s to process any pending exit signal
                # before declaring victory.
                await asyncio.sleep(1.5)
                status2 = await self._status(model)
                if status2 == "loaded":
                    self._loaded = model
                    self._loaded_at = time.monotonic()
                    logging.info(
                        "[recover_worker] worker stable for %s (attempt %d)",
                        model, load_attempt,
                    )
                    return True
                logging.warning(
                    "[recover_worker] fresh worker for %s exited immediately "
                    "(status after 1.5s: %s, attempt %d/%d)",
                    model, status2, load_attempt, max_load_attempts,
                )
                return "retry_now"  # queued exit consumed — try next load attempt
            if status == "failed":
                logging.error("[recover_worker] model %s failed to load", model)
                return False
            if status == "unloaded" and seen_loading:
                # The worker started loading but was then stopped before ever
                # reaching "loaded" — a queued exit signal consumed it (the
                # router force-kills it after its ~10s stop timeout) or it
                # crashed mid-load. For a slow-loading model the kill lands
                # before "loaded" is ever observed, so the loaded→flash case
                # above never fires. Retry the next load immediately instead
                # of spinning here for the full per_load_timeout.
                logging.warning(
                    "[recover_worker] fresh worker for %s died during load "
                    "(status: unloaded, attempt %d/%d) — retrying",
                    model, load_attempt, max_load_attempts,
                )
                return "retry_now"  # queued exit consumed — try next load attempt
            await asyncio.sleep(0.25)

        logging.error(
            "[recover_worker] timed out waiting for %s to load (attempt %d/%d)",
            model, load_attempt, max_load_attempts,
        )
        return "retry_after_unload"  # timeout — caller may retry unload before next attempt

    async def _recover_retry_unload(
        self, model: str, url_unload: str, auth_headers: dict,
    ) -> None:
        """Retry unload after a timed-out load attempt, then pause 2 s."""
        logging.info("[recover_worker] retrying unload before next load attempt")
        try:
            async with self.session.post(
                url_unload, headers=auth_headers, json={"model": model},
                timeout=ClientTimeout(total=10),
            ) as r:
                body_text = await r.text()
                if r.status >= 400:
                    logging.warning(
                        "[recover_worker] retry-unload returned %s: %s",
                        r.status, body_text,
                    )
                else:
                    logging.info(
                        "[recover_worker] retry-unload OK for %s", model
                    )
        except (ClientError, asyncio.TimeoutError) as e:
            logging.warning(
                "[recover_worker] retry-unload error (ignored): %s", e
            )
        # Brief pause to let the router settle before the next load.
        await asyncio.sleep(2)

    async def recover_worker(self, model: str, dead_detected_at: float) -> bool:
        """Force-cycle the router worker for *model* after a dead-worker 500.

        Acquires ``_load_lock`` so concurrent callers serialise. Only the
        first caller actually cycles unload/load; subsequent callers whose
        ``dead_detected_at`` is before a recent ``_loaded_at`` know a peer
        already completed recovery and return True immediately.

        NOTE: We do NOT check the router's ``_status()`` to skip the cycle —
        the router can report ``loaded`` while the worker child is already
        dead. We always force an unload+reload on the first caller.

        Returns True when the worker is confirmed loaded, False on timeout/
        error so the caller can return 503.
        """
        async with self._load_lock:
            if await self._recover_check_peer(model, dead_detected_at):
                return True

            logging.warning(
                "[recover_worker] cycling unload/load for dead worker %s", model
            )
            self._loaded = None  # invalidate proxy cache immediately

            # Unload — force the router to tear down the dead worker entry.
            # Best-effort: the router may error if it can't contact the child,
            # but we continue to the load step regardless.
            url_unload = f"{self.config.backend_base_url}/models/unload"
            auth_headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            }
            await self._recover_unload(model, url_unload, auth_headers)

            # Wait for the router to reflect the unloaded state so that
            # _load_locked doesn't see "loaded" and short-circuit.
            await self._recover_wait_unloaded(model)

            # Reload — spawn fresh worker(s), draining any buffered exit
            # signals the router queued during unload. Each exit signal
            # consumes exactly one fresh worker; we loop until the worker
            # stays alive for a full second after reporting "loaded".
            #
            # We do NOT use _load_locked here because its poll
            # loops until "loaded" is observed — if a queued exit fires
            # between the worker reporting ready and our next poll (< 500 ms),
            # the status flips to "unloaded" and the loop spins for 300 s.
            # Instead we implement a custom load + rapid-poll that detects
            # the loaded→unloaded flash and retries the load immediately.
            load_url = f"{self.config.backend_base_url}/models/load"
            max_load_attempts = 4

            for load_attempt in range(1, max_load_attempts + 1):
                result = await self._recover_load_once(
                    model, load_attempt, max_load_attempts,
                    load_url, auth_headers,
                )
                if result is True:
                    return True
                if result is False:
                    return False
                # retryable: "retry_now" → next attempt immediately;
                #            "retry_after_unload" → unload + 2s pause first
                #            (only if attempts remain)
                if result == "retry_after_unload" and load_attempt < max_load_attempts:
                    await self._recover_retry_unload(
                        model, url_unload, auth_headers,
                    )

            logging.error("[recover_worker] all %d load attempts failed for %s", max_load_attempts, model)
            return False

    def mark_worker_suspect(self) -> None:
        """Flag the worker as possibly desynced after a mid-generation client
        abort (cancel/disconnect). The next model request will probe before use.
        See _worker_suspect for the upstream bug this guards against."""
        self._worker_suspect = True

    async def _probe_worker(self, model: str) -> bool:
        """Send a minimal generation to detect a wedged/crashed worker after a
        cancel. Returns True if the worker answers normally; False if it returns
        a dead-worker error, errors out, or times out (a busy-finishing-an-orphan
        or genuinely-wedged worker both warrant a recovery cycle)."""
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
            "stream": False,
        }
        try:
            async with self._post_json(
                "/v1/chat/completions", payload,
                timeout=GUARD_PROBE_TIMEOUT,
            ) as r:
                body = await r.read()
                if _is_dead_worker_response(r.status, body):
                    return False
                return r.status < 500
        except (ClientError, asyncio.TimeoutError) as e:
            logging.warning("[guard] probe error for %s: %s", model, e)
            return False

    async def guard_after_cancel(self, model: str) -> None:
        """If a prior request was aborted mid-generation, probe the worker once
        before the next request uses it. Recover proactively if the probe shows
        it wedged/dead — so the next real request lands on a clean worker instead
        of triggering the upstream cancel→next-request crash."""
        if not self._worker_suspect:
            return
        async with self._guard_lock:
            if not self._worker_suspect:
                return  # a concurrent caller already handled this episode
            # Only the currently-loaded worker can be the wedged one. If a different
            # model (or nothing) is loaded, use_model will spawn a fresh worker
            # anyway, so there is nothing to probe — just clear the flag.
            if self._loaded != model:
                self._worker_suspect = False
                return
            logging.info("[guard] worker suspect after cancel — probing %s", model)
            healthy = await self._probe_worker(model)
            if healthy:
                logging.info("[guard] worker %s healthy after cancel", model)
            else:
                logging.warning(
                    "[guard] worker %s wedged/dead after cancel — recovering", model
                )
                await self.recover_worker(model, time.monotonic())
            self._worker_suspect = False

    async def unload_if_idle(self) -> None:
        # Never unload while a streaming forward is in progress.
        if self._active != 0 or self._forwarding != 0 or self._loaded is None:
            return
        idle = time.monotonic() - self._last_activity
        if idle >= self.config.idle_timeout:
            await self.unload(f"idle for {int(idle)}s")
