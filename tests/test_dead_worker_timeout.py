"""Integration tests for hung-worker detection in the chat proxy.

Reproduces the 2026-07-19 incident: the llama.cpp router accepted a completion
request, forwarded it to a *hung* model-worker child, and never sent a response.
proxy.py's `_do_forward` awaited the response with ``timeout=None``, so it hung
indefinitely — its dead-worker recovery (which only fires on a 5xx body) was
never reached.

These tests drive the real ``proxy_request`` handler against a scripted mock
upstream (standing in for the llama.cpp router) in three states:

* ``hung``  — accept, never respond (the incident). Expect: detect within the
              first-byte timeout -> recover_worker -> 503, not an infinite hang.
* ``slow``  — delay response headers (a cold model load), then answer 200.
              Expect: NOT tripped (the guardrail that separates "slow" from
              "hung").
* ``healthy`` — immediate 200 (sanity).

No GPU / real router needed. stdlib unittest, aiohttp test utils.
"""
from __future__ import annotations

import asyncio
import contextlib
import unittest

from aiohttp import ClientSession, web
from aiohttp.test_utils import TestClient, TestServer

import proxy_request_handlers
from proxy_request_handlers import proxy_request
from proxy_base import ProxyConfig

# Small so the GREEN test finishes fast; production default is 90s.
TEST_FIRST_BYTE_TIMEOUT = 1.0


class _FakeManager:
    """Minimal stand-in for ChatRouterManager exposing only what proxy_request
    touches. ``recover_worker`` records calls instead of cycling a real worker."""

    def __init__(self, config: ProxyConfig, recover_result: bool = True) -> None:
        self.config = config
        self.recover_result = recover_result
        self.recover_calls = 0
        self.suspect_marks = 0

    def begin_request(self) -> None: ...
    def end_request(self) -> None: ...
    def mark_worker_suspect(self) -> None:
        self.suspect_marks += 1

    async def guard_after_cancel(self, model: str) -> None: ...

    @contextlib.asynccontextmanager
    async def use_model(self, model: str):
        yield

    async def recover_worker(self, model: str, dead_detected_at: float) -> bool:
        self.recover_calls += 1
        return self.recover_result


class HungWorkerTimeoutTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        # Force a short first-byte timeout for the duration of the test.
        self._orig_timeout = proxy_request_handlers.FIRST_BYTE_TIMEOUT
        proxy_request_handlers.FIRST_BYTE_TIMEOUT = TEST_FIRST_BYTE_TIMEOUT

        # --- mock upstream (stands in for the llama.cpp router) ---
        # State lives on the test, not the app, so tests can flip it after the
        # server has started without mutating a running Application.
        self.state = {"mode": "healthy", "delay": 0.0}
        self.upstream_app = web.Application()
        self.upstream_app.router.add_route(
            "*", "/{tail:.*}", self._upstream_handler
        )
        self.upstream = TestServer(self.upstream_app)
        await self.upstream.start_server()

        self.config = ProxyConfig(
            proxy_host="127.0.0.1", proxy_port=0,
            server_host=self.upstream.host, server_port=self.upstream.port,
            idle_timeout=600, idle_check_interval=30, health_poll_interval=0.5,
            boot_timeout=120, default_model="test-model", api_key="",
        )
        self.manager = _FakeManager(self.config)
        self.session = ClientSession()

        # --- proxy app under test ---
        self.proxy_app = web.Application()
        self.proxy_app["manager"] = self.manager
        self.proxy_app["session"] = self.session
        self.proxy_app["chat_logger"] = None
        self.proxy_app.router.add_route("*", "/{tail:.*}", proxy_request)
        self.client = TestClient(TestServer(self.proxy_app))
        await self.client.start_server()

    async def asyncTearDown(self) -> None:
        proxy_request_handlers.FIRST_BYTE_TIMEOUT = self._orig_timeout
        await self.client.close()
        await self.session.close()
        await self.upstream.close()

    async def _upstream_handler(self, request: web.Request) -> web.Response:
        mode = self.state["mode"]
        if mode == "hung":
            await asyncio.sleep(30)  # never respond within the test window
            return web.json_response({"ok": True})
        if mode == "slow":
            await asyncio.sleep(self.state["delay"])
            return web.json_response({"ok": True})
        return web.json_response({"ok": True})

    async def _post_completion(self):
        return await self.client.post(
            "/v1/chat/completions",
            json={"model": "test-model",
                  "messages": [{"role": "user", "content": "hi"}]},
        )

    async def test_hung_worker_is_detected_and_recovered(self) -> None:
        """A worker that accepts but never responds must not hang the proxy:
        it should be detected within the first-byte timeout, trigger a worker
        recovery, and (recovery not fixing the mock) return 503."""
        self.state["mode"] = "hung"
        # Wrap in a generous bound: without the fix the proxy hangs forever and
        # this raises TimeoutError -> test fails (the RED signal).
        resp = await asyncio.wait_for(self._post_completion(), timeout=8)
        self.assertEqual(resp.status, 503)
        self.assertGreaterEqual(
            self.manager.recover_calls, 1,
            "hung worker should have triggered recover_worker",
        )

    async def test_slow_cold_load_is_not_tripped(self) -> None:
        """Delayed response headers (a cold model load) shorter than the
        first-byte timeout must pass through untouched — the guardrail that a
        slow worker is not mistaken for a hung one."""
        self.state["mode"] = "slow"
        self.state["delay"] = TEST_FIRST_BYTE_TIMEOUT * 0.4
        resp = await asyncio.wait_for(self._post_completion(), timeout=8)
        self.assertEqual(resp.status, 200)
        self.assertEqual(self.manager.recover_calls, 0)

    async def test_healthy_passthrough(self) -> None:
        self.state["mode"] = "healthy"
        resp = await asyncio.wait_for(self._post_completion(), timeout=8)
        self.assertEqual(resp.status, 200)
        self.assertEqual(self.manager.recover_calls, 0)


if __name__ == "__main__":
    unittest.main()
