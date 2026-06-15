"""Tests for tool-call XML rescue from reasoning_content deltas."""

from __future__ import annotations

import asyncio
import json
import unittest

from chat_logger import SSEChunkLogger


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeContent:
    """Yields preset byte chunks then b"" (EOF)."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)
        self._idx = 0

    async def readany(self) -> bytes:
        if self._idx >= len(self._chunks):
            return b""
        chunk = self._chunks[self._idx]
        self._idx += 1
        return chunk


class ChunkedContent:
    """Takes a single bytes blob and yields it in fixed-size slices, then b"" (EOF).
    Simulates real network chunk boundaries that split SSE events mid-stream."""

    def __init__(self, blob: bytes, slice_size: int = 13) -> None:
        self._blob = blob
        self._slice_size = slice_size
        self._pos = 0

    async def readany(self) -> bytes:
        if self._pos >= len(self._blob):
            return b""
        chunk = self._blob[self._pos : self._pos + self._slice_size]
        self._pos += self._slice_size
        return chunk


class FakeUpstream:
    """Minimal upstream response with .content and .headers."""

    def __init__(self, chunks: list[bytes]) -> None:
        self.content = FakeContent(chunks)
        self.headers = {}


class FakeChatLogger:
    """Records log_response calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []

    async def log_response(self, data: str, is_done: bool) -> None:
        self.calls.append((data, is_done))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sse_event(payload: dict, extra_headers: str = "") -> bytes:
    """Build a raw SSE event byte string from a JSON-serialisable dict."""
    body = json.dumps(payload, ensure_ascii=False)
    return f"{extra_headers}data: {body}\r\n\r\n".encode()


def _split_sse_events(raw: bytes) -> list[dict]:
    """Split raw SSE bytes into parsed JSON payloads (data: lines only)."""
    events: list[dict] = []
    # Normalise line endings
    text = raw.decode("utf-8", errors="replace")
    for block in text.split("\r\n\r\n"):
        if not block.strip():
            continue
        for line in block.splitlines():
            if line.startswith("data:"):
                payload = line[5:].strip()
                if payload:
                    try:
                        events.append(json.loads(payload))
                    except json.JSONDecodeError:
                        pass  # skip [DONE] and other non-JSON payloads
    return events


def _make_logger(chunks: list[bytes]) -> SSEChunkLogger:
    return SSEChunkLogger(FakeUpstream(chunks), FakeChatLogger())


async def _drive(chunks: list[bytes]) -> tuple[str, FakeChatLogger]:
    """Run SSEChunkLogger through all chunks, return (combined output, logger)."""
    logger = FakeChatLogger()
    wrapped = SSEChunkLogger(FakeUpstream(chunks), logger)
    out: list[bytes] = []
    while True:
        piece = await wrapped.readany()
        if not piece:
            break
        out.append(piece)
    return b"".join(out).decode("utf-8", errors="replace"), logger


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestToolCallRescue(unittest.TestCase):

    # 1. Passthrough — normal stream, nothing rescued
    def test_passthrough_normal_stream(self) -> None:
        """Real tool_calls delta is preserved; finish_reason stays 'stop'."""
        chunks = [
            _sse_event({
                "id": "req1", "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }),
            _sse_event({
                "id": "req1", "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"reasoning_content": "Let me think..."}, "finish_reason": None}],
            }),
            _sse_event({
                "id": "req1", "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}],
            }),
            _sse_event({
                "id": "req1", "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {
                    "tool_calls": [{"index": 0, "id": "call_abc", "type": "function",
                                   "function": {"name": "read", "arguments": '{"path":"/x"}'}}],
                }, "finish_reason": None}],
            }),
            _sse_event({
                "id": "req1", "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }),
        ]
        raw, _ = asyncio.run(_drive(chunks))
        events = _split_sse_events(raw.encode())

        # Find the tool_calls event
        tc_events = [e for e in events if e.get("choices", [{}])[0].get("delta", {}).get("tool_calls")]
        self.assertEqual(len(tc_events), 1)
        tc = tc_events[0]["choices"][0]["delta"]["tool_calls"][0]
        self.assertEqual(tc["function"]["name"], "read")

        # finish_reason should stay "stop" (nothing rescued)
        stop_events = [e for e in events if e.get("choices", [{}])[0].get("finish_reason") == "stop"]
        self.assertEqual(len(stop_events), 1)

    # 2. Stuck call split across chunks AND events
    def test_rescue_split_across_chunks(self) -> None:
        """Tool-call XML split across multiple SSE events/chunks is rescued."""
        # Build the XML tool call split across several events
        xml_parts = [
            "<tool_call>\n",
            "<function=r",
            "ead>\n",
            "<parameter=path>\n",
            "/x\n",
            "</parameter>\n",
            "</function>\n",
            "</tool_call>",
        ]
        chunks: list[bytes] = []
        for part in xml_parts:
            chunks.append(_sse_event({
                "id": "req1", "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"reasoning_content": part}, "finish_reason": None}],
            }))
        chunks.append(_sse_event({
            "id": "req1", "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }))

        raw, logger = asyncio.run(_drive(chunks))
        events = _split_sse_events(raw.encode())

        # Should have a tool_calls event
        tc_events = [e for e in events if e.get("choices", [{}])[0].get("delta", {}).get("tool_calls")]
        self.assertEqual(len(tc_events), 1)
        tc = tc_events[0]["choices"][0]["delta"]["tool_calls"][0]
        self.assertEqual(tc["function"]["name"], "read")
        args = json.loads(tc["function"]["arguments"])
        self.assertEqual(args, {"path": "/x"})

        # No reasoning_content should contain any XML markers
        for e in events:
            rc = e.get("choices", [{}])[0].get("delta", {}).get("reasoning_content")
            if rc:
                self.assertNotIn("<function", rc)
                self.assertNotIn("<tool_call>", rc)

        # Combined reasoning must also be clean (catches leaked </tool_call> etc.)
        combined_reasoning = "".join(
            e["choices"][0]["delta"].get("reasoning_content", "")
            for e in events
            if e.get("choices", [{}])[0].get("delta", {}).get("reasoning_content")
        )
        self.assertNotIn("<function", combined_reasoning)
        self.assertNotIn("</function>", combined_reasoning)
        self.assertNotIn("<tool_call>", combined_reasoning)
        self.assertNotIn("</tool_call>", combined_reasoning)

        # finish_reason rewritten to "tool_calls"
        stop_events = [e for e in events if e.get("choices", [{}])[0].get("finish_reason") == "stop"]
        self.assertEqual(len(stop_events), 0)
        tc_finish = [e for e in events if e.get("choices", [{}])[0].get("finish_reason") == "tool_calls"]
        self.assertEqual(len(tc_finish), 1)

    # 3. Prose around call
    def test_prose_around_call(self) -> None:
        """Prose before and after XML block is forwarded; XML stripped."""
        xml_block = "<tool_call>\n<function=read>\n<parameter=path>\n/x\n</parameter>\n</function>\n</tool_call>"
        chunks = [
            _sse_event({
                "id": "req1", "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"reasoning_content": "Let me read it."}, "finish_reason": None}],
            }),
            _sse_event({
                "id": "req1", "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"reasoning_content": xml_block}, "finish_reason": None}],
            }),
            _sse_event({
                "id": "req1", "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"reasoning_content": " done"}, "finish_reason": None}],
            }),
            _sse_event({
                "id": "req1", "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }),
        ]
        raw, _ = asyncio.run(_drive(chunks))
        events = _split_sse_events(raw.encode())

        # Collect all reasoning_content
        reasonings = [
            e["choices"][0]["delta"].get("reasoning_content")
            for e in events
            if e.get("choices", [{}])[0].get("delta", {}).get("reasoning_content")
        ]
        combined_reasoning = "".join(reasonings)
        self.assertEqual(combined_reasoning, "Let me read it. done")
        self.assertNotIn("<function", combined_reasoning)
        self.assertNotIn("<tool_call>", combined_reasoning)
        self.assertNotIn("</function>", combined_reasoning)
        self.assertNotIn("</tool_call>", combined_reasoning)

        # Tool call rescued
        tc_events = [e for e in events if e.get("choices", [{}])[0].get("delta", {}).get("tool_calls")]
        self.assertEqual(len(tc_events), 1)

    # 4. No wrapper (no <tool_call>)
    def test_no_wrapper(self) -> None:
        """<function=...> without <tool_call> is rescued."""
        xml = "<function=bash><parameter=command>ls</parameter></function>"
        chunks = [
            _sse_event({
                "id": "req1", "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"reasoning_content": xml}, "finish_reason": None}],
            }),
            _sse_event({
                "id": "req1", "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }),
        ]
        raw, _ = asyncio.run(_drive(chunks))
        events = _split_sse_events(raw.encode())

        tc_events = [e for e in events if e.get("choices", [{}])[0].get("delta", {}).get("tool_calls")]
        self.assertEqual(len(tc_events), 1)
        tc = tc_events[0]["choices"][0]["delta"]["tool_calls"][0]
        self.assertEqual(tc["function"]["name"], "bash")
        args = json.loads(tc["function"]["arguments"])
        self.assertEqual(args, {"command": "ls"})

    # 5. Bare empty <tool_call></tool_call>
    def test_bare_empty(self) -> None:
        """Bare empty <tool_call></tool_call> is stripped; no tool_calls event; finish_reason stays stop."""
        chunks = [
            _sse_event({
                "id": "req1", "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"reasoning_content": "<tool_call></tool_call>"}, "finish_reason": None}],
            }),
            _sse_event({
                "id": "req1", "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }),
        ]
        raw, _ = asyncio.run(_drive(chunks))
        events = _split_sse_events(raw.encode())

        tc_events = [e for e in events if e.get("choices", [{}])[0].get("delta", {}).get("tool_calls")]
        self.assertEqual(len(tc_events), 0)

        # finish_reason should stay "stop"
        stop_events = [e for e in events if e.get("choices", [{}])[0].get("finish_reason") == "stop"]
        self.assertEqual(len(stop_events), 1)

    # 6. Arg coercion
    def test_arg_coercion(self) -> None:
        """Numeric args are coerced to int; string args stay strings."""
        xml = "<tool_call>\n<function=read>\n<parameter=path>\n/some/path\n</parameter>\n<parameter=limit>200</parameter>\n</function>\n</tool_call>"
        chunks = [
            _sse_event({
                "id": "req1", "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"reasoning_content": xml}, "finish_reason": None}],
            }),
            _sse_event({
                "id": "req1", "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }),
        ]
        raw, _ = asyncio.run(_drive(chunks))
        events = _split_sse_events(raw.encode())

        tc_events = [e for e in events if e.get("choices", [{}])[0].get("delta", {}).get("tool_calls")]
        self.assertEqual(len(tc_events), 1)
        tc = tc_events[0]["choices"][0]["delta"]["tool_calls"][0]
        args = json.loads(tc["function"]["arguments"])
        self.assertEqual(args["limit"], 200)
        self.assertIsInstance(args["limit"], int)
        self.assertEqual(args["path"], "/some/path")
        self.assertIsInstance(args["path"], str)


    # 7. Two tool calls in one turn
    def test_multiple_calls_one_turn(self) -> None:
        """Two tool-call blocks back-to-back are both rescued with correct indices."""
        xml_block = (
            "</tool_call><function=read><parameter=path>/a</parameter></function></tool_call>\n"
            "</tool_call><function=bash><parameter=command>ls</parameter></function></tool_call>"
        )
        chunks = [
            _sse_event({
                "id": "req1", "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"reasoning_content": xml_block}, "finish_reason": None}],
            }),
            _sse_event({
                "id": "req1", "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }),
        ]
        raw, _ = asyncio.run(_drive(chunks))
        events = _split_sse_events(raw.encode())

        tc_events = [e for e in events if e.get("choices", [{}])[0].get("delta", {}).get("tool_calls")]
        self.assertEqual(len(tc_events), 2)

        # First call: read with path=/a, index 0
        tc0 = tc_events[0]["choices"][0]["delta"]["tool_calls"][0]
        self.assertEqual(tc0["index"], 0)
        self.assertEqual(tc0["function"]["name"], "read")
        args0 = json.loads(tc0["function"]["arguments"])
        self.assertEqual(args0, {"path": "/a"})

        # Second call: bash with command=ls, index 1
        tc1 = tc_events[1]["choices"][0]["delta"]["tool_calls"][0]
        self.assertEqual(tc1["index"], 1)
        self.assertEqual(tc1["function"]["name"], "bash")
        args1 = json.loads(tc1["function"]["arguments"])
        self.assertEqual(args1, {"command": "ls"})

        # finish_reason rewritten to "tool_calls"
        stop_events = [e for e in events if e.get("choices", [{}])[0].get("finish_reason") == "stop"]
        self.assertEqual(len(stop_events), 0)
        tc_finish = [e for e in events if e.get("choices", [{}])[0].get("finish_reason") == "tool_calls"]
        self.assertEqual(len(tc_finish), 1)


    # 8. Sub-event chunk boundaries — normal stream fed in tiny slices
    def test_subevent_chunk_boundaries(self) -> None:
        """A normal multi-event stream assembled into one blob, fed in 13-byte
        slices, still produces correct output (readany loops until complete
        events or true EOF)."""
        blob = b""
        blob += _sse_event({"id": "r1", "object": "chat.completion.chunk",
                            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
        blob += _sse_event({"id": "r1", "object": "chat.completion.chunk",
                            "choices": [{"index": 0, "delta": {"reasoning_content": "Let me "}, "finish_reason": None}]})
        blob += _sse_event({"id": "r1", "object": "chat.completion.chunk",
                            "choices": [{"index": 0, "delta": {"reasoning_content": "think"}, "finish_reason": None}]})
        blob += _sse_event({"id": "r1", "object": "chat.completion.chunk",
                            "choices": [{"index": 0, "delta": {"reasoning_content": " here."}, "finish_reason": None}]})
        blob += _sse_event({"id": "r1", "object": "chat.completion.chunk",
                            "choices": [{"index": 0, "delta": {"content": "The "}, "finish_reason": None}]})
        blob += _sse_event({"id": "r1", "object": "chat.completion.chunk",
                            "choices": [{"index": 0, "delta": {"content": "answer."}, "finish_reason": None}]})
        blob += _sse_event({"id": "r1", "object": "chat.completion.chunk",
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
        blob += b"data: [DONE]\n\n"

        content = ChunkedContent(blob, slice_size=13)
        wrapped = SSEChunkLogger(
            FakeUpstream([]),  # headers unused; we swap content
            FakeChatLogger(),
        )
        # Replace the wrapped content with our chunked source
        wrapped._wrapped.content = content

        out: list[bytes] = []
        while True:
            piece = asyncio.run(wrapped.readany())
            if not piece:
                break
            out.append(piece)

        raw = b"".join(out).decode("utf-8", errors="replace")
        self.assertTrue(len(raw) > 0, "output should be non-empty")

        events = _split_sse_events(raw.encode())
        reasoning_parts = [
            e["choices"][0]["delta"].get("reasoning_content", "")
            for e in events
            if e.get("choices", [{}])[0].get("delta", {}).get("reasoning_content")
        ]
        self.assertEqual("".join(reasoning_parts), "Let me think here.")

        content_parts = [
            e["choices"][0]["delta"].get("content", "")
            for e in events
            if e.get("choices", [{}])[0].get("delta", {}).get("content")
        ]
        self.assertEqual("".join(content_parts), "The answer.")

        stop_events = [e for e in events if e.get("choices", [{}])[0].get("finish_reason") == "stop"]
        self.assertEqual(len(stop_events), 1)

        self.assertIn("[DONE]", raw)

    # 9. Rescue survives arbitrary sub-event chunk boundaries
    def test_rescue_subevent_chunks(self) -> None:
        """Stuck-tool-call XML assembled into one blob and fed in 13-byte slices
        is still rescued (name "read", args {"path":"/x"}), finish_reason
        rewritten to "tool_calls"."""
        xml_block = "<function=read>\n<parameter=path>\n/x\n</parameter>\n</function>"
        blob = b""
        blob += _sse_event({"id": "r1", "object": "chat.completion.chunk",
                            "choices": [{"index": 0, "delta": {"reasoning_content": f"<tool_call>\n{xml_block}\n</tool_call>"}, "finish_reason": None}]})
        blob += _sse_event({"id": "r1", "object": "chat.completion.chunk",
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
        blob += b"data: [DONE]\n\n"

        content = ChunkedContent(blob, slice_size=13)
        wrapped = SSEChunkLogger(
            FakeUpstream([]),
            FakeChatLogger(),
        )
        wrapped._wrapped.content = content

        out: list[bytes] = []
        while True:
            piece = asyncio.run(wrapped.readany())
            if not piece:
                break
            out.append(piece)

        raw = b"".join(out).decode("utf-8", errors="replace")
        self.assertTrue(len(raw) > 0, "output should be non-empty")

        events = _split_sse_events(raw.encode())

        # Tool call rescued
        tc_events = [e for e in events if e.get("choices", [{}])[0].get("delta", {}).get("tool_calls")]
        self.assertEqual(len(tc_events), 1)
        tc = tc_events[0]["choices"][0]["delta"]["tool_calls"][0]
        self.assertEqual(tc["function"]["name"], "read")
        args = json.loads(tc["function"]["arguments"])
        self.assertEqual(args, {"path": "/x"})

        # finish_reason rewritten to "tool_calls"
        stop_events = [e for e in events if e.get("choices", [{}])[0].get("finish_reason") == "stop"]
        self.assertEqual(len(stop_events), 0)
        tc_finish = [e for e in events if e.get("choices", [{}])[0].get("finish_reason") == "tool_calls"]
        self.assertEqual(len(tc_finish), 1)


if __name__ == "__main__":
    unittest.main()
