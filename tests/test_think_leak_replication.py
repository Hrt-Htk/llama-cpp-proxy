"""Replication test harness for think-tag leaks and thinking-only stalls (issue #8).

Two categories:
  Category 1 — REPLICATION tests: replicate real production failure streams
                and assert the corrected behaviour (implemented in chat_logger.py).
  Category 2 — CONTROL tests: guard against regressions / over-eager fixes
                (legit tag mentions, fences, pure stalls, normal streams).
"""

from __future__ import annotations

import asyncio
import json
import re
import unittest

from tests._helpers import (
    ChunkedContent,
    FakeChatLogger,
    FakeUpstream,
    _drive,
    _sse_event,
    _split_sse_events,
)


# ---------------------------------------------------------------------------
# Shared fixture builders
# ---------------------------------------------------------------------------

def _base_chunk(id_: str = "req1") -> dict:
    """Minimal chat.completion.chunk envelope."""
    return {
        "id": id_,
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
    }


def _reasoning_chunk(text: str, id_: str = "req1") -> bytes:
    """SSE event with a reasoning_content delta."""
    obj = _base_chunk(id_)
    obj["choices"][0]["delta"]["reasoning_content"] = text
    return _sse_event(obj)


def _content_chunk(text: str, id_: str = "req1") -> bytes:
    """SSE event with a content delta."""
    obj = _base_chunk(id_)
    obj["choices"][0]["delta"]["content"] = text
    return _sse_event(obj)


def _finish_chunk(finish_reason: str = "stop", id_: str = "req1") -> bytes:
    """SSE event with finish_reason and empty delta."""
    obj = _base_chunk(id_)
    obj["choices"][0]["delta"] = {}
    obj["choices"][0]["finish_reason"] = finish_reason
    return _sse_event(obj)


def _collect_reasoning(events: list) -> str:
    """Concatenate all reasoning_content deltas from *events*."""
    parts: list[str] = []
    for e in events:
        rc = e.get("choices", [{}])[0].get("delta", {}).get("reasoning_content")
        if rc:
            parts.append(rc)
    return "".join(parts)


def _collect_content(events: list) -> str:
    """Concatenate all content deltas from *events*."""
    parts: list[str] = []
    for e in events:
        c = e.get("choices", [{}])[0].get("delta", {}).get("content")
        if c:
            parts.append(c)
    return "".join(parts)


def _has_tool_calls(events: list) -> list:
    """Return all tool_calls entries found across events."""
    results: list = []
    for e in events:
        tcs = e.get("choices", [{}])[0].get("delta", {}).get("tool_calls")
        if tcs:
            results.extend(tcs)
    return results


def _finish_reasons(events: list) -> list:
    """Return all non-None finish_reason values."""
    reasons: list[str] = []
    for e in events:
        fr = e.get("choices", [{}])[0].get("finish_reason")
        if fr:
            reasons.append(fr)
    return reasons


# ===================================================================
# Category 1: REPLICATION tests (real production failure streams)
# ===================================================================

class TestThinkLeakReplication(unittest.TestCase):
    """Tests replicating real production think-tag leaks and thinking-only
    stalls, asserting the corrected stream behaviour."""

    # --- Shape A: double </think> (leak) ---

    def test_shapeA_draft_rerouted_to_reasoning(self) -> None:
        """Content carrying text before a standalone </think> line should be
        rerouted: pre-tag text → reasoning_content, tag dropped,
        post-tag text → normal content.

        Real production seam (session 8d62feae):
          reasoning ends "...ically check the abort signal while waiting for user input.\\n"
          content = "\\nNow I have the complete picture. The freeze is a **deadlock**:\\n</think>\\n\\nGot it — this is a deadlock. Let me update the issue draft with the precise root cause.\\n\\n"
        """
        reasoning_tail = (
            "Let me think through this carefully. "
            "I need to check the abort signal while waiting for user input.\n"
        )
        # The leaked content: draft answer + standalone </think> + final answer
        leaked_content = (
            "\nNow I have the complete picture. The freeze is a **deadlock**:\n"
            "</think>\n\n"
            "Got it — this is a deadlock. Let me update the issue draft with the precise root cause.\n\n"
        )

        chunks = [
            _reasoning_chunk(reasoning_tail),
            _content_chunk(leaked_content),
            _finish_chunk(),
        ]

        raw, _ = asyncio.run(_drive(chunks))
        events = _split_sse_events(raw.encode())

        # Desired: reasoning = original thinking + draft (text before </think>)
        combined_reasoning = _collect_reasoning(events)
        expected_reasoning = reasoning_tail + "\nNow I have the complete picture. The freeze is a **deadlock**:\n"
        self.assertEqual(combined_reasoning, expected_reasoning)

        # Desired: content = final answer only, NO literal </think>
        combined_content = _collect_content(events)
        expected_content = "Got it — this is a deadlock. Let me update the issue draft with the precise root cause.\n\n"
        self.assertEqual(combined_content, expected_content)
        self.assertNotIn("</think>", combined_content)

    # --- Shape B: mention consumed as real tag (leak) ---

    def test_shapeB_mention_leak_rerouted(self) -> None:
        """Same standalone-line rule applied to Shape B: everything in content
        up to and including the standalone </think> line goes to reasoning;
        the rest is content.

        Real production seam: the model's thinking discusses `</think>` inside
        backticks. llama-server's parser closes reasoning at the mention and
        eats the mention text.
        """
        # Reasoning ends mid-code-span (parser ate the rest)
        reasoning_part = (
            "I need to check if there's a "
        )
        # Content starts where reasoning was cut, has the real close tag
        # appear literally later, then the final answer
        content_part = (
            "`</think>` variant (without brackets) that might be what they mean\n"
            "Let me extract them.\n"
            "</think>\n\n"
            "Here are the extracted items.\n\n"
        )

        chunks = [
            _reasoning_chunk(reasoning_part),
            _content_chunk(content_part),
            _finish_chunk(),
        ]

        raw, _ = asyncio.run(_drive(chunks))
        events = _split_sse_events(raw.encode())

        # Desired: reasoning = original + everything up to standalone </think>
        combined_reasoning = _collect_reasoning(events)
        expected_reasoning = (
            reasoning_part
            + "`</think>` variant (without brackets) that might be what they mean\n"
            "Let me extract them.\n"
        )
        self.assertEqual(combined_reasoning, expected_reasoning)

        # Desired: content = text after the standalone </think> only
        combined_content = _collect_content(events)
        expected_content = "Here are the extracted items.\n\n"
        self.assertEqual(combined_content, expected_content)
        self.assertNotIn("</think>", combined_content)

    # --- Shape A split across chunks ---

    def test_shapeA_tag_split_across_chunks(self) -> None:
        """Same as test_shapeA but the literal </think> is split across
        content deltas and SSE event boundaries."""
        reasoning_text = "Analyzing the problem step by step.\n"

        # The </think> tag is split: "</th" in one delta, "ink>" in the next
        draft_before_tag = "\nDraft answer here.\n</th"
        after_tag = "ink>\n\nFinal answer.\n\n"

        # Build as one blob then feed through ChunkedContent with tiny slices
        blob = b""
        blob += _reasoning_chunk(reasoning_text)
        blob += _content_chunk(draft_before_tag)
        blob += _content_chunk(after_tag)
        blob += _finish_chunk()

        content = ChunkedContent(blob, slice_size=11)
        logger = FakeChatLogger()
        from chat_logger import SSEChunkLogger
        wrapped = SSEChunkLogger(FakeUpstream([]), logger)
        wrapped._wrapped.content = content

        out: list[bytes] = []
        while True:
            piece = asyncio.run(wrapped.readany())
            if not piece:
                break
            out.append(piece)

        raw = b"".join(out).decode("utf-8", errors="replace")
        events = _split_sse_events(raw.encode())

        # Desired: reasoning = thinking + draft (before </think>)
        combined_reasoning = _collect_reasoning(events)
        expected_reasoning = reasoning_text + "\nDraft answer here.\n"
        self.assertEqual(combined_reasoning, expected_reasoning)

        # Desired: content = final answer only, no </think>
        combined_content = _collect_content(events)
        expected_content = "Final answer.\n\n"
        self.assertEqual(combined_content, expected_content)
        self.assertNotIn("</think>", combined_content)

    # --- Shape C1: trapped tool-call rescued ---
    # NOTE: this is NOT expectedFailure — the existing rescue state machine
    # in SSEChunkLogger already handles tool-call XML (<tool_call>...</tool_call>) in
    # reasoning_content.  It passes today and must keep passing.

    def test_shapeC1_trapped_toolcall_rescued(self) -> None:
        """Reasoning-only stream whose reasoning ends with a complete
        <tool_call>...</tool_call> block and finish_reason=stop with empty content: the
        client should receive a synthesized tool_call and the XML
        stripped from reasoning_content tail.

        37 of 94 real stalls look like this — the model wrote its tool
        call inside the unclosed think block.
        """
        reasoning_prose = "Let me check the directory listing.\n"
        xml_block = (
            "<tool_call>\n"
            "<function=bash>\n"
            "<parameter=command>\n"
            "ls C:\\some\\path\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )

        chunks = [
            _reasoning_chunk(reasoning_prose),
            _reasoning_chunk(xml_block),
            _finish_chunk(),
        ]

        raw, _ = asyncio.run(_drive(chunks))
        events = _split_sse_events(raw.encode())

        # Desired: tool call is synthesized
        tc_list = _has_tool_calls(events)
        self.assertEqual(len(tc_list), 1)
        tc = tc_list[0]
        self.assertEqual(tc["function"]["name"], "bash")
        args = json.loads(tc["function"]["arguments"])
        self.assertEqual(args["command"], "ls C:\\some\\path")

        # Desired: reasoning = prose only, XML stripped
        combined_reasoning = _collect_reasoning(events)
        self.assertEqual(combined_reasoning, reasoning_prose)
        self.assertNotIn("<tool_call>", combined_reasoning)
        self.assertNotIn("</tool_call>", combined_reasoning)
        self.assertNotIn("<function", combined_reasoning)

        # Desired: finish_reason rewritten to "tool_calls"
        reasons = _finish_reasons(events)
        self.assertNotIn("stop", reasons)
        self.assertIn("tool_calls", reasons)

    # --- Shape C1 variant: truncated closing tag ---

    def test_shapeC1_truncated_toolcall_rescued(self) -> None:
        """Reasoning-only stream whose reasoning ends with a tool-call XML
        block whose closing </tool_call> tag is truncated (missing final '>')
        and finish_reason=stop with empty content: the tool call should still
        be synthesized, the XML (including truncated tail) stripped from
        reasoning_content, and finish_reason rewritten to tool_calls.

        Real production seam — the model stalled inside an unclosed think
        block and the stream ended with the tool-call XML cut off mid
        closing tag.
        """
        reasoning_prose = "Let me list the docs directory.\n"
        xml_block = (
            "<tool_call>\n"
            "<function=bash>\n"
            "<parameter=command>\n"
            'ls "C:\\Users\\HTK\\AppData\\Roaming\\npm\\node_modules\\@earendil-works\\pi-coding-agent\\docs"\n'
            "</parameter>\n"
            "</function>\n"
            "</tool_call"  # truncated — missing closing '>'
        )

        chunks = [
            _reasoning_chunk(reasoning_prose),
            _reasoning_chunk(xml_block),
            _finish_chunk(),
        ]

        raw, _ = asyncio.run(_drive(chunks))
        events = _split_sse_events(raw.encode())

        # Desired: tool call is synthesized despite truncated closing tag
        tc_list = _has_tool_calls(events)
        self.assertEqual(len(tc_list), 1)
        tc = tc_list[0]
        self.assertEqual(tc["function"]["name"], "bash")
        args = json.loads(tc["function"]["arguments"])
        self.assertIn("command", args)
        self.assertIn("ls", args["command"])
        self.assertIn("pi-coding-agent", args["command"])

        # Desired: reasoning = prose only, XML stripped (including truncated tail)
        combined_reasoning = _collect_reasoning(events)
        self.assertEqual(combined_reasoning, reasoning_prose)
        self.assertNotIn("<function", combined_reasoning)
        self.assertNotIn("</function>", combined_reasoning)
        self.assertNotIn("</tool_call", combined_reasoning)

        # Desired: finish_reason rewritten to "tool_calls"
        reasons = _finish_reasons(events)
        self.assertNotIn("stop", reasons)
        self.assertIn("tool_calls", reasons)


# ===================================================================
# Category 2: CONTROL tests (must pass today AND after any fix)
# ===================================================================

class ThinkLeakControls(unittest.TestCase):
    """Control tests that must stay green regardless of any think-tag fix."""

    # --- 5. Legit inline mention untouched ---

    def test_legit_inline_mention_untouched(self) -> None:
        """A normal stream whose content legitimately mentions the tag inline
        in backticks with no standalone-line tag: must pass through byte-identical.
        Guard against over-eager fixes."""
        reasoning_text = "I've analyzed the problem.\n"
        content_text = (
            "Some models leak `</think>` into output. "
            "This is just a mention of the closing tag, not an actual one.\n"
        )

        chunks = [
            _reasoning_chunk(reasoning_text),
            _content_chunk(content_text),
            _finish_chunk(),
        ]

        raw, _ = asyncio.run(_drive(chunks))
        events = _split_sse_events(raw.encode())

        # Content must be byte-identical
        combined_content = _collect_content(events)
        self.assertEqual(combined_content, content_text)

        # Reasoning must be intact
        combined_reasoning = _collect_reasoning(events)
        self.assertEqual(combined_reasoning, reasoning_text)

    # --- 6. Legit fenced block mention untouched ---

    def test_legit_fenced_block_mention_untouched(self) -> None:
        """Content containing a fenced code block in which </think> sits alone
        on a line must pass through unmodified. The standalone-line rule must
        not fire inside a code fence.

        NOTE: today's passthrough makes this green; a naive fix would break it.
        """
        reasoning_text = "Here is an example.\n"
        content_text = (
            "Example of the bug:\n"
            "```text\n"
            "thinking...\n"
            "</think>\n"
            "```\n"
            "That was the example.\n"
        )

        chunks = [
            _reasoning_chunk(reasoning_text),
            _content_chunk(content_text),
            _finish_chunk(),
        ]

        raw, _ = asyncio.run(_drive(chunks))
        events = _split_sse_events(raw.encode())

        combined_content = _collect_content(events)
        self.assertEqual(combined_content, content_text)

        combined_reasoning = _collect_reasoning(events)
        self.assertEqual(combined_reasoning, reasoning_text)

    # --- 7. Shape C2: pure stall passthrough ---

    def test_shapeC2_pure_stall_passthrough(self) -> None:
        """Reasoning-only stream with NO tool-call XML: passes through
        unchanged (we do not invent content). Green today, must stay green."""
        reasoning_text = (
            "Let me think about this problem. "
            "There are several approaches we could take. "
            "I need more information to proceed.\n"
        )

        chunks = [
            _reasoning_chunk(reasoning_text),
            _finish_chunk(),
        ]

        raw, _ = asyncio.run(_drive(chunks))
        events = _split_sse_events(raw.encode())

        # Reasoning passes through intact
        combined_reasoning = _collect_reasoning(events)
        self.assertEqual(combined_reasoning, reasoning_text)

        # No content invented
        combined_content = _collect_content(events)
        self.assertEqual(combined_content, "")

        # No tool calls synthesized
        tc_list = _has_tool_calls(events)
        self.assertEqual(len(tc_list), 0)

        # finish_reason stays "stop"
        reasons = _finish_reasons(events)
        self.assertIn("stop", reasons)

    # --- 8. Normal stream with reasoning untouched ---

    def test_normal_stream_with_reasoning_untouched(self) -> None:
        """Ordinary reasoning + content + native tool_calls stream passes
        through unchanged (extension of existing passthrough test but with
        reasoning deltas)."""
        chunks = [
            _sse_event({
                "id": "req1", "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }),
            _reasoning_chunk("Let me think about this carefully.\n"),
            _reasoning_chunk("I have a plan.\n"),
            _content_chunk("Here is the answer.\n"),
            _content_chunk("It should be correct.\n"),
            _sse_event({
                "id": "req1", "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {
                    "tool_calls": [{"index": 0, "id": "call_xyz", "type": "function",
                                   "function": {"name": "write", "arguments": '{"path":"/out"}'}}],
                }, "finish_reason": None}],
            }),
            _finish_chunk(),
        ]

        raw, _ = asyncio.run(_drive(chunks))
        events = _split_sse_events(raw.encode())

        combined_reasoning = _collect_reasoning(events)
        self.assertEqual(combined_reasoning, "Let me think about this carefully.\nI have a plan.\n")

        combined_content = _collect_content(events)
        self.assertEqual(combined_content, "Here is the answer.\nIt should be correct.\n")

        # Tool call preserved
        tc_list = _has_tool_calls(events)
        self.assertEqual(len(tc_list), 1)
        self.assertEqual(tc_list[0]["function"]["name"], "write")

        # finish_reason stays "stop"
        reasons = _finish_reasons(events)
        self.assertIn("stop", reasons)


if __name__ == "__main__":
    unittest.main()
