"""Shared test helpers for SSEChunkLogger unit tests.

Extracted from tests/test_toolcall_rescue.py to avoid duplication across
test modules.
"""

from __future__ import annotations

import asyncio
import json
from typing import List

from chat_logger import SSEChunkLogger


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeContent:
    """Yields preset byte chunks then b"" (EOF)."""

    def __init__(self, chunks: List[bytes]) -> None:
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

    def __init__(self, chunks: List[bytes]) -> None:
        self.content = FakeContent(chunks)
        self.headers = {}


class FakeChatLogger:
    """Records log_response calls."""

    def __init__(self) -> None:
        self.calls: List[tuple] = []

    async def log_response(self, data: str, is_done: bool) -> None:
        self.calls.append((data, is_done))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sse_event(payload: dict, extra_headers: str = "") -> bytes:
    """Build a raw SSE event byte string from a JSON-serialisable dict."""
    body = json.dumps(payload, ensure_ascii=False)
    return f"{extra_headers}data: {body}\r\n\r\n".encode()


def _split_sse_events(raw: bytes) -> List[dict]:
    """Split raw SSE bytes into parsed JSON payloads (data: lines only)."""
    events: List[dict] = []
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


def _make_logger(chunks: List[bytes]) -> SSEChunkLogger:
    return SSEChunkLogger(FakeUpstream(chunks), FakeChatLogger())


async def _drive(chunks: List[bytes]) -> tuple:
    """Run SSEChunkLogger through all chunks, return (combined output str, logger)."""
    logger = FakeChatLogger()
    wrapped = SSEChunkLogger(FakeUpstream(chunks), logger)
    out: List[bytes] = []
    while True:
        piece = await wrapped.readany()
        if not piece:
            break
        out.append(piece)
    return b"".join(out).decode("utf-8", errors="replace"), logger
